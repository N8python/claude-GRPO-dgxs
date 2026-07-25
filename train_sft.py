"""SFT on the 100 synthesized reversal traces. Runs on Alice, single GPU.

Torch port of the MLX reference. Same recipe: masked next-token loss on the
assistant tokens only, rank-8 LoRA at scale 32/rank, lr 1e-4, 3 epochs.

Its job is to teach the FORMAT. GRPO teaches the skill -- that split is what
makes the RL delta visible, so resist the urge to make this stage better.

Usage:
    python train_sft.py                    # reference recipe
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from lora_torch import apply_lora, save_adapter_dir
from mlx_adamw import MLXAdamW

PAD_MULTIPLE = 32


def encode_example(tokenizer, messages, max_seq_len):
    full = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False)
    prompt = tokenizer.apply_chat_template(
        messages[:1], tokenize=True, return_dict=False, add_generation_prompt=True
    )
    assert full[: len(prompt)] == prompt, "prompt is not a prefix of the full chat"
    assert len(full) <= max_seq_len, f"example too long: {len(full)}"
    return full, len(prompt)


def make_batches(examples, batch_size, pad_id, device):
    """Length-sorted batches padded to a multiple of PAD_MULTIPLE.

    Right padding + causal attention + a loss mask means no attention_mask is
    needed: real tokens never attend to trailing pads, and pad positions are
    zeroed out of the loss.
    """
    examples = sorted(examples, key=lambda e: len(e[0]))
    batches = []
    for i in range(0, len(examples), batch_size):
        chunk = examples[i : i + batch_size]
        max_len = max(len(t) for t, _ in chunk)
        max_len = ((max_len + PAD_MULTIPLE - 1) // PAD_MULTIPLE) * PAD_MULTIPLE
        x = torch.tensor(
            [t + [pad_id] * (max_len - len(t)) for t, _ in chunk],
            dtype=torch.long, device=device,
        )
        # target-space weights: target t (= token t+1) trains iff it is a
        # completion token, i.e. prompt_len <= t+1 < len(tokens)
        w = torch.tensor(
            [
                [1.0 if p <= t + 1 < len(toks) else 0.0 for t in range(max_len - 1)]
                for toks, p in chunk
            ],
            dtype=torch.float32, device=device,
        )
        batches.append((x, w))
    return batches


def lr_lambda_factory(warmup, total_steps):
    """MLX's join_schedules(linear_schedule(0, lr, warmup), cosine_decay(...)).

    Note step/warmup, not (step+1)/warmup: MLX evaluates the schedule at its
    step counter BEFORE incrementing, so the first update genuinely runs at
    lr 0. Matching that keeps the reference's step-for-step behaviour.
    """
    def fn(step):
        if warmup > 0 and step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Llama-3.2-1B-Instruct-bf16")
    ap.add_argument("--data", default="data/sft_train.jsonl")
    ap.add_argument("--out", default="models/sft-lora")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="tuned for LoRA at scale 32/rank; the ~10x-of-full-FT "
                    "heuristic applies HERE and nowhere in the RL phase")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-scale", type=float, default=None, help="default 32/rank")
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    ap.add_argument("--adam-bias-correction", action="store_true",
                    help="use bias-corrected Adam (torch's default). OFF to "
                    "match mlx.optimizers.AdamW, which is what --lr was tuned "
                    "against; turning it on needs a ~5x higher --lr")
    args = ap.parse_args()

    torch.manual_seed(args.seed)  # pins the LoRA init
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=args.attn
    ).to(device)
    model.config.use_cache = False

    lora_parameters = apply_lora(
        model, args.lora_rank, args.lora_scale, args.lora_dropout
    )
    model.to(device)

    rows = [json.loads(l) for l in open(args.data)]
    examples = [
        encode_example(tokenizer, r["messages"], args.max_seq_len) for r in rows
    ]
    pad_id = tokenizer.eos_token_id
    batches = make_batches(examples, args.batch_size, pad_id, device)
    total_steps = len(batches) * args.epochs
    print(f"{len(examples)} examples, {len(batches)} batches/epoch, "
          f"{total_steps} total steps")

    params = [p for p in model.parameters() if p.requires_grad]
    # NOT torch.optim.AdamW -- see mlx_adamw.py. The reference's lr was tuned
    # against an un-bias-corrected optimizer and means something ~5x weaker
    # under torch's.
    opt = MLXAdamW(params, lr=args.lr, weight_decay=0.01,
                   bias_correction=args.adam_bias_correction)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda_factory(args.warmup, total_steps)
    )

    rng = random.Random(args.seed)
    model.train()
    step, t0 = 0, time.time()
    for epoch in range(args.epochs):
        order = list(range(len(batches)))
        rng.shuffle(order)
        for i in order:
            x, w = batches[i]
            logits = model(input_ids=x[:, :-1]).logits.float()
            ce = F.cross_entropy(
                logits.transpose(1, 2), x[:, 1:], reduction="none"
            )
            loss = (ce * w).sum() / w.sum()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % 10 == 0 or step == total_steps:
                peak = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0
                print(f"epoch {epoch} step {step}/{total_steps} "
                      f"loss {loss.item():.4f} "
                      f"({step / (time.time() - t0):.2f} steps/s, "
                      f"peak {peak:.1f} GB)", flush=True)

    save_adapter_dir(args.out, model, lora_parameters, args.model)
    print(f"saved adapter dir -> {args.out}")


if __name__ == "__main__":
    main()
