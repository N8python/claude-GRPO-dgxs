"""Integration check for the Alice<->Bob on-policy contract. Run this first.

Three things can be silently wrong in a split GRPO setup, and none of them
make training crash -- they make it train on the wrong thing while the loss
curve looks entirely reasonable:

  A. weight sync is a no-op (name mapping wrong, load_weights matched nothing)
  B. the sampler mirror is wrong, so the trainer scores a different
     distribution than vLLM sampled from
  C. Bob and Alice are holding different base checkpoints

This script catches all three in about a minute:

  TEST 1  agreement -- push the adapter, sample, recompute the sampled tokens'
          logprobs on Alice, and measure mean |logp_torch - logp_vllm|.
          bf16-vs-fp32 noise is O(1e-3). Anything O(1) means B or C.
  TEST 2  liveness -- perturb the LoRA, push again, sample the same prompt
          with the same seed, and require the output to CHANGE. A no-op sync
          (A) passes TEST 1 trivially, because both sides agree about the
          base model; only this test separates them.

Usage:
    python check_sync.py --adapter-path models/sft-lora
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lora_torch import load_adapter, lora_state_dict
from rollout_client import RolloutClient
from task import make_prompt
from train_grpo import token_logps

WORDS = ["pluto", "orange", "traffic", "wonderful", "silver", "planet"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Llama-3.2-1B-Instruct-bf16")
    ap.add_argument("--rollout-url", default="http://192.168.100.11:8300")
    ap.add_argument("--adapter-path", default="models/sft-lora")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.25,
                    help="mean |dlogp| in nats/token above which TEST 1 fails")
    ap.add_argument("--attn", default="sdpa")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    client = RolloutClient(args.rollout_url)
    client.wait_ready()
    info = client.info()
    print(f"Bob: {info}\n")

    problems = []
    if info["model"] != args.model:
        problems.append(f"Bob serves {info['model']}, trainer holds {args.model}")
    if not str(info.get("logprobs_mode", "")).startswith("processed"):
        problems.append(
            f"logprobs_mode={info.get('logprobs_mode')!r} -- TIS needs "
            f"processed_logprobs (post-temperature, post-truncation)"
        )
    if problems:
        for p in problems:
            print(f"CONFIG PROBLEM: {p}")
        raise SystemExit(1)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=args.attn
    ).to(device)
    model.config.use_cache = False
    lora_parameters = load_adapter(model, args.adapter_path)
    model.to(device).eval()
    scale = lora_parameters["scale"]

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": make_prompt(w)}],
            tokenize=True, return_dict=False, add_generation_prompt=True,
        )
        for w in WORDS
    ]

    # ---- TEST 1: do the two machines agree about the policy? -------------
    print("TEST 1  agreement")
    client.update_weights(lora_state_dict(model), scale=scale, step=0)
    resp = client.generate(
        prompts, n=args.group_size, temperature=args.temp, top_p=args.top_p,
        top_k=args.top_k, max_tokens=args.max_tokens, seed=1234,
    )

    diffs = []
    pad_id = tokenizer.eos_token_id
    with torch.no_grad():
        for i, (comp, blp) in enumerate(zip(resp["completions"], resp["logprobs"])):
            if not comp:
                continue
            p_ids = prompts[i // args.group_size]
            x = torch.tensor([p_ids + comp], dtype=torch.long, device=device)
            logp = token_logps(
                model, x, args.temp, args.top_p, args.top_k, 2048
            )[0]
            start = len(p_ids) - 1
            for j, ref in enumerate(blp):
                if ref is not None:
                    diffs.append(abs(logp[start + j].item() - ref))

    d = torch.tensor(diffs)
    mean_d = d.mean().item()
    print(f"  tokens compared : {d.numel()}")
    print(f"  mean |dlogp|    : {mean_d:.5f} nats/token")
    print(f"  median          : {d.median().item():.5f}")
    print(f"  p99             : {d.quantile(0.99).item():.5f}")
    print(f"  max             : {d.max().item():.5f}")
    over_cap = (d > torch.log(torch.tensor(2.0))).float().mean().item()
    print(f"  tokens a TIS cap of 2.0 would clip: {100 * over_cap:.3f}%")
    test1 = mean_d <= args.threshold
    print(f"  -> {'PASS' if test1 else 'FAIL'} (threshold {args.threshold})\n")

    # ---- TEST 2: does a weight push actually reach the sampler? ----------
    # TEST 1 alone cannot distinguish "sync works" from "sync is a no-op and
    # both sides are running the untouched base", because in the latter case
    # they still agree with each other.
    print("TEST 2  liveness")
    base_out = client.generate(
        prompts[:1], n=1, temperature=0.0, max_tokens=48, logprobs=False
    )["completions"][0]

    modules = [m for m in model.modules() if hasattr(m, "lora_b")]
    saved = [m.lora_b.detach().clone() for m in modules]
    with torch.no_grad():
        for m in modules:
            m.lora_b.add_(torch.randn_like(m.lora_b) * 0.05)
    client.update_weights(lora_state_dict(model), scale=scale, step=-1)
    perturbed_out = client.generate(
        prompts[:1], n=1, temperature=0.0, max_tokens=48, logprobs=False
    )["completions"][0]

    with torch.no_grad():
        for m, s in zip(modules, saved):
            m.lora_b.copy_(s)
    client.update_weights(lora_state_dict(model), scale=scale, step=-1)
    restored_out = client.generate(
        prompts[:1], n=1, temperature=0.0, max_tokens=48, logprobs=False
    )["completions"][0]

    changed = perturbed_out != base_out
    restored = restored_out == base_out
    print(f"  greedy output changed after perturbation : {changed}")
    print(f"  greedy output restored after undo        : {restored}")
    test2 = changed and restored
    if not changed:
        print("  -> FAIL: pushing materially different weights did not change "
              "Bob's output. The sync is a no-op; check Bob's log for the "
              "'loaded' count on the first /update_weights.")
    elif not restored:
        print("  -> FAIL: output did not return to baseline. Bob's weight state "
              "is drifting -- the base snapshot may be getting overwritten.")
    else:
        print("  -> PASS\n")

    print("=" * 60)
    if test1 and test2:
        print("READY: Alice and Bob agree, and weight sync is live.")
        print(f"Expect `dlogp` ~{mean_d:.4f} in the training log; a sudden jump "
              f"means something changed underneath you.")
    else:
        print("NOT READY -- fix the failures above before training.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
