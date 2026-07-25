"""Evaluate a policy on the word-reversal task. Decoding runs on Bob's vLLM.

Alice only tokenizes and scores here -- the model is never loaded locally, so
this is fast and can run while nothing else holds GPU memory.

Evaluation protocol (unchanged from the reference): base/SFT policies decode
greedily; a GRPO policy was optimized as a temperature sampler, so evaluate it
at its training temperature for an on-policy measurement.

Usage:
    python evaluate.py --adapter-path models/sft-lora  --out preds_sft.jsonl
    python evaluate.py --adapter-path models/grpo-lora --temp 0.7 --out preds_grpo.jsonl
    python evaluate.py --adapter-path none                       # base model
"""

import argparse
import json
import time
from pathlib import Path

from transformers import AutoTokenizer

from rollout_client import RolloutClient
from task import levenshtein, make_prompt, parse_answer


def push_adapter(client, adapter_path):
    """Ship an adapter dir to Bob without loading the model on Alice."""
    from safetensors.torch import load_file

    adapter_path = Path(adapter_path)
    cfg = json.loads((adapter_path / "adapter_config.json").read_text())
    scale = cfg["lora_parameters"]["scale"]
    state = load_file(str(adapter_path / "adapters.safetensors"))
    res = client.update_weights(state, scale=scale)
    print(f"pushed {adapter_path} to Bob "
          f"(scale {scale}, {res['bytes'] / 1e6:.1f} MB, {res['seconds']:.2f}s, "
          f"weight_version {res['weight_version']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Llama-3.2-1B-Instruct-bf16")
    ap.add_argument("--rollout-url", default="http://192.168.100.11:8300")
    ap.add_argument("--adapter-path", default=None,
                    help="adapter dir to push to Bob; 'none' resets to base")
    ap.add_argument("--words", default="data/eval_words.txt")
    ap.add_argument("--out", default=None, help="write per-word predictions jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=448,
                    help="448 is the reference's eval budget -- keep it to stay "
                    "comparable with the published leaderboard numbers")
    ap.add_argument("--temp", type=float, default=0.0,
                    help="0 = greedy. Evaluate GRPO policies at their training "
                    "temperature (--temp 0.7) for an on-policy measurement")
    ap.add_argument("--top-p", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    client = RolloutClient(args.rollout_url)
    client.wait_ready()

    if args.adapter_path and args.adapter_path.lower() != "none":
        push_adapter(client, args.adapter_path)
    elif args.adapter_path and args.adapter_path.lower() == "none":
        try:
            client.reset_weights()
            print("reset Bob to the base checkpoint")
        except Exception as e:
            print(f"(reset skipped: {e})")

    words = Path(args.words).read_text().split()
    if args.limit:
        words = words[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": make_prompt(w)}],
            tokenize=True, return_dict=False, add_generation_prompt=True,
        )
        for w in words
    ]

    t = time.time()
    resp = client.generate(
        prompts,
        n=1,
        temperature=args.temp,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        seed=args.seed if args.temp > 0 else None,
        logprobs=False,
    )
    dt = time.time() - t
    texts = [tokenizer.decode(c) for c in resp["completions"]]

    n = len(words)
    tagged = exact = lev_sum = 0
    rows = []
    for word, text in zip(words, texts):
        target = word[::-1]
        parsed = parse_answer(text)
        row = {"word": word, "target": target, "parsed": parsed}
        if parsed is not None:
            tagged += 1
            norm = parsed.lower()
            row["lev"] = levenshtein(norm, target)
            lev_sum += row["lev"]
            if norm == target:
                exact += 1
        row["exact"] = parsed is not None and parsed.lower() == target
        rows.append(row)

    adapter = f" + {args.adapter_path}" if args.adapter_path else ""
    print(f"\nmodel: {args.model}{adapter}  "
          f"({'greedy' if args.temp == 0 else f'temp={args.temp}'})")
    print(f"n={n}  wall={dt:.0f}s  ({resp['n_tokens'] / dt:.0f} tok/s aggregate)")
    print(f"tag rate:    {tagged}/{n} = {100 * tagged / n:.1f}%")
    print(f"exact match: {exact}/{n} = {100 * exact / n:.1f}%")
    if tagged:
        print(f"avg levenshtein (over {tagged} tagged): {lev_sum / tagged:.2f}")

    if args.out:
        with open(args.out, "w") as f:
            for row, text in zip(rows, texts):
                row["response"] = text
                f.write(json.dumps(row) + "\n")
        print(f"predictions -> {args.out}")


if __name__ == "__main__":
    main()
