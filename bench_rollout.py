"""Benchmark Bob's generation throughput in a GRPO-shaped workload.

Exists to compare weight-delivery modes fairly. `merge-addmm` folds the LoRA
into the weights so decode runs at base speed but every push rewrites the
model; `vllm-lora` makes the push nearly free but pays punica overhead on
every generated token. Which wins is an empirical question about *your* model
size and group size, so measure it rather than reasoning about it.

Sends the same shape a training step does -- P prompts, n=G samples each --
and reports completion tokens/sec. Push an adapter first with --adapter-path
so the measurement includes whatever the serving mode costs.

    python bench_rollout.py --adapter-path models/qwen-sft-lora --rounds 5
"""

import argparse
import statistics as st
import time

from transformers import AutoTokenizer

from rollout_client import RolloutClient
from task import make_prompt

WORDS = ["pluto", "orange", "traffic", "wonderful", "silver", "planet",
         "elephant", "mountain", "keyboard", "javelin", "umbrella", "cathedral"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--rollout-url", default="http://192.168.100.11:8300")
    ap.add_argument("--adapter-path", default=None)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--prompts-per-step", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    client = RolloutClient(args.rollout_url)
    client.wait_ready()
    info = client.info()
    mode = info.get("weight_mode", "?")
    print(f"mode={mode}  model={info['model']}  vllm={info['vllm_version']}")

    if args.adapter_path:
        import json
        from pathlib import Path
        from safetensors.torch import load_file

        p = Path(args.adapter_path)
        cfg = json.loads((p / "adapter_config.json").read_text())
        t = time.time()
        res = client.update_weights(
            load_file(str(p / "adapters.safetensors")),
            scale=cfg["lora_parameters"]["scale"],
        )
        print(f"adapter push: {time.time() - t:.3f}s wall "
              f"({res['seconds']:.3f}s server-side, {res['bytes'] / 1e6:.1f} MB)")

    tok = AutoTokenizer.from_pretrained(args.model)
    P, G = args.prompts_per_step, args.group_size
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": make_prompt(w)}],
            tokenize=True, return_dict=False, add_generation_prompt=True,
        )
        for w in WORDS[:P]
    ]

    tps, secs, lens = [], [], []
    for i in range(args.rounds + args.warmup):
        t = time.time()
        r = client.generate(prompts, n=G, temperature=args.temp,
                            max_tokens=args.max_tokens, seed=1000 + i)
        dt = time.time() - t
        n_tok = sum(len(c) for c in r["completions"])
        if i < args.warmup:
            print(f"  warmup: {n_tok} tok in {dt:.2f}s")
            continue
        tps.append(n_tok / dt)
        secs.append(dt)
        lens.append(n_tok / len(r["completions"]))
        print(f"  round {i - args.warmup}: {n_tok} tok, {dt:.2f}s, {n_tok / dt:.0f} tok/s")

    print(f"\n{mode}: {P}x{G} = {P * G} completions, mean len {st.mean(lens):.0f} tok")
    print(f"  tokens/sec : {st.mean(tps):.0f}"
          + (f" +/- {st.stdev(tps):.0f}" if len(tps) > 1 else ""))
    print(f"  seconds    : {st.mean(secs):.2f}"
          + (f" +/- {st.stdev(secs):.2f}" if len(secs) > 1 else ""))


if __name__ == "__main__":
    main()
