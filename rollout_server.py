"""Rollout server -- runs on Bob. vLLM behind a tiny HTTP API.

Not the OpenAI server. GRPO needs three things the chat API makes awkward:

  * token ids in and token ids out. Re-tokenizing generated text is subtly
    off-policy whenever the tokenizer round-trip doesn't reproduce the sampled
    ids, and the trainer differentiates the exact ids that were sampled.
  * the sampled token's logprob under the ACTUAL sampling distribution, which
    is the behavior policy in the importance ratio.
  * a weight-update hook that reaches into the live engine.

Endpoints:
    GET  /health          liveness + weight version
    GET  /info            engine config the trainer asserts against
    POST /generate        {prompt_token_ids, n, temperature, ...} -> ids+logprobs
    POST /update_weights  raw safetensors body (LoRA A/B) -> merged into weights
    POST /reset_weights   drop back to the pristine base checkpoint

Launch via launch_bob.sh.
"""

import argparse
import threading
import time

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import uvicorn

app = FastAPI()
ENGINE = {}
LOCK = threading.Lock()


class GenerateRequest(BaseModel):
    prompt_token_ids: list[list[int]]
    n: int = 1
    temperature: float = 0.7
    top_p: float = 0.0  # 0 = off, matching the reference's CLI convention
    top_k: int = 0  # 0 = off
    max_tokens: int = 768
    seed: int | None = None
    logprobs: bool = True


def _sampling_params(req: GenerateRequest, seed=None):
    from vllm import SamplingParams

    kwargs = dict(
        n=req.n,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        top_p=req.top_p if 0 < req.top_p < 1 else 1.0,
        top_k=req.top_k if req.top_k > 0 else ENGINE["top_k_off"],
        seed=seed,
    )
    if req.logprobs:
        # 0 = "just the sampled token's logprob", which is all TIS needs.
        kwargs["logprobs"] = ENGINE["logprobs_arg"]
    return SamplingParams(**kwargs)


@app.get("/health")
def health():
    return {"ok": True, "weight_version": ENGINE["weight_version"]}


@app.get("/info")
def info():
    return {
        "model": ENGINE["model"],
        "logprobs_mode": ENGINE["logprobs_mode"],
        "prefix_caching": ENGINE["prefix_caching"],
        "max_model_len": ENGINE["max_model_len"],
        "weight_version": ENGINE["weight_version"],
        "lora_synced": ENGINE["lora_synced"],
        "vllm_version": ENGINE["vllm_version"],
    }


@app.post("/generate")
def generate(req: GenerateRequest):
    from vllm import TokensPrompt

    if not req.prompt_token_ids:
        raise HTTPException(400, "prompt_token_ids is empty")

    prompts = [TokensPrompt(prompt_token_ids=ids) for ids in req.prompt_token_ids]
    # Per-prompt seeds. A single seed shared across the batch correlates the
    # groups with each other, which is not what "G independent samples per
    # prompt" is supposed to mean.
    params = (
        _sampling_params(req)
        if req.seed is None
        else [_sampling_params(req, seed=req.seed + i) for i in range(len(prompts))]
    )

    t0 = time.time()
    with LOCK:
        outs = ENGINE["llm"].generate(prompts, params, use_tqdm=False)
    dt = time.time() - t0

    # Flatten group-major: result index p*n + k is sample k of prompt p, which
    # is the layout the trainer's advantage grouping assumes.
    completions, logprobs, finish = [], [], []
    for out in outs:
        if len(out.outputs) != req.n:
            raise HTTPException(
                500, f"expected {req.n} samples, engine returned {len(out.outputs)}"
            )
        for comp in out.outputs:
            ids = list(comp.token_ids)
            completions.append(ids)
            finish.append(comp.finish_reason)
            if req.logprobs:
                lps = comp.logprobs or []
                row = []
                for pos, tok in enumerate(ids):
                    entry = lps[pos].get(tok) if pos < len(lps) else None
                    row.append(None if entry is None else entry.logprob)
                logprobs.append(row)
            else:
                logprobs.append([None] * len(ids))

    n_tok = sum(len(c) for c in completions)
    return {
        "completions": completions,
        "logprobs": logprobs,
        "finish_reasons": finish,
        "weight_version": ENGINE["weight_version"],
        "n_tokens": n_tok,
        "seconds": dt,
        "tokens_per_second": n_tok / dt if dt > 0 else 0.0,
    }


@app.post("/update_weights")
async def update_weights(request: Request, scale: float, step: int = -1):
    payload = await request.body()
    if not payload:
        raise HTTPException(400, "empty weight payload")

    t0 = time.time()
    with LOCK:
        llm = ENGINE["llm"]
        if not ENGINE["lora_synced"]:
            res = llm.collective_rpc(
                "init_lora_sync",
                kwargs=dict(
                    model_dir=ENGINE["model"],
                    target_modules=ENGINE["target_modules"],
                    base_device=ENGINE["base_device"],
                ),
            )
            ENGINE["lora_synced"] = True
            print(f"[sync] base snapshot: {res[0]}", flush=True)

        res = llm.collective_rpc(
            "apply_lora_update", kwargs=dict(payload=payload, scale=scale)
        )
        _invalidate_caches(llm)
        ENGINE["weight_version"] += 1

    return {
        "weight_version": ENGINE["weight_version"],
        "step": step,
        "bytes": len(payload),
        "seconds": time.time() - t0,
        "detail": res[0],
    }


@app.post("/reset_weights")
def reset_weights():
    with LOCK:
        llm = ENGINE["llm"]
        if not ENGINE["lora_synced"]:
            raise HTTPException(409, "no weights have been pushed; already at base")
        llm.collective_rpc("reset_to_base")
        _invalidate_caches(llm)
        ENGINE["weight_version"] += 1
    return {"weight_version": ENGINE["weight_version"], "at_base": True}


def _invalidate_caches(llm):
    """Any KV computed under the previous weights is now stale.

    With prefix caching on, blocks cached before an update would be reused
    afterwards -- the prefill half of a rollout would come from step k-1 and
    the decode half from step k. That is silently off-policy, and nothing in
    the loss would flag it. Prefix caching is off by default here; this reset
    is the belt to that suspenders (and matters if you flip it on).
    """
    for attr in ("reset_prefix_cache",):
        fn = getattr(llm, attr, None) or getattr(
            getattr(llm, "llm_engine", None), attr, None
        )
        if fn is not None:
            try:
                fn()
            except Exception as e:  # non-fatal: caching is off by default
                print(f"[sync] {attr} failed ({e}); prefix caching is "
                      f"{ENGINE['prefix_caching']}", flush=True)


def build_engine(args):
    import vllm
    from vllm import LLM
    from transformers import AutoConfig

    # Derive the LoRA-target module list from the config rather than hardcoding
    # a layer count, so a different base model doesn't silently sync a subset.
    from lora_torch import LORA_KEYS

    cfg = AutoConfig.from_pretrained(args.model)
    n_layers = cfg.num_hidden_layers
    targets = [
        f"model.layers.{i}.{k}" for i in range(n_layers) for k in LORA_KEYS
    ]

    kwargs = dict(
        model=args.model,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=args.enable_prefix_caching,
        enforce_eager=args.enforce_eager,
        seed=args.seed,
        worker_extension_cls="vllm_weight_sync.WeightSyncWorkerExtension",
        logprobs_mode=args.logprobs_mode,
    )
    try:
        llm = LLM(**kwargs)
    except TypeError as e:
        if "logprobs_mode" in str(e):
            raise SystemExit(
                "This vLLM build does not accept logprobs_mode. Truncated "
                "importance sampling needs logprobs under the ACTUAL sampling "
                "distribution (post-temperature, post-truncation); the default "
                "raw logprobs would make the ratio meaningless. Upgrade vLLM, "
                "or run the trainer with --no-tis and accept the mismatch."
            ) from e
        raise

    ENGINE.update(
        llm=llm,
        model=args.model,
        target_modules=targets,
        base_device=args.base_device,
        logprobs_mode=args.logprobs_mode,
        prefix_caching=args.enable_prefix_caching,
        max_model_len=args.max_model_len,
        weight_version=0,
        lora_synced=False,
        vllm_version=getattr(vllm, "__version__", "unknown"),
    )
    ENGINE["top_k_off"], ENGINE["logprobs_arg"] = _probe_sampling_sentinels()
    print(
        f"[bob] vLLM {ENGINE['vllm_version']} ready: {args.model} "
        f"({n_layers} layers, {len(targets)} sync targets), "
        f"logprobs_mode={args.logprobs_mode}, "
        f"prefix_caching={args.enable_prefix_caching}",
        flush=True,
    )


def _probe_sampling_sentinels():
    """vLLM has churned on 'disabled' sentinels for top_k and logprobs=0."""
    from vllm import SamplingParams

    top_k_off = 0
    try:
        SamplingParams(top_k=0)
    except Exception:
        top_k_off = -1

    logprobs_arg = 0
    try:
        SamplingParams(logprobs=0)
    except Exception:
        logprobs_arg = 1
    return top_k_off, logprobs_arg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Llama-3.2-1B-Instruct-bf16")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8300)
    ap.add_argument("--dtype", default="bfloat16")
    # Explicit and low. vLLM's ~0.9 default pre-allocates ~99 GB of GB10's
    # 121 GB unified pool; a 1B model needs a fraction of that and the box
    # does not OOM gracefully when the pool is oversubscribed.
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    ap.add_argument("--max-model-len", type=int, default=1024)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--enforce-eager", action="store_true",
                    help="disable CUDA graphs; weight updates copy in place so "
                    "graphs stay valid, but this is the escape hatch")
    ap.add_argument("--enable-prefix-caching", action="store_true",
                    help="OFF by default: cached KV from before a weight "
                    "update is off-policy. n=G already shares the group's "
                    "prefill, so this buys ~nothing here")
    ap.add_argument("--logprobs-mode", default="processed_logprobs",
                    help="MUST stay processed_* for TIS: the behavior policy "
                    "is the post-temperature, post-truncation distribution")
    ap.add_argument("--base-device", default="cuda",
                    help="where Bob parks the pristine base snapshot (~2 GB "
                    "for a 1B); 'cpu' if you are tight on the vLLM pool")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    build_engine(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
