"""GRPO trainer -- runs on Alice, rollouts come from Bob's vLLM.

DEFAULT: PIPELINED, ONE OPTIMIZER STEP STALE

Strict on-policy serializes the two machines -- Alice idles while Bob samples,
Bob idles while Alice trains -- so a two-box setup runs at about one box's
pace. That is not a throughput bug, it is what "sample from exactly the weights
you are about to differentiate" costs.

So by default the boxes are pipelined: Bob generates batch N+1 while Alice
trains on batch N. Ordering is await -> push -> launch next -> train (the push
must sit between the two generations; vLLM cannot take a weight update while
decoding). Every batch is then exactly ONE optimizer step stale.

Measured over a full 300-step run at P=4 G=8, seed 42, against the strict path:
27.8 -> 17.3 min wall (1.6x end-to-end, 1.75x per step) for 54.3% vs 54.2%
exact at temp 0.7 -- one word in a thousand, i.e. no measurable cost.
--no-async-rollouts restores the strictly on-policy path.

THE LOSS

Stale samples need a trust region, so the ratio enters the gradient with PPO
clipping:

    loss = -(1/N) sum_g sum_t min(r*A_g, clip(r, 1-e_lo, 1+e_hi)*A_g) / max_tokens
    r_gt = pi_theta(o_gt) / pi_vllm(o_gt),  r NOT detached

Under --no-async-rollouts the ratio is ~1 by construction, so it becomes a
detached truncated-importance-sampling weight instead -- variance reduction
rather than a constraint:

    loss = -(1/N) sum_g A_g sum_t min(r_gt, C) * logp_theta(o_gt) / max_tokens

At r == 1 the two coincide, so this is one objective with two regimes, not two
algorithms. Do not use the detached form with stale data: it is not a trust
region, and nothing in it stops an update walking away from the data it was
fit on.

WHY THERE IS A RATIO AT ALL, EVEN STRICTLY ON-POLICY

Weights are identical across the boxes -- pushed before every rollout -- so the
ratio is 1 in the ALGORITHMIC sense. It is not 1 numerically, and no amount of
care makes it so:

  * vLLM samples with paged-attention kernels in bf16; the trainer scores with
    a dense teacher-forced pass in fp32.
  * Alice keeps base bf16 + a fp32 LoRA branch and evaluates them separately;
    Bob folds them into one bf16 matrix. Merging is lossy at the low-rank tail.
  * vLLM's batching is not batch-invariant -- the same prompt in a different
    batch shape can produce slightly different logits.

The behavior logprobs are vLLM's own, so in async mode the single PPO ratio
absorbs engine mismatch and policy drift together -- which is why TIS is not
additionally applied there (it would double-count).

DIAGNOSTICS

`dlogp` (mean |logp_torch - logp_vllm|) is the weight-sync health metric:
measured median 0.0013 strict / 0.0012 async, stable across 300 pushes. If it
is O(1) the sync is broken and step 1 aborts (--sync-check-nats).

`clip_frac` is the staleness metric: 0.0000 strict -> 0.0010 async (median),
max 0.0053, and it does NOT escalate once lr reaches full 3e-6. Watch this one,
not dlogp -- the drift lives entirely in the tail. Percent-level means the
staleness has stopped being benign.

Everything else is the reference recipe, unchanged: reward = -levenshtein,
advantage = reward - group mean with NO std normalization, Dr.GRPO fixed-
constant length normalization, 50-step warmup, truncated rollouts count toward
the group mean but are masked from the gradient, all-identical groups dropped.

Usage:
    python train_grpo.py --adapter-path models/sft-lora --rollout-url http://192.168.100.11:8300
"""

import argparse
import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer

from lora_torch import load_adapter, lora_state_dict, save_adapter_dir
from mlx_adamw import MLXAdamW
from rollout_client import RolloutClient
from task import levenshtein, make_prompt, parse_answer

PAD_MULTIPLE = 32
NEG_INF = float("-inf")


# --------------------------------------------------------------------------
# scoring the sampling distribution
# --------------------------------------------------------------------------
def sampler_mirror(logits, targets, temp, top_p, top_k):
    """Reproduce vLLM's sampler chain on training-pass logits.

    ORDER MATTERS AND DIFFERS FROM MLX. vLLM applies temperature FIRST, then
    top_k, then top_p on the already-tempered logits (v1 Sampler.forward ->
    apply_temperature -> apply_top_k_top_p). MLX truncates on the temp-1
    distribution and applies temperature after. top_k is invariant to that
    difference (monotone rescaling), top_p is NOT -- the nucleus is computed
    over a differently-peaked distribution. Mirroring the wrong order would
    silently misstate the behavior policy at any temp != 1.

    The keep-set is part of the sampling procedure -- a constant w.r.t. the
    parameters -- so it is built on detached values; gradient reaches the
    logits only through the final masked_fill.
    """
    logits = logits / temp
    if top_k <= 0 and not (0 < top_p < 1):
        return logits

    sg = logits.detach()
    keep = torch.ones_like(sg, dtype=torch.bool)
    if top_k > 0:
        kth = torch.topk(sg, min(top_k, sg.shape[-1]), dim=-1).values[..., -1:]
        keep &= sg >= kth
    if 0 < top_p < 1:
        masked = sg.masked_fill(~keep, NEG_INF)
        sorted_logits, sorted_idx = masked.sort(dim=-1, descending=False)
        cum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        drop_sorted = cum <= (1.0 - top_p)
        drop_sorted[..., -1] = False  # always keep at least the argmax
        drop = torch.zeros_like(drop_sorted).scatter(-1, sorted_idx, drop_sorted)
        keep &= ~drop

    # Force-keep the sampled token. Generation-time (bf16, paged) and
    # training-time (fp32, dense) logits can land on opposite sides of the
    # truncation boundary; a dropped target would give logp = -inf and poison
    # the whole batch.
    keep.scatter_(-1, targets[..., None], True)
    return logits.masked_fill(~keep, NEG_INF)


def _chunk_logp(head, hidden, targets, temp, top_p, top_k):
    logits = head(hidden).float()
    logits = sampler_mirror(logits, targets, temp, top_p, top_k)
    logp = torch.log_softmax(logits, dim=-1)
    return logp.gather(-1, targets[:, None]).squeeze(-1)


def token_logps(model, x, temp, top_p, top_k, chunk_size):
    """logp of each target token under the tempered/truncated policy. (B, L-1)

    The vocab projection is chunked and recomputed in backward. A 1B Llama has
    a 128k vocab, so a microbatch of 8 x 1024 positions would otherwise
    materialize ~4 GB of fp32 logits and as much again for log_softmax -- the
    single largest allocation in the step, and the reason naive ports OOM the
    box long before the model itself is a problem.
    """
    ids, targets = x[:, :-1], x[:, 1:]
    hidden = model.model(input_ids=ids)[0]
    B, L, H = hidden.shape
    h = hidden.reshape(-1, H)
    t = targets.reshape(-1)

    grad = torch.is_grad_enabled()
    outs = []
    for i in range(0, h.shape[0], chunk_size):
        chunk_args = (
            model.lm_head, h[i : i + chunk_size], t[i : i + chunk_size],
            temp, top_p, top_k,
        )
        # Recomputation only earns anything when there is a backward pass to
        # save activations for; under no_grad it is pure overhead (and warns).
        outs.append(
            checkpoint(_chunk_logp, *chunk_args, use_reentrant=False)
            if grad else _chunk_logp(*chunk_args)
        )
    return torch.cat(outs).view(B, L)


# --------------------------------------------------------------------------
# task glue
# --------------------------------------------------------------------------
def reward_fn(completion_text, word):
    target = word[::-1]
    parsed = parse_answer(completion_text)
    answer = "" if parsed is None else parsed.lower()
    return -levenshtein(answer, target), parsed is not None


def pad_batch(rows, pad_id, device):
    """rows: (tokens, prompt_len, behavior_logps). -> x, w, behav.

    behav is aligned to TARGET positions and NaN wherever no behavior logprob
    exists (prompt tokens, padding, and any completion token vLLM did not
    return a logprob for). NaN means "ratio 1" downstream.
    """
    max_len = max(len(t) for t, _, _ in rows)
    max_len = ((max_len + PAD_MULTIPLE - 1) // PAD_MULTIPLE) * PAD_MULTIPLE

    x = torch.tensor(
        [t + [pad_id] * (max_len - len(t)) for t, _, _ in rows],
        dtype=torch.long, device=device,
    )
    w = torch.zeros(len(rows), max_len - 1, dtype=torch.float32, device=device)
    behav = torch.full(
        (len(rows), max_len - 1), float("nan"), dtype=torch.float32, device=device
    )
    for r, (toks, p, blp) in enumerate(rows):
        # target index t trains iff p <= t+1 < len(toks)
        w[r, p - 1 : len(toks) - 1] = 1.0
        for j, lp in enumerate(blp):
            if lp is not None:
                behav[r, p - 1 + j] = lp
    return x, w, behav


def length_bucket_microbatches(rows, advs, microbatch):
    paired = sorted(
        zip(rows, advs),
        key=lambda item: (
            (len(item[0][0]) + PAD_MULTIPLE - 1) // PAD_MULTIPLE,
            len(item[0][0]),
        ),
    )
    for i in range(0, len(paired), microbatch):
        chunk = paired[i : i + microbatch]
        yield [row for row, _ in chunk], [adv for _, adv in chunk]


def lr_lambda_factory(warmup):
    """MLX's linear_schedule(0, lr, warmup) then constant.

    step/warmup, not (step+1)/warmup: MLX evaluates its schedule before
    incrementing the step counter, so the first update runs at lr 0.
    """
    def fn(step):
        return min(1.0, step / warmup) if warmup > 0 else 1.0

    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Llama-3.2-1B-Instruct-bf16")
    ap.add_argument("--rollout-url", default="http://192.168.100.11:8300",
                    help="Bob, over the ConnectX-7 link")
    ap.add_argument("--words", default="data/rl_words.txt")
    ap.add_argument("--out", default="models/grpo-lora")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--prompts-per-step", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.0,
                    help="nucleus sampling; 0 = off. Mirrored in the logp "
                    "computation in vLLM's order (temp, then top_k, then top_p)")
    ap.add_argument("--top-k", type=int, default=0, help="0 = off")
    ap.add_argument("--max-tokens", type=int, default=768,
                    help="must clear the longest LEGITIMATE trace, else correct "
                    "rollouts get truncated and punished for it")
    ap.add_argument("--train-truncated", action="store_true",
                    help="include truncated rollouts in the gradient (default: "
                    "they count toward the group mean but are masked out)")
    ap.add_argument("--lr", type=float, default=3e-6,
                    help="3e-6 is stable for BOTH full-FT and LoRA here; the "
                    "10x-for-LoRA heuristic is SFT-only and degrades RL")
    ap.add_argument("--warmup", type=int, default=50,
                    help="un-warmed Adam at RL onset wrecks the SFT policy "
                    "before the second-moment estimates calibrate")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--microbatch", type=int, default=8, help="sequences per fwd/bwd")
    ap.add_argument("--no-grad-checkpoint", action="store_true",
                    help="stop recomputing the transformer body in backward. "
                    "Measured on GB10: buys ~15%% of train_s at G=32 and ~0%% at "
                    "G=8, for 6x the peak memory (6 -> 39 GB). The chunked "
                    "128k-vocab projection dominates the step, not the body, so "
                    "this is rarely the right trade")
    ap.add_argument("--logit-chunk", type=int, default=2048,
                    help="positions per chunked vocab projection")
    # -- the two-machine correction ---------------------------------------
    ap.add_argument("--no-tis", action="store_true",
                    help="disable truncated importance sampling. The ratio is "
                    "then assumed to be 1, which is what most vLLM GRPO setups "
                    "do -- and is a lie by exactly the `dlogp` this run logs")
    ap.add_argument("--tis-cap", type=float, default=2.0,
                    help="upper truncation C on the per-token ratio; bounds the "
                    "variance a few badly-mismatched tokens can inject")
    # -- one-step-stale pipelining ----------------------------------------
    ap.add_argument("--async-rollouts", default=True,
                    action=argparse.BooleanOptionalAction,
                    help="overlap Bob's sampling with Alice's backward pass "
                    "(default). Batches are ONE optimizer step stale and the "
                    "loss is PPO-clipped. --no-async-rollouts gives the "
                    "strictly on-policy path with a detached TIS weight: 1.6x "
                    "slower end-to-end, and measured to be worth 0.1pp")
    ap.add_argument("--clip-eps", type=float, default=0.2,
                    help="PPO clip range, async mode only")
    ap.add_argument("--clip-eps-high", type=float, default=None,
                    help="asymmetric upper clip (DAPO 'clip-higher', e.g. 0.28) "
                    "to slow entropy collapse; default = --clip-eps")
    ap.add_argument("--sync-check-nats", type=float, default=0.25,
                    help="abort on step 1 if mean |logp_torch - logp_vllm| "
                    "exceeds this. bf16-vs-fp32 noise is ~1e-3; a broken "
                    "weight sync or sampler mirror is O(1)")
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--adapter-path", default="models/sft-lora")
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    ap.add_argument("--adam-bias-correction", action="store_true",
                    help="use bias-corrected Adam (torch's default). OFF to "
                    "match mlx.optimizers.AdamW, which is what --lr 3e-6 was "
                    "tuned against")
    ap.add_argument("--wandb-project", default=None)
    ap.add_argument("--wandb-run-name", default=None)
    args = ap.parse_args()

    use_tis = not args.no_tis
    assert args.temp > 0, "GRPO needs stochastic rollouts (temp > 0)"

    run = None
    if args.wandb_project:
        import wandb

        run = wandb.init(
            project=args.wandb_project, name=args.wandb_run_name, config=vars(args)
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    # -- Bob ---------------------------------------------------------------
    client = RolloutClient(args.rollout_url)
    print(f"waiting for rollout server at {args.rollout_url} ...", flush=True)
    client.wait_ready()
    info = client.info()
    print(f"rollout server: {info}", flush=True)
    if info["model"] != args.model:
        raise SystemExit(
            f"Bob is serving {info['model']} but the trainer holds {args.model}. "
            f"The LoRA merge would be applied to the wrong base weights."
        )
    if (use_tis or args.async_rollouts) and not str(
        info.get("logprobs_mode", "")
    ).startswith("processed"):
        raise SystemExit(
            f"Bob reports logprobs_mode={info.get('logprobs_mode')!r}. The "
            f"importance ratio needs logprobs under the ACTUAL sampling "
            f"distribution; raw logprobs are pre-temperature and pre-truncation, "
            f"so the ratio would be garbage. Relaunch Bob with --logprobs-mode "
            f"processed_logprobs"
            + ("." if args.async_rollouts else ", or pass --no-tis.")
        )
    if args.async_rollouts and args.no_tis:
        print("NOTE: --no-tis is ignored under --async-rollouts. The PPO ratio "
              "is computed against vLLM's own logprobs, so it already absorbs "
              "the engine mismatch that TIS existed to correct.", flush=True)
    if info.get("prefix_caching"):
        print("WARNING: prefix caching is on. KV cached before a weight update "
              "is off-policy; the server resets it after each sync, but the "
              "safe setting is off.", flush=True)

    # -- Alice -------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=args.attn
    ).to(device)
    model.config.use_cache = False
    lora_parameters = load_adapter(model, args.adapter_path)
    model.to(device)
    assert lora_parameters.get("dropout", 0.0) == 0.0, \
        "GRPO requires lora dropout == 0 (on-policy contract)"

    # Kept ON by default. Once rollouts move to Bob the backward pass is the
    # step's bottleneck, so switching recompute off looks tempting -- but it
    # was measured, and the win is ~0 at G=8 for 3x the memory. The 128k-vocab
    # projection is the expensive part, and that is chunked separately.
    if not args.no_grad_checkpoint:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
    model.train()

    pad_id = tokenizer.eos_token_id
    eos_ids = set()
    for src in (tokenizer.eos_token_id, getattr(model.generation_config, "eos_token_id", None)):
        if isinstance(src, int):
            eos_ids.add(src)
        elif isinstance(src, (list, tuple)):
            eos_ids.update(src)

    words = Path(args.words).read_text().split()
    rng = random.Random(args.seed)
    rng.shuffle(words)

    prompt_ids = {}

    def get_prompt(word):
        if word not in prompt_ids:
            prompt_ids[word] = tokenizer.apply_chat_template(
                [{"role": "user", "content": make_prompt(word)}],
                tokenize=True, return_dict=False, add_generation_prompt=True,
            )
        return prompt_ids[word]

    params = [p for p in model.parameters() if p.requires_grad]
    # NOT torch.optim.AdamW -- see mlx_adamw.py. 3e-6 is the reference's tuned
    # RL rate under an un-bias-corrected optimizer; torch's would make it ~5x
    # weaker, and the GRPO lr ladder here is sharp (3e-5 collapses, 1e-5
    # plateaus downward), so silently landing off it is not survivable.
    opt = MLXAdamW(params, lr=args.lr, weight_decay=0.01,
                   bias_correction=args.adam_bias_correction)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(args.warmup))

    weight_version = [0]

    def sync_weights(step):
        t = time.time()
        res = client.update_weights(
            lora_state_dict(model), scale=lora_parameters["scale"], step=step
        )
        weight_version[0] = res["weight_version"]
        return time.time() - t, res

    P, G = args.prompts_per_step, args.group_size
    cursor = {"iter": iter(words), "epoch": 0}
    warned_eos = False
    eps_lo = 1.0 - args.clip_eps
    eps_hi = 1.0 + (args.clip_eps if args.clip_eps_high is None else args.clip_eps_high)

    def next_words():
        out = []
        for _ in range(P):
            try:
                out.append(next(cursor["iter"]))
            except StopIteration:
                cursor["epoch"] += 1
                rng.shuffle(words)
                cursor["iter"] = iter(words)
                out.append(next(cursor["iter"]))
        return out

    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rollout")

    def launch(batch_words, step):
        """Submit a generation to Bob. Returns immediately."""
        fut = pool.submit(
            client.generate,
            [get_prompt(w) for w in batch_words],
            n=G, temperature=args.temp, top_p=args.top_p, top_k=args.top_k,
            max_tokens=args.max_tokens, seed=args.seed + step * 100003,
        )
        return {"future": fut, "words": batch_words, "step": step,
                "version": weight_version[0], "t": time.time()}

    def collect(handle):
        t = time.time()
        resp = handle["future"].result()
        wait_s = time.time() - t
        if resp["weight_version"] != handle["version"]:
            raise RuntimeError(
                f"rollouts came from weight_version {resp['weight_version']} but "
                f"were launched under {handle['version']} -- another client is "
                f"writing to Bob, or a sync landed mid-generation."
            )
        return resp, wait_s, time.time() - handle["t"]

    # Async bootstrap: prime the pipe so step 1 has a batch in hand. Note the
    # first batch is generated AND trained under theta_0, i.e. it is genuinely
    # on-policy -- which is what keeps the step-1 sync check meaningful even
    # here. Staleness starts at step 2.
    pending = None
    t0 = time.time()
    if args.async_rollouts:
        sync_weights(0)
        pending = launch(next_words(), 1)

    for step in range(1, args.steps + 1):
        if args.async_rollouts:
            # await -> push -> launch next -> train. The push has to sit
            # between the two generations because vLLM cannot take a weight
            # update while it is decoding; everything after the launch runs
            # concurrently with Bob.
            resp, wait_s, roll_dt = collect(pending)
            batch_words = pending["words"]
            stale = 0 if step == 1 else 1
            sync_dt, sync_res = sync_weights(step)
            pending = launch(next_words(), step + 1) if step < args.steps else None
        else:
            # Strictly on-policy: Bob must sample from the exact parameters
            # this step is about to differentiate, so nothing overlaps.
            batch_words = next_words()
            stale = 0
            sync_dt, sync_res = sync_weights(step)
            resp, wait_s, roll_dt = collect(launch(batch_words, step))

        completions = resp["completions"]
        behav_logps = resp["logprobs"]
        finish = resp["finish_reasons"]

        if not warned_eos:
            stopped = [
                i for i, f in enumerate(finish)
                if f == "stop" and completions[i] and completions[i][-1] not in eos_ids
            ]
            if stopped:
                print(
                    "WARNING: this vLLM build omits the stop token from "
                    "token_ids, so the policy receives no gradient on stopping. "
                    "Not synthesizing one (we cannot know which stop token "
                    "fired, and training on a guess would be worse).",
                    flush=True,
                )
            warned_eos = True

        rewards, taggeds = [], []
        for i, comp in enumerate(completions):
            r, tagged = reward_fn(tokenizer.decode(comp), batch_words[i // G])
            rewards.append(float(r))
            taggeds.append(tagged)

        # --- 3. advantages: reward - group mean, no std normalization ----
        rows, advs = [], []
        kept_groups, masked_trunc = 0, 0
        for g in range(P):
            group_r = rewards[g * G : (g + 1) * G]
            mean_r = sum(group_r) / G
            if all(r == group_r[0] for r in group_r):
                continue  # zero advantage everywhere -> no gradient
            kept_groups += 1
            p_ids = get_prompt(batch_words[g])
            for k in range(G):
                i = g * G + k
                if finish[i] == "length" and not args.train_truncated:
                    masked_trunc += 1
                    continue
                if not completions[i]:
                    continue
                rows.append((p_ids + completions[i], len(p_ids), behav_logps[i]))
                advs.append(group_r[k] - mean_r)

        n_tokens = sum(len(c) for c in completions)
        stats = {
            "reward_mean": sum(rewards) / len(rewards),
            "reward_max": max(rewards),
            "exact_rate": sum(r == 0 for r in rewards) / len(rewards),
            "tag_rate": sum(taggeds) / len(taggeds),
            "trunc_rate": sum(f == "length" for f in finish) / len(finish),
            "kept_groups": kept_groups,
            "masked_trunc": masked_trunc,
            "len_mean": n_tokens / len(completions),
            "rollout_tps": resp.get("tokens_per_second", n_tokens / max(roll_dt, 1e-9)),
            "sync_s": sync_dt,
            "rollout_s": roll_dt,
            # How long the trainer actually sat idle waiting on Bob. In async
            # mode this is the number that says who the bottleneck is: ~0 means
            # the backward pass covers generation entirely, large means Bob is
            # the wall and the overlap has run out of road.
            "gen_wait_s": wait_s,
            "stale": stale,
            "epoch": cursor["epoch"],
        }

        # --- 4. policy gradient over microbatches ------------------------
        if rows:
            t_train = time.time()
            n_total = len(rows)
            opt.zero_grad(set_to_none=True)
            loss_val = 0.0
            d_sum, d_n, clipped, ratio_sum, ratio_max = 0.0, 0, 0, 0.0, 0.0

            for chunk, chunk_advs in length_bucket_microbatches(
                rows, advs, args.microbatch
            ):
                x, w, behav = pad_batch(chunk, pad_id, device)
                adv = torch.tensor(chunk_advs, dtype=torch.float32, device=device)
                adv = adv / n_total

                logp = token_logps(
                    model, x, args.temp, args.top_p, args.top_k, args.logit_chunk
                )

                have = (~torch.isnan(behav)) & (w > 0)
                if have.any():
                    d = (logp.detach() - behav)[have]
                    d_sum += d.abs().sum().item()
                    d_n += d.numel()
                    r_det = d.exp()
                    ratio_sum += r_det.sum().item()
                    ratio_max = max(ratio_max, r_det.max().item())
                    clipped += int(
                        (((r_det < eps_lo) | (r_det > eps_hi)).sum()
                         if args.async_rollouts
                         else (r_det > args.tis_cap).sum()).item()
                    )

                delta = logp - torch.nan_to_num(behav, nan=0.0)

                # Dr.GRPO fixed-constant normalization in both branches:
                # dividing by the sequence's own length dilutes per-token
                # punishment on longer failures, which teaches the policy to
                # bloat wrong answers until they hit the token cap.
                if args.async_rollouts:
                    # PPO-clipped surrogate. The ratio is IN the gradient here,
                    # which is the whole difference from the strict path: with
                    # genuinely stale data the ratio has to form a trust region,
                    # and a detached weight cannot do that -- nothing would stop
                    # the update walking away from the data it was fit on.
                    # At ratio == 1 this reduces to the strict objective, so the
                    # two branches agree in the on-policy limit.
                    ratio = torch.where(have, delta.exp(), torch.ones_like(delta))
                    a = adv[:, None]
                    per_tok = torch.min(ratio * a, ratio.clamp(eps_lo, eps_hi) * a)
                    loss = -((per_tok * w).sum(dim=1) / args.max_tokens).sum()
                else:
                    # Detached truncated importance sampling. Correct when the
                    # ratio is ~1 by construction and the cap essentially never
                    # fires; it is variance reduction, not a constraint.
                    if use_tis:
                        ratio = torch.where(
                            have, delta.detach().exp(), torch.ones_like(delta)
                        ).clamp(max=args.tis_cap)
                    else:
                        ratio = torch.ones_like(logp)
                    seq_logp = (logp * ratio * w).sum(dim=1) / args.max_tokens
                    loss = -(adv * seq_logp).sum()

                loss.backward()
                loss_val += loss.item()

            gnorm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip) \
                if args.grad_clip > 0 else torch.tensor(0.0)
            opt.step()
            sched.step()

            stats["loss"] = loss_val
            stats["grad_norm"] = float(gnorm)
            stats["lr"] = sched.get_last_lr()[0]
            stats["train_s"] = time.time() - t_train
            stats["dlogp"] = d_sum / max(d_n, 1)
            if d_n:
                stats["ratio_mean"] = ratio_sum / d_n
                stats["ratio_max"] = ratio_max
                stats["clip_frac"] = clipped / d_n

            # A broken weight sync or a mis-mirrored sampler is O(1) nats.
            # Catch it on step 1 rather than after a 300-step run that looks
            # plausible and is quietly training on someone else's samples.
            # Guarded on stale == 0: from step 2 of an async run, dlogp is
            # SUPPOSED to be non-trivial -- it is measuring one optimizer step
            # of policy drift, not a bug. Step 1 is on-policy in both modes,
            # which is exactly why the check survives the async path.
            if stale == 0 and step == 1 and d_n and stats["dlogp"] > args.sync_check_nats:
                raise SystemExit(
                    f"\nSYNC CHECK FAILED: mean |logp_torch - logp_vllm| = "
                    f"{stats['dlogp']:.3f} nats/token (threshold "
                    f"{args.sync_check_nats}).\nThe two machines do not agree on "
                    f"what policy was sampled. Likely causes, in order:\n"
                    f"  1. weight sync is not reaching vLLM (check Bob's log for "
                    f"the 'loaded' count on the first /update_weights)\n"
                    f"  2. Bob's base checkpoint differs from Alice's --model\n"
                    f"  3. --top-p/--top-k set but Bob's logprobs_mode is not "
                    f"processed_logprobs\n"
                    f"  4. temperature mismatch between sampling and scoring\n"
                    f"Re-run with --sync-check-nats inf to train anyway."
                )
        else:
            stats["loss"] = 0.0

        def _fmt(k, v):
            if not isinstance(v, float):
                return f"{k}={v}"
            # lr spends the first 50 steps under 3e-6 and loss is ~1e-3; both
            # print as a flat 0.0000 in fixed notation, which reads as a stall.
            return f"{k}={v:.3e}" if k in ("lr", "loss") else f"{k}={v:.4f}"

        line = " ".join(_fmt(k, v) for k, v in stats.items())
        peak = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
        print(f"step {step}/{args.steps} {line} "
              f"({(time.time() - t0) / step:.1f} s/step, peak {peak:.1f} GB)",
              flush=True)
        if run:
            run.log(stats, step=step)

        if args.save_every and step % args.save_every == 0:
            save_adapter_dir(f"{args.out}-step{step}", model, lora_parameters, args.model)
            print(f"checkpoint -> {args.out}-step{step}", flush=True)

    pool.shutdown(wait=False, cancel_futures=True)
    save_adapter_dir(args.out, model, lora_parameters, args.model)
    print(f"saved -> {args.out}")
    if run:
        run.finish()


if __name__ == "__main__":
    main()
