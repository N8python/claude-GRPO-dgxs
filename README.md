# Two-box on-policy GRPO — vLLM on Bob, trainer on Alice

Word reversal (`"pluto"` → `"otulp"`), ported from the single-box MLX reference
at [N8python/grpoDGX](https://github.com/N8python/grpoDGX).

The MLX version runs rollouts and training in one process, which makes it
on-policy by arithmetic: same weights, same engine, importance ratio
identically 1. It also means generation runs at training-framework speed. This
port splits the two halves across the Sparks:

```
        Alice  192.168.100.10                 Bob  192.168.100.11
   ┌──────────────────────────┐          ┌──────────────────────────┐
   │ train_grpo.py (torch)    │  LoRA    │ rollout_server.py        │
   │  · frozen bf16 base      │  A/B     │  · vLLM engine           │
   │  · fp32 LoRA r=8         │ ~22 MB   │  · pristine base copy    │
   │  · AdamW, Dr.GRPO loss   │ ───────► │  · merge → load_weights  │
   │                          │          │                          │
   │                          │ ◄─────── │  token ids + logprobs    │
   └──────────────────────────┘  JSON    └──────────────────────────┘
                        ConnectX-7 200GbE
```

Per step: **await batch → push weights → launch next batch → train**. Bob
generates batch N+1 while Alice trains on batch N, so every batch is exactly
one optimizer step stale. `--no-async-rollouts` serializes it back to strictly
on-policy.

## Two regimes of one objective

**Default (pipelined).** Stale samples need a trust region, so the ratio enters
the gradient with PPO clipping:

```
loss = -(1/N) Σ_g Σ_t min(r·A_g, clip(r, 1-ε_lo, 1+ε_hi)·A_g) / max_tokens
r_gt = π_θ(o_gt) / π_vLLM(o_gt)     (NOT detached)
```

**`--no-async-rollouts` (strict).** The ratio is ~1 by construction, so it
becomes a detached truncated-importance-sampling weight — variance reduction
rather than a constraint:

```
loss = -(1/N) Σ_g A_g Σ_t min(r_gt, C) · logp_θ(o_gt) / max_tokens
                                    (detached; C = --tis-cap, default 2.0)
```

At `r == 1` these coincide, so it's one objective with two regimes, not two
algorithms. Don't use the detached form with stale data: it isn't a trust
region, and nothing in it stops an update walking away from the data it was fit
on.

## Why there's a ratio at all, even strictly on-policy

Weights are identical across the two machines — pushed before every rollout —
so the ratio is 1 in the *algorithmic* sense. It is not 1 numerically, and
nothing about being careful makes it so:

- vLLM samples with paged-attention kernels in bf16; the trainer scores with a
  dense teacher-forced pass in fp32.
- Alice keeps base bf16 + a separate fp32 LoRA branch; Bob folds them into one
  bf16 matrix. The merge is lossy at the low-rank tail.
- vLLM is not batch-invariant — the same prompt in a different batch shape can
  produce slightly different logits.

That residual is **corrected, not assumed away**. The behavior logprobs are
vLLM's own, so in the default mode the single PPO ratio absorbs engine mismatch
and policy drift together — which is why TIS isn't additionally applied there
(it would double-count). `--no-tis` drops the correction in strict mode, which
is what most naive vLLM GRPO setups run, and is wrong by exactly the `dlogp`
this trainer prints every step.

Either way this requires vLLM's **`processed_logprobs`** (post-temperature,
post-truncation). The default `raw_logprobs` are pre-everything and would make
the ratio meaningless; the trainer refuses to start if Bob reports the wrong
mode.

## Everything else is the reference recipe

Unchanged, because each piece fixed a measured pathology: reward =
−levenshtein; advantage = reward − group mean with **no** std normalization;
Dr.GRPO fixed-constant length normalization (per-sequence `1/|o|` teaches the
policy to bloat its failures); 50-step linear warmup (un-warmed Adam at RL
onset wrecks the SFT policy); truncated rollouts count toward the group mean
but are masked from the gradient; all-identical-reward groups dropped.

LoRA rank 8 at scale 32/rank, lr **3e-6**. The "LoRA wants ~10× lr" heuristic
applies to SFT and *not* to RL — 3e-5 collapses, 1e-5 plateaus downward.

## Setup

**No installs required.** Existing envs already cover both sides:

| box | env | has |
|---|---|---|
| Alice (trainer) | `femtogpt-torch` | torch 2.12.0+cu130, transformers 5.11, requests, safetensors |
| Bob (rollouts) | `gemmacev` | vLLM 0.23.0, fastapi 0.137, uvicorn 0.49 |

The base model is `mlx-community/Llama-3.2-1B-Instruct-bf16` — already cached
on both boxes, ungated (unlike `meta-llama/…`), a standard HF `LlamaForCausalLM`
repo, and the exact weights the MLX reference trained on, so the leaderboard
numbers stay comparable. Bob merges LoRA against **its own** copy of that
checkpoint, so it has to be the same on both boxes; it is.

Copy this directory to both machines (there is no shared filesystem):

```bash
rsync -av --exclude models --exclude __pycache__ ./ Alice:~/dgxGRPOProper/
rsync -av --exclude models --exclude __pycache__ ./ Bob:~/dgxGRPOProper/
```

## Bring-up

```bash
# 1. Bob: start the rollout server (~1-2 min for vLLM to warm up)
ssh Bob 'cd ~/dgxGRPOProper && PY=~/miniforge3/envs/gemmacev/bin/python setsid nohup ./launch_bob.sh > bob_server.log 2>&1 < /dev/null &'
```

```bash
# 2. Alice: SFT (~1 min), then verify the two boxes agree BEFORE training
ssh Alice
cd ~/dgxGRPOProper && conda activate femtogpt-torch
python train_sft.py
python check_sync.py --adapter-path models/sft-lora
```

`check_sync.py` is the whole point of running this before a 300-step job. It
runs two tests that catch the three failure modes which do *not* crash:

- **agreement** — mean `|logp_torch − logp_vllm|`. Healthy is O(1e-3). O(1)
  means a broken sampler mirror or mismatched base checkpoints.
- **liveness** — perturb the LoRA, push, and require Bob's greedy output to
  change. A no-op sync passes the agreement test *trivially*, because both
  sides then agree about the untouched base model. Only this test separates
  them.

```bash
# 3. Alice: GRPO
python train_grpo.py --adapter-path models/sft-lora --wandb-project dgx-grpo-proper
```

Or `./run_reference.sh` for the whole pipeline.

## Reading the log

```
step 12/300 reward_mean=-2.8750 exact_rate=0.1875 tag_rate=1.0000 trunc_rate=0.0000
  kept_groups=4 len_mean=181.3 rollout_tps=2140.5 sync_s=0.184 rollout_s=2.91
  loss=-0.0031 grad_norm=0.4127 dlogp=0.0042 tis_ratio_mean=1.0009
  tis_clip_frac=0.0001 (4.2 s/step, peak 14.8 GB)
```

| field | what it tells you |
|---|---|
| `dlogp` | **the health metric for this architecture.** Mean \|logp_torch − logp_vllm\|. Should sit flat around wherever `check_sync.py` left it. A jump means the two boxes drifted apart. |
| `clip_frac` | **the staleness metric.** Fraction of tokens outside the clip range. ~0.001 is expected by default; percent-level means the staleness stopped being benign. Exactly 0.0000 under `--no-async-rollouts`. |
| `gen_wait_s` | how long the trainer sat idle waiting on Bob. ~0 means the backward pass fully covers generation; large means Bob is the wall and the overlap has run out of road. |
| `stale` | optimizer steps between the sampling policy and the target policy. 1 by default (0 on step 1, which is why the step-1 sync check survives); always 0 under `--no-async-rollouts`. |
| `sync_s` | weight push time. ~0.3 s. If this climbs, the link degraded. |
| `exact_rate` | rollout exact-match. This is the thing that should go up. |
| `kept_groups` | groups with non-zero advantage. Persistent 0 = no gradient at all. |

## Why pipelining is the default

Strict on-policy serializes the machines — Alice idles while Bob samples, Bob
idles while Alice trains — which is why a strictly on-policy two-box setup runs
at roughly *one* box's pace. That's not a throughput bug; it's what "sample from
exactly the weights you're about to differentiate" costs.

Measured over the full 300-step A/B above, P=4 G=8, seed 42:

| | `--no-async-rollouts` | default |
|---|---|---|
| wall clock, 300 steps | 27.8 min | **17.3 min** (1.6×) |
| s/step | 5.8 | **3.3** (1.75×) |
| exact @ t0.7 | 54.2% | **54.3%** |
| `gen_wait_s` | = rollout time | median 0.19, p90 1.33 |
| `dlogp` median | 0.0013 | 0.0012 |
| `clip_frac` median | 0.0000 | 0.0010 |
| `ratio_max` | ~1.9 | 2.80 |

One step of drift on a rank-8 LoRA at 3e-6 barely moves the mean policy — the
`dlogp` columns are indistinguishable — and shows up only as a thin tail, which
is exactly what the clip is for. It also does **not** compound: `clip_frac`
restricted to steps > 50 (full lr) is identical to the whole-run figure.

`--clip-eps-high` (DAPO-style asymmetric clipping, e.g. 0.28) is there if
entropy collapse shows up on longer runs.

Step 1 is on-policy in both modes — the pipeline bootstrap generates *and*
trains the first batch under θ₀ — which is why the step-1 sync check survives
the default path. From step 2 it's suppressed, because a non-trivial `dlogp`
is then the intended behaviour rather than a bug.

Reference trajectory (MLX, single box, seed 42): SFT 34.5% greedy → GRPO
**51.6%** exact at temp 0.7 after 300 steps. That is the number to beat.

## Results: strict vs async, 300 steps, seed 42

Both arms trained from the same SFT adapter and were evaluated on the held-out
1000 at the training temperature. `./run_ab.sh` reproduces this.

| | exact @ t0.7 | avg lev | tag |
|---|---|---|---|
| SFT baseline (greedy) | 34.3% | 1.47 | 98.6% |
| strict — step 100 | 40.5% | 1.29 | 99.9% |
| strict — step 200 | 51.0% | 1.05 | 99.9% |
| **strict — step 300** | **54.2%** | 0.94 | 100.0% |
| async — step 100 | 40.4% | 1.28 | 100.0% |
| async — step 200 | 50.5% | 1.03 | 99.9% |
| **async — step 300** | **54.3%** | 0.93 | 100.0% |
| *MLX reference, canonical seed 42* | *51.6%* | *0.98* | *99.8%* |

**One-step staleness cost nothing measurable.** 54.3 vs 54.2 is one word in a
thousand, the async arm's edit distance is marginally better, and the two
curves agree at every checkpoint — all differences far inside binomial noise
(SE ≈ 1.6 pp at n=1000). Training wall clock was **27.8 min → 17.3 min, 1.6×
end-to-end** (1.75× per step; the gap is startup amortized).

Both arms clear the published 51.6%. Be careful reading that as an improvement,
though: 51.6% is the canonical A6000 run, and MLX on a *Spark* has hit 55.3%
and 57.4% with the same recipe. 54.2–54.3% sits above canonical and just under
the best MLX-on-Spark runs — i.e. this reproduces the reference rather than
beating it.

## Results: Qwen3-4B-Instruct-2507

Same pipeline, **zero code changes** — vLLM's Qwen3 shares the `qkv_proj` /
`gate_up_proj` fusion mapping, and the LoRA target filter skips Qwen3's
`q_norm`/`k_norm` correctly (RMSNorm, not Linear). `check_sync` passed at
0.00167 nats, median 0. Reproduce with `./run_qwen.sh`.

| stage | exact | avg lev | tag |
|---|---|---|---|
| base, zero-shot (greedy) | 17.6% | 2.54 | 100.0% |
| + SFT, 100 traces (greedy) | 72.4% | 0.43 | 100.0% |
| + SFT (temp 0.7) | 63.3% | 0.55 | 92.6% |
| + GRPO 100 steps (t0.7) | 76.8% | 0.34 | 99.9% |
| + GRPO 200 steps (t0.7) | 81.3% | 0.27 | 99.9% |
| **+ GRPO 300 steps (t0.7)** | **83.5%** | **0.23** | **99.9%** |

**Matched-decode uplift: +20.2 points** (63.3 → 83.5 at temp 0.7), edit distance
down 58%, and the tag rate restored from 92.6% → 99.9%. Compare like with like:
the greedy SFT number is 72.4%, but sampling costs that policy 9.1 points, so
quoting 72.4 → 83.5 understates GRPO by roughly half. Part of what GRPO buys
here is *format robustness under sampling* — an untagged answer scores as empty,
i.e. maximum penalty, so the policy learns to never drop the tags.

**52m01s for 300 steps** (10.2 s/step); SFT 70 s; each 1000-word eval ~2 min.
Peak 12.7 GB on Alice, 51 GB on Bob.

Two things differ from the 1B and are worth knowing before you scale up:

1. **The bottleneck inverts.** Generation dominates outright — a late step
   showed `gen_wait_s` 4.12 s against `train_s` 3.28 s, i.e. the trainer finishes
   its whole backward and then waits longer than that again on Bob. Rollout
   throughput is ~576–740 tok/s vs ~2,100 on the 1B. This is the regime the
   split exists for; pipelining hides the entire training pass and Bob is *still*
   the critical path.
2. **A strong starting policy starves the gradient.** ~32% of groups were
   dropped for zero advantage and 12% of steps produced no gradient at all,
   because a 72.4% policy returns many all-correct groups. You pay full rollout
   cost for ~264 steps of usable gradient out of 300. Raise `--group-size` or
   `--prompts-per-step`, or shape the reward (exact-match bonus, short-word
   curriculum), if you rerun.

`sync_s` also rose 0.3 → ~1.3 s (12% of the step): 66 MB of rank-8 LoRA in fp32
plus a 7.3 GB base re-merge per push. Both are reducible — bf16 on the wire
would halve the transfer — and neither is optimized.

Health metrics held for all 300 steps in both arms. `dlogp` median 0.0013
(strict) / 0.0012 (async), max ~0.003 — the weight sync never degraded across
300 pushes, and the staleness is invisible in the mean. It shows up only in the
tail: `clip_frac` median 0.0000 → 0.0010, `ratio_max` 1.9 → 2.8. Crucially
`clip_frac` did **not** escalate once lr reached full 3e-6 — restricted to
steps > 50 it is identical (median 0.0010, max 0.0053). `trunc_rate` stayed
0.0000 and `len_mean` flat at 140–167 throughout, so neither the truncation
storm nor the completion-explosion failure mode appeared.

## The porting trap: MLX's AdamW is not torch's AdamW

`mlx.optimizers.AdamW` defaults to **`bias_correction=False`**. `torch.optim.AdamW`
always applies bias correction and gives you no way to switch it off. With
comparable gradients the update magnitudes differ by `(1-β₁ᵗ)/√(1-β₂ᵗ)` —
**3.2× at step 1, 6.5× at step 10, still 5.0× at step 39.**

Every learning rate in the reference (1e-4 SFT, 3e-6 GRPO, and the 50-step
warmup tuned around it) was found against the un-bias-corrected optimizer.
Handing those numbers to `torch.optim.AdamW` silently under-trains. Measured
here on the 100-trace SFT:

| optimizer | exact | avg lev | tag |
|---|---|---|---|
| `torch.optim.AdamW`, lr 1e-4 | 25.8 / 26.4 / 25.8% *(seeds 42/0/7)* | 2.19–2.28 | 97.9–99.2% |
| `MLXAdamW`, lr 1e-4 | **34.3%** | **1.47** | 99.8% |
| MLX reference (seed 42) | 34.5% | 1.49 | ~100% |

Consistent across three seeds, so it was systematic, not variance. Base-model
loading was ruled out separately: MLX and transformers produce the same token
ids and the same top-10 next-token logprobs to ~0.01 nats on this checkpoint,
with `rope_theta=500000` correctly preserved.

Hence `mlx_adamw.py`, a faithful port. This matters more for the RL phase than
for SFT — the GRPO lr ladder on this task is sharp (3e-5 collapses, 1e-5
plateaus downward, 3e-6 wins), so landing 5× off it produces a mediocre curve
that looks like an honest negative result. Pass `--adam-bias-correction` to opt
back into torch semantics, and raise `--lr` by roughly 5× if you do.

## Measured on Alice + Bob

vLLM 0.23.0, Llama-3.2-1B-Instruct-bf16, LoRA r8, defaults unless noted.

| stage | wall | peak (Alice) | note |
|---|---|---|---|
| SFT, 39 steps | ~30 s | 13.1 GB | → 34.3% exact greedy, lev 1.47 |
| GRPO step, P=4 G=8 (default) | **~3.3 s** | 6.0 GB | overlapped; `gen_wait_s` median 0.19 s |
| GRPO step, P=4 G=8, `--no-async-rollouts` | ~5.8 s | 6.0 GB | sync 0.3 s · rollout 2.2–3.3 s · train 2.3–3.8 s |
| GRPO step, P=4 G=32, `--no-async-rollouts` | ~19 s | 6.0 GB | sync 0.3 s · rollout 4.7 s · **train 11 s** |
| eval, 1000 words | ~40 s | — | 4,052 tok/s aggregate on Bob |

Agreement between the boxes, steady state: `dlogp` **0.0015–0.003** nats/token,
`tis_ratio_mean` 1.0000, `tis_clip_frac` ~0. Median per-token `|Δlogp|` is
exactly 0 — most tokens agree bit-for-bit; the mismatch lives in a thin tail
(p99 ≈ 0.028, max ≈ 0.42), which is precisely the shape truncated IS exists to
handle.

**Two findings worth knowing before you tune:**

1. **Strictly on-policy at G=8 is roughly par with the single-box MLX
   reference** (~6 s/step either way) — because strict on-policy serializes the
   two boxes, so you own two machines and use one at a time. 32 sequences of
   ~160 tokens is also far too small to show vLLM's throughput advantage.
   Pipelining is what actually breaks the tie (3.3 s/step, 1.75×), which is why
   it is the default; scaling G is the other lever, since rollout throughput
   goes 2,100 → 3,875 tok/s from G=8 to G=32. Note the two levers oppose each
   other — see finding 2.
2. **Past about G=16 the trainer is the bottleneck, not the sampler** — at
   G=32 it is 11 s of backward against 5 s of sampling. That inverts the usual
   GRPO picture and means further speedup has to come from Alice. Turning
   gradient checkpointing off is *not* it: measured, it buys ~15% of `train_s`
   at G=32 and ~0% at G=8 for 6× the peak memory, because the chunked
   128k-vocab projection dominates the step rather than the transformer body.
   `--no-grad-checkpoint` exists if you want to re-measure on your own shapes.

## Gotchas specific to these boxes

- **Unified memory does not OOM gracefully.** Oversubscribing GB10's 121 GB
  wedges the whole box. `--gpu-memory-utilization` is 0.35 here, not vLLM's
  0.9 default, which would pre-allocate ~99 GB for a 1B model.
- **`stop_bob.sh`, not `pkill`.** Killing the HTTP process leaves vLLM's
  `EngineCore` child orphaned holding the GPU pool, which starves the next
  launch. The bracket patterns in that script are also load-bearing —
  `pkill -f rollout_server.py` over ssh matches the ssh command line and kills
  your own session.
- **Prefix caching is off.** KV cached before a weight update is off-policy if
  reused after it. `n=G` already shares each group's prefill, so it buys
  nothing here. The server resets the cache after every sync anyway.
- **Detached launches** need `setsid nohup … < /dev/null &`; a plain
  `nohup … &` hangs the ssh session on these machines.
- **No shared filesystem.** `models/` on Alice is not visible to Bob. Bob only
  ever needs the base checkpoint; adapters travel over HTTP.

## Files

| file | box | what |
|---|---|---|
| `rollout_server.py` | Bob | vLLM behind `/generate`, `/update_weights`, `/reset_weights` |
| `vllm_weight_sync.py` | Bob | worker extension: merge LoRA into base, `load_weights` |
| `train_grpo.py` | Alice | the trainer — sync-then-rollout loop, Dr.GRPO + TIS |
| `train_sft.py` | Alice | 100-trace SFT, produces the starting adapter |
| `check_sync.py` | Alice | **run before training**; agreement + liveness tests |
| `evaluate.py` | Alice | eval via Bob; never loads the model locally |
| `lora_torch.py` | both | hand-rolled LoRA keyed by HF checkpoint names |
| `mlx_adamw.py` | Alice | AdamW matching MLX's, incl. `bias_correction=False` |
| `rollout_client.py` | Alice | HTTP client |
| `task.py`, `make_dataset.py`, `data/` | — | verbatim from the reference, so the eval set stays byte-identical |
