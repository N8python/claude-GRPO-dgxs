"""vLLM worker extension: merge a LoRA update into the live model weights.

Runs inside the vLLM worker process (vLLM instantiates it via
``worker_extension_cls``, so ``self`` is the Worker and ``self.model_runner``
is available). Reached from the server process with ``collective_rpc``.

Why merge instead of using vLLM's native LoRA runtime:

  * ``load_weights`` is the same code path for a merged LoRA and for a full
    fine-tune. vLLM's LoRA runtime only ever handles LoRA, so it is a dead end
    the moment you want full-parameter GRPO.
  * ``load_weights`` has been a stable API for years; the LoRA runtime has
    churned considerably across vLLM versions.
  * No punica kernels in the decode path, so generation runs at base speed.
  * Hot-swapping an adapter every optimizer step means minting a fresh
    lora_int_id each step to defeat the adapter cache. (Weaker than it looks:
    older vLLM required a filesystem path for LoRARequest, which would mean
    writing the adapter to disk every step, but newer versions accept tensors
    directly. Not re-checked against 0.23.0.)

NOT a reason, despite an earlier version of this comment saying so: that
merging is somehow numerically cleaner. It is not. The trainer keeps base and
LoRA as *separate* branches (``base(x) + scale * (x @ A.T) @ B.T``), which is
structurally what vLLM's LoRA path also computes. Merging is the side that
introduces a distinct math path -- folding the low-rank term into the weight
matrix and rounding to bf16 before the GEMM. It measures fine in practice
(dlogp ~0.0013, median 0), but "fine" is not "better", and the comparison was
never run.

The honest cost of this choice: Bob holds a pristine base snapshot (~2 GB for
a 1B, ~7.3 GB for a 4B) that the LoRA runtime would not need, and every push
rewrites the full merged weight. Profiled on Qwen3-4B, a sync is 0.85 s of
which 0.73 s is this merge and only 0.068 s is the 66 MB crossing the wire --
the expression below materializes ~5 full-size fp32 intermediates, ~130 GB of
memory traffic against 273 GB/s. That is self-inflicted, not inherent: folding
it into a single ``torch.addmm`` with the scale pushed onto B would cut it to
one pass (~15 GB). Worth doing before blaming the interconnect.

The merged tensors are handed to ``model.load_weights()`` rather than poked
into parameters by hand: vLLM fuses q/k/v into ``qkv_proj`` and gate/up into
``gate_up_proj``, and ``load_weights`` owns that shard mapping. Reimplementing
it here is exactly the kind of thing that silently half-works.
"""

import glob
import os

import torch


class WeightSyncWorkerExtension:
    # populated on init_lora_sync
    _base: dict = None
    _base_device: str = "cuda"
    _sync_count: int = 0

    def _model(self):
        return self.model_runner.model

    def init_lora_sync(self, model_dir: str, target_modules: list, base_device: str):
        """Snapshot the pristine base weights for every LoRA-target linear.

        Read straight from the checkpoint on disk rather than from the live
        model: vLLM has already fused q/k/v and gate/up, so the live model no
        longer has the per-linear tensors we need to merge against.
        """
        from safetensors.torch import load_file

        model_dir = _resolve_model_dir(model_dir)
        wanted = {f"{m}.weight" for m in target_modules}
        shards = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
        if not shards:
            raise RuntimeError(f"no *.safetensors under {model_dir}")

        dtype = next(self._model().parameters()).dtype
        base = {}
        for shard in shards:
            tensors = load_file(shard)
            for name in wanted & set(tensors):
                base[name] = tensors[name].to(device=base_device, dtype=dtype)
            del tensors

        missing = wanted - set(base)
        if missing:
            raise RuntimeError(
                f"{len(missing)} LoRA-target weights absent from {model_dir}, "
                f"e.g. {sorted(missing)[:3]} -- does Bob's base match Alice's?"
            )
        self._base = base
        self._base_device = base_device
        nbytes = sum(t.numel() * t.element_size() for t in base.values())
        return {"tensors": len(base), "bytes": nbytes, "dtype": str(dtype)}

    def apply_lora_update(self, payload: bytes, scale: float, fp32_merge: bool = False):
        """Merge A/B into the base snapshot and write into the live model."""
        from safetensors.torch import load as load_bytes

        if self._base is None:
            raise RuntimeError("init_lora_sync must be called before apply_lora_update")

        lora = load_bytes(payload)
        model = self._model()
        dev = self._base_device
        count = [0]

        def merged():
            for name, w0 in self._base.items():
                key = name[: -len(".weight")]
                a = lora.get(f"{key}.lora_a")
                b = lora.get(f"{key}.lora_b")
                if a is None or b is None:
                    raise RuntimeError(f"LoRA payload has no A/B for {key}")
                if fp32_merge:
                    delta = torch.mm(
                        b.to(dev, torch.float32), a.to(dev, torch.float32)
                    ).mul_(scale)
                    out = delta.add_(w0.float()).to(w0.dtype)
                else:
                    # One fused pass instead of ~5 full-size fp32 intermediates:
                    # ~15 GB of traffic rather than ~130 GB. cuBLAS accumulates
                    # bf16 GEMMs in fp32 and alpha applies inside that
                    # accumulator, so the rank-r sum and the scale stay fp32 --
                    # only the A/B inputs and the single output rounding are
                    # bf16. Measured on Qwen3-4B: 0.73 s -> see README.
                    out = torch.addmm(
                        w0, b.to(dev, w0.dtype), a.to(dev, w0.dtype),
                        beta=1.0, alpha=scale,
                    )
                count[0] += 1
                yield name, out

        # A generator, not a list: load_weights consumes lazily, so each merged
        # tensor is freed after its copy instead of holding the whole model
        # (7.3 GB on a 4B) resident alongside the base snapshot.
        loaded = model.load_weights(merged())
        self._sync_count += 1

        # First sync doubles as a wiring check: if the HF->vLLM name mapping
        # were wrong, load_weights would quietly load nothing and the policy
        # would sample from the un-updated base forever.
        if self._sync_count == 1:
            n = len(loaded) if loaded is not None else None
            if n == 0:
                raise RuntimeError(
                    f"load_weights accepted 0 of {count[0]} tensors -- the "
                    f"HF->vLLM parameter name mapping is wrong for this arch"
                )
            return {"merged": count[0], "loaded": n, "fp32_merge": fp32_merge}
        return {"merged": count[0]}

    def reset_to_base(self):
        """Write the pristine base snapshot back into the live model."""
        if self._base is None:
            raise RuntimeError("init_lora_sync must be called before reset_to_base")
        self._model().load_weights([(n, w.clone()) for n, w in self._base.items()])
        return {"restored": len(self._base)}

    def lora_sync_stats(self):
        return {"syncs": self._sync_count, "base_loaded": self._base is not None}


def _resolve_model_dir(model_dir: str) -> str:
    """Accept a local path or an HF repo id; return a local directory."""
    if os.path.isdir(model_dir):
        return model_dir
    from huggingface_hub import snapshot_download

    return snapshot_download(
        model_dir, allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"]
    )
