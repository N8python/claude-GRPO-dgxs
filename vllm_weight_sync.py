"""vLLM worker extension: merge a LoRA update into the live model weights.

Runs inside the vLLM worker process (vLLM instantiates it via
``worker_extension_cls``, so ``self`` is the Worker and ``self.model_runner``
is available). Reached from the server process with ``collective_rpc``.

Why merge instead of using vLLM's native LoRA runtime:

  * The trainer must sample from *exactly* the weights it is about to
    differentiate. A merged weight is unambiguous; a punica-kernel LoRA path
    is a second numerical path to reconcile.
  * Hot-swapping an adapter every optimizer step means a fresh lora_int_id
    every step to defeat the adapter cache, which grows without bound.
  * Only ~22 MB crosses the wire either way (rank-8 A/B), because the merge
    happens here against a pristine base snapshot Bob holds locally. Sending
    merged weights instead would be ~2 GB per step.

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

    def apply_lora_update(self, payload: bytes, scale: float):
        """Merge A/B into the base snapshot and write into the live model."""
        from safetensors.torch import load as load_bytes

        if self._base is None:
            raise RuntimeError("init_lora_sync must be called before apply_lora_update")

        lora = load_bytes(payload)
        model = self._model()
        merged = []
        for name, w0 in self._base.items():
            key = name[: -len(".weight")]
            a = lora.get(f"{key}.lora_a")
            b = lora.get(f"{key}.lora_b")
            if a is None or b is None:
                raise RuntimeError(f"LoRA payload has no A/B for {key}")
            delta = torch.mm(
                b.to(self._base_device, torch.float32),
                a.to(self._base_device, torch.float32),
            ).mul_(scale)
            merged.append((name, delta.add_(w0.float()).to(w0.dtype)))

        loaded = model.load_weights(merged)
        self._sync_count += 1

        # First sync doubles as a wiring check: if the HF->vLLM name mapping
        # were wrong, load_weights would quietly load nothing and the policy
        # would sample from the un-updated base forever.
        if self._sync_count == 1:
            n = len(loaded) if loaded is not None else None
            if n == 0:
                raise RuntimeError(
                    f"load_weights accepted 0 of {len(merged)} tensors -- the "
                    f"HF->vLLM parameter name mapping is wrong for this arch"
                )
            return {"merged": len(merged), "loaded": n}
        return {"merged": len(merged)}

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
