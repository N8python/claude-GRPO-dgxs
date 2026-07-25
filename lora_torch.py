"""Hand-rolled LoRA for the torch side, keyed by HF checkpoint names.

Deliberately not peft. Two reasons:

  1. Adapter keys must be HF module paths (``model.layers.0.self_attn.q_proj``)
     because Bob merges them against the base checkpoint's own tensors and
     hands the result to vLLM's ``load_weights``, which speaks HF names. peft
     mangles names (``base_model.model.…lora_A.default.weight``) and we would
     be un-mangling them on the other machine.
  2. This repo already lives downstream of a documented transformers/peft/
     unsloth version minefield. One fewer pinned dependency is worth 60 lines.

Init and forward mirror mlx_lm's LoRALinear exactly (A ~ U(-1/sqrt(in),
1/sqrt(in)), B = 0, y = base(x) + scale * (x @ A.T) @ B.T) so an adapter
trained here behaves like one trained by the MLX reference.

The LoRA branch runs in fp32 while the frozen base runs in bf16. That is the
same split Bob uses when it merges (fp32 accumulate, cast once at the end),
which keeps the two machines' effective weights as close as this architecture
allows. What's left over is measured, not assumed -- see the mismatch
diagnostics in train_grpo.py.
"""

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# all linears except embeddings / lm_head, matching the reference's lora_util
LORA_KEYS = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a trainable rank-r branch."""

    def __init__(self, base: nn.Linear, rank: int, scale: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = rank
        self.scale = scale
        self.dropout = dropout
        out_features, in_features = base.weight.shape
        bound = 1.0 / math.sqrt(in_features)
        self.lora_a = nn.Parameter(
            torch.empty(rank, in_features, dtype=torch.float32).uniform_(-bound, bound)
        )
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank, dtype=torch.float32))

    def forward(self, x):
        y = self.base(x)
        z = x if self.dropout == 0.0 else F.dropout(x, self.dropout, self.training)
        z = F.linear(F.linear(z.to(self.lora_a.dtype), self.lora_a), self.lora_b)
        return y + self.scale * z.to(y.dtype)


def merge_delta(base_weight, lora_a, lora_b, scale):
    """W_base + scale * (B @ A), accumulated in fp32 then cast back.

    Shared by the trainer (for checkpoint fusing) and by Bob's worker
    extension, so both machines merge with identical numerics.
    """
    delta = torch.mm(lora_b.float(), lora_a.float()).mul_(scale)
    return delta.add_(base_weight.float()).to(base_weight.dtype)


def _set_submodule(root: nn.Module, path: str, new: nn.Module):
    parent_path, _, leaf = path.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    setattr(parent, leaf, new)


def target_paths(model: nn.Module):
    """HF module paths of the linears LoRA is applied to (pre-wrapping)."""
    return [
        name
        for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear) and any(name.endswith(k) for k in LORA_KEYS)
    ]


def apply_lora(model: nn.Module, rank: int, scale=None, dropout: float = 0.0):
    """Freeze the model and wrap every target linear. Returns lora_parameters.

    scale=None uses 32/rank (alpha=32) -- the convention the "LoRA wants ~10x
    lr" heuristic assumes. That heuristic is SFT-only; see train_grpo.py.
    """
    if scale is None:
        scale = 32.0 / rank
    for p in model.parameters():
        p.requires_grad_(False)

    paths = target_paths(model)
    if not paths:
        raise RuntimeError(
            f"no LoRA targets found; expected module names ending in one of {LORA_KEYS}"
        )
    for path in paths:
        base = model.get_submodule(path)
        _set_submodule(model, path, LoRALinear(base, rank, scale, dropout))

    lora_parameters = {
        "rank": rank,
        "scale": scale,
        "dropout": dropout,
        "keys": LORA_KEYS,
        "targets": paths,
    }
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(
        f"LoRA rank {rank} scale {scale}: {n_train:,} trainable / "
        f"{n_total:,} total ({100 * n_train / n_total:.3f}%) over {len(paths)} linears"
    )
    return lora_parameters


def lora_state_dict(model: nn.Module):
    """{'<hf module path>.lora_a': tensor, '...lora_b': tensor} on CPU fp32.

    These keys are the wire format between Alice and Bob: strip the suffix and
    append '.weight' to get the base checkpoint tensor to merge against.
    """
    out = {}
    for name, mod in model.named_modules():
        if isinstance(mod, LoRALinear):
            out[f"{name}.lora_a"] = mod.lora_a.detach().to("cpu", torch.float32)
            out[f"{name}.lora_b"] = mod.lora_b.detach().to("cpu", torch.float32)
    return out


def save_adapter_dir(out, model, lora_parameters, base_model):
    """Write adapter_config.json + adapters.safetensors."""
    from safetensors.torch import save_file

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    config = {
        "fine_tune_type": "lora",
        "lora_parameters": lora_parameters,
        "base_model": str(base_model),
        "format": "dgxGRPOProper-v1",
    }
    (out / "adapter_config.json").write_text(json.dumps(config, indent=2))
    save_file(lora_state_dict(model), str(out / "adapters.safetensors"))


def load_adapter(model: nn.Module, adapter_path):
    """Apply LoRA per the saved config, then load the saved weights.

    Returns lora_parameters. Mirrors mlx_lm's freeze-then-load_adapters flow.
    """
    from safetensors.torch import load_file

    adapter_path = Path(adapter_path)
    config = json.loads((adapter_path / "adapter_config.json").read_text())
    lp = config["lora_parameters"]
    lora_parameters = apply_lora(model, lp["rank"], lp["scale"], lp.get("dropout", 0.0))

    state = load_file(str(adapter_path / "adapters.safetensors"))
    modules = {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}
    missing = set(f"{n}.lora_a" for n in modules) - set(state)
    if missing:
        raise RuntimeError(
            f"adapter {adapter_path} is missing {len(missing)} tensors, "
            f"e.g. {sorted(missing)[:3]} -- base model mismatch?"
        )
    unexpected = set(state) - {f"{n}.{s}" for n in modules for s in ("lora_a", "lora_b")}
    if unexpected:
        raise RuntimeError(
            f"adapter {adapter_path} has {len(unexpected)} unexpected tensors, "
            f"e.g. {sorted(unexpected)[:3]}"
        )
    with torch.no_grad():
        for name, mod in modules.items():
            mod.lora_a.copy_(state[f"{name}.lora_a"].to(mod.lora_a.dtype))
            mod.lora_b.copy_(state[f"{name}.lora_b"].to(mod.lora_b.dtype))
    print(f"loaded adapter {adapter_path} ({len(modules)} linears)")
    return lora_parameters


def fuse_adapter(model: nn.Module):
    """Fold every LoRA branch into its base linear and unwrap, in place.

    Used to write a plain HF checkpoint that vLLM can serve with no adapter.
    """
    paths = [n for n, m in model.named_modules() if isinstance(m, LoRALinear)]
    with torch.no_grad():
        for path in paths:
            mod = model.get_submodule(path)
            mod.base.weight.data = merge_delta(
                mod.base.weight.data, mod.lora_a.data, mod.lora_b.data, mod.scale
            )
            _set_submodule(model, path, mod.base)
    return len(paths)
