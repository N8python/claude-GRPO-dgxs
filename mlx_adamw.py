"""AdamW matching mlx.optimizers.AdamW, bias correction included.

WHY THIS EXISTS

`mlx.optimizers.AdamW` defaults to **bias_correction=False**. `torch.optim.AdamW`
always applies it and offers no way to turn it off. That is not a cosmetic
difference: with all-equal gradients the ratio of the two update magnitudes is

    (1 - b1^t) / sqrt(1 - b2^t)

which is 3.2x at t=1, 6.5x at t=10, and still 5.0x at t=39. So an un-bias-
corrected optimizer takes steps several times larger than a bias-corrected one
at the same nominal learning rate, for the entire length of a short run.

Every learning rate in the MLX reference -- 1e-4 for SFT, 3e-6 for GRPO, and
the 50-step warmup tuned around it -- was found against the un-bias-corrected
optimizer. Handing those numbers to torch.optim.AdamW silently under-trains.
Measured here on the 100-trace SFT: 25.8 / 26.4 / 25.8% exact across three
seeds with torch AdamW, against the reference's 33.4-34.5%.

Rather than re-tune every hyperparameter for a different optimizer, port the
optimizer. Then the reference's recipe means what it says.

Faithful to mlx/optimizers/optimizers.py (Adam.apply_single + AdamW.apply_single):
    m = b1*m + (1-b1)*g
    v = b2*v + (1-b2)*g^2
    w = w * (1 - lr*wd)                       # decoupled, applied BEFORE the step
    w = w - lr * m / (sqrt(v) + eps)          # bias_correction=False
"""

import torch


class MLXAdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        bias_correction: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"invalid lr: {lr}")
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            bias_correction=bias_correction,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps, wd = group["eps"], group["weight_decay"]
            bias_correction = group["bias_correction"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["step"] += 1
                t = state["step"]
                m, v = state["m"], state["v"]

                m.mul_(b1).add_(g, alpha=1.0 - b1)
                v.mul_(b2).addcmul_(g, g, value=1.0 - b2)

                # MLX applies decoupled weight decay to the parameter BEFORE
                # the Adam step (AdamW.apply_single passes `parameter * (1 -
                # lr*wd)` into Adam), not after. Order matters at large lr.
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                if bias_correction:
                    num = m / (1.0 - b1**t)
                    den = (v / (1.0 - b2**t)).sqrt_().add_(eps)
                else:
                    num = m
                    den = v.sqrt().add_(eps)
                p.addcdiv_(num, den, value=-lr)

        return loss
