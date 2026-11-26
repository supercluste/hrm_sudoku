import math
import torch
from torch.optim.optimizer import Optimizer


class AdamATan2(Optimizer):
    """
    Pure-PyTorch implementation of Adam-atan2 (Everett et al., 2024).
    Replaces m / (sqrt(v) + eps) with atan2(m_hat, sqrt(v_hat)) for
    epsilon-free, bounded, scale-invariant updates in low precision (bf16).
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamATan2 does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)

                state["step"] += 1
                step = state["step"]

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                grad_f32 = grad.float()
                exp_avg.lerp_(grad_f32, 1.0 - beta1)
                exp_avg_sq.lerp_(grad_f32.square(), 1.0 - beta2)

                bc1 = 1.0 - beta1 ** step
                bc2 = 1.0 - beta2 ** step

                m_hat = exp_avg / bc1
                v_hat_sqrt = (exp_avg_sq / bc2).sqrt()

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                update = torch.atan2(m_hat, v_hat_sqrt)
                p.add_(update.to(p.dtype), alpha=-lr)

        return loss
