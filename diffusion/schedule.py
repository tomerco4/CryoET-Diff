# diffusion/schedule.py

from typing import Literal, Tuple
import numpy as np
import torch
from torch import Tensor


def make_beta_schedule(
    T: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    schedule: Literal["linear", "cosine"] = "linear",
) -> np.ndarray:
    """
    Create a beta schedule for DDPM.

    Returns:
        betas: (T,) numpy array
    """
    if schedule == "linear":
        return np.linspace(beta_start, beta_end, T, dtype=np.float64)
    elif schedule == "cosine":
        # from Nichol & Dhariwal, 2021 (simplified)
        s = 0.008
        steps = np.arange(T + 1, dtype=np.float64)
        t = steps / T
        alphas_cumprod = np.cos((t + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = np.clip(betas, 0.0001, 0.999)
        return betas
    else:
        raise ValueError(f"Unknown schedule type: {schedule}")


class DiffusionSchedule:
    """
    Holds betas, alphas, cumulative products, etc. for DDPM.
    """

    def __init__(
        self,
        T: int,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        schedule: str = "linear",
        device: str | torch.device = "cpu",
    ):
        self.T = int(T)
        betas_np = make_beta_schedule(T, beta_start, beta_end, schedule=schedule)
        self.betas = torch.tensor(betas_np, dtype=torch.float32, device=device)  # (T,)
        self.alphas = 1.0 - self.betas                                       # (T,)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)              # (T,)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]], dim=0
        )                                                                     # (T,)

        # Precompute posterior coefficients for sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self.posterior_variance = (
            self.betas
            * (1.0 - self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        # for numerical stability; clamp minimum
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-20)
        self.posterior_log_variance = torch.log(self.posterior_variance)

        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x0: Tensor, t: Tensor, noise: Tensor | None = None) -> Tensor:
        """
        Sample from q(x_t | x_0).

        Args:
            x0: (B, 1, D, H, W)
            t:  (B,) timesteps in [0, T-1] (0-based)
            noise: optional noise tensor; if None, sampled from N(0, I).

        Returns:
            x_t: (B, 1, D, H, W)
        """
        if noise is None:
            noise = torch.randn_like(x0)

        # gather alphas_cumprod[t] for each element in batch
        # t is 0..T-1
        alphas_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod_t)
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1.0 - alphas_cumprod_t)
        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise

    def p_sample_x0(
        self,
        x_t: Tensor,
        t: Tensor,
        x0_pred: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute p(x_{t-1} | x_t, x0_pred) parameters (mean, log_var).

        Args:
            x_t:      (B, 1, D, H, W)
            t:        (B,) timesteps, 0-based
            x0_pred:  (B, 1, D, H, W)

        Returns:
            mean:     (B, 1, D, H, W)
            log_var:  (B, 1, 1, 1, 1)
        """
        # gather
        betas_t = self.betas[t].view(-1, 1, 1, 1, 1)
        alpha_t = self.alphas[t].view(-1, 1, 1, 1, 1)
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        alpha_cumprod_prev_t = self.alphas_cumprod_prev[t].view(-1, 1, 1, 1, 1)

        posterior_var_t = self.posterior_variance[t].view(-1, 1, 1, 1, 1)
        posterior_log_var_t = self.posterior_log_variance[t].view(-1, 1, 1, 1, 1)

        coef1 = (
            betas_t * torch.sqrt(alpha_cumprod_prev_t) / (1.0 - alpha_cumprod_t)
        )
        coef2 = (
            (1.0 - alpha_cumprod_prev_t) * torch.sqrt(alpha_t) / (1.0 - alpha_cumprod_t)
        )

        mean = coef1 * x0_pred + coef2 * x_t
        return mean, posterior_log_var_t
