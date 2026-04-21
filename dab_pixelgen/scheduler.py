"""
DABFlowScheduler: Rectified Flow Matching on 1-channel DAB density space.

Forward (training):
    x_t = (1 - t) * x_0 + t * dab_gt
    v_target = dab_gt - x_0

where x_0 is the starting distribution (noise or H&E-warm-started noise).

The model predicts the clean DAB image x_1 directly (PixelGen convention).
Velocity is recovered as:  v = (x_1_pred - x_t) / (1 - t).clamp(1e-3)

Sampling: Euler integration t=0 -> t=1, single path.
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Optional


class DABFlowScheduler:
    """
    Flow Matching (Rectified Flow) scheduler for 1-channel DAB density.

    t=0 = pure noise / warm start
    t=1 = clean DAB density
    """

    def __init__(
        self,
        num_timesteps: int = 50,
        he_init_alpha: float = 0.3,
    ):
        """
        Args:
            num_timesteps: Number of Euler integration steps during sampling.
            he_init_alpha:  Warm-start coefficient mixing H-density into noise.
                            x_0 = (1 - alpha) * N(0,1) + alpha * h_he_density
                            Set to 0 for pure Gaussian start.
        """
        self.num_timesteps = num_timesteps
        self.he_init_alpha = he_init_alpha
        self.timesteps = torch.linspace(0.0, 1.0, num_timesteps + 1)

    def to(self, device):
        self.timesteps = self.timesteps.to(device)
        return self

    # ── Training (forward process) ────────────────────────────────────────────

    def q_sample(
        self,
        dab_gt: torch.Tensor,    # [B, 1, H, W]  clean DAB density
        t:      torch.Tensor,    # [B]            float in [0, 1]
        x_0:    Optional[torch.Tensor] = None,   # [B, 1, H, W] starting noise
    ):
        """
        Sample a noisy intermediate x_t along the rectified flow path.

        Returns:
            x_t:      [B, 1, H, W]  noisy DAB at timestep t
            v_target: [B, 1, H, W]  ground truth velocity = dab_gt - x_0
        """
        if x_0 is None:
            x_0 = torch.randn_like(dab_gt)

        t_exp = t.view(-1, 1, 1, 1).to(dab_gt.device)
        x_t = (1.0 - t_exp) * x_0 + t_exp * dab_gt
        v_target = dab_gt - x_0
        return x_t, v_target

    def make_x0(
        self,
        dab_gt:       torch.Tensor,    # [B, 1, H, W]  used for shape / device only
        h_he_density: Optional[torch.Tensor] = None,  # [B, 1, H, W]
        alpha:        Optional[float] = None,
    ) -> torch.Tensor:
        """
        Build the starting distribution x_0 for training or sampling.

        x_0 = (1 - alpha) * noise + alpha * h_he_density
        """
        if alpha is None:
            alpha = self.he_init_alpha
        noise = torch.randn_like(dab_gt)
        if h_he_density is not None and alpha > 0.0:
            cond = F.interpolate(
                h_he_density,
                size=dab_gt.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ) if h_he_density.shape[-2:] != dab_gt.shape[-2:] else h_he_density
            return (1.0 - alpha) * noise + alpha * cond
        return noise

    # ── Sampling (reverse process) ────────────────────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        model,                               # DABPixelGenModel
        he_condition:    torch.Tensor,       # [B, 3, H, W]
        shape:           tuple,              # (B, 1, H, W)
        device:          str = "cuda",
        h_he_density:    Optional[torch.Tensor] = None,  # [B, 1, H, W] for warm start
        he_init_alpha:   Optional[float] = None,
        progress:        bool = True,
    ):
        """
        Run Euler sampling from t=0 (noise) to t=1 (clean DAB).

        Returns:
            dab_pred:     [B, 1, H, W]   final predicted DAB density
            h_norm_params:[B, 2]          H-normalization params from last step
        """
        model.eval()
        alpha = he_init_alpha if he_init_alpha is not None else self.he_init_alpha

        # Build starting x_0
        dummy = torch.zeros(shape, device=device)
        h_he_dev = h_he_density.to(device) if h_he_density is not None else None
        x = self.make_x0(dummy, h_he_dev, alpha)

        he_cond = he_condition.to(device)
        steps = self.timesteps.to(device)

        iterator = range(self.num_timesteps)
        if progress:
            iterator = tqdm(iterator, desc="DAB Sampling (Flow Matching)")

        h_norm_params = None
        for i in iterator:
            t_cur  = steps[i]
            t_next = steps[i + 1]
            dt = t_next - t_cur

            t_batch = torch.full((shape[0],), t_cur.item(), device=device)

            # Model predicts clean x_1 and H-norm params
            x1_pred, h_norm_params = model(x, t_batch, he_cond)

            # Rectified flow velocity: v = (x_1 - x_t) / (1 - t)
            denom = (1.0 - t_batch.view(-1, 1, 1, 1)).clamp_min(1e-3)
            velocity = (x1_pred - x) / denom

            # Euler step
            x = x + velocity * dt

        return x, h_norm_params
