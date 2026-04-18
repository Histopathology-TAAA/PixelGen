"""
StarDiff Flow Matching (Rectified Flow) Scheduler with dual pathways.

Adapted StarDiff concept for Flow Matching:
- Forward (Training): x_t = (1-t)*N(0,1) + t*x_1
- Velocity Target: v = x_1 - noise
- Inference (Euler): x_{t+dt} = x_t + v_pred * dt

Dual Path Inference:
v_eff = v_noise + gamma_t * v_restoration
"""
import torch
import torch.nn.functional as F
from typing import Tuple
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment


class StarDiffScheduler:
    """
    Flow Matching (Rectified Flow) scheduler for StarDiff Dual-Path.
    Timesteps t range from 0.0 (pure noise) to 1.0 (clean image).
    """

    def __init__(
        self,
        num_timesteps: int = 50,  # Flow Matching needs far fewer steps!
        restoration_weight: float = 0.5,
        schedule_type: str = "cosine",
        integration_method: str = "heun",
        he_init_alpha: float = 0.3,
        late_t_threshold: float = 0.95,
        use_ot_coupling: bool = True,
        ot_feature_size: int = 32,
        **kwargs # Ignore DDPM kwargs
    ):
        self.num_timesteps = num_timesteps
        self.restoration_weight = restoration_weight
        self.schedule_type = schedule_type
        self.integration_method = integration_method
        self.he_init_alpha = he_init_alpha
        self.late_t_threshold = late_t_threshold
        self.use_ot_coupling = use_ot_coupling
        self.ot_feature_size = ot_feature_size

        # Timesteps sequence for inference: [0.0, ..., 1.0]
        self.timesteps = torch.linspace(0.0, 1.0, num_timesteps + 1)

        # Restoration schedule
        t = torch.linspace(0.0, 1.0, num_timesteps + 1)
        if schedule_type == "cosine":
            self.restoration_schedule = restoration_weight * torch.sin(t * torch.pi / 2) ** 2
        elif schedule_type == "linear":
            self.restoration_schedule = torch.linspace(0.0, restoration_weight, num_timesteps + 1)
        elif schedule_type == "constant":
            self.restoration_schedule = torch.full((num_timesteps + 1,), restoration_weight)
        else:
            raise ValueError(f"Unknown schedule_type '{schedule_type}'. Choose from ['cosine', 'linear', 'constant']")

    def to(self, device):
        """Move scheduler tensors to device."""
        self.timesteps = self.timesteps.to(device)
        self.restoration_schedule = self.restoration_schedule.to(device)
        return self

    # ── Forward process (Training) ──

    def ot_coupled_noise(self, x_1: torch.Tensor) -> torch.Tensor:
        """
        Minibatch OT coupling between data and Gaussian noise.
        Uses pooled features for tractable cost on high resolutions.
        """
        b = x_1.shape[0]
        noise = torch.randn_like(x_1)
        if b <= 1:
            return noise

        pool_size = self.ot_feature_size
        if pool_size is not None and pool_size > 0:
            x_feat = F.adaptive_avg_pool2d(x_1, (pool_size, pool_size))
            n_feat = F.adaptive_avg_pool2d(noise, (pool_size, pool_size))
        else:
            x_feat = x_1
            n_feat = noise

        x_flat = x_feat.view(b, -1).detach().cpu()
        n_flat = n_feat.view(b, -1).detach().cpu()
        cost = torch.cdist(x_flat, n_flat).numpy()
        _, col_idx = linear_sum_assignment(cost)
        col_idx = torch.as_tensor(col_idx, device=x_1.device, dtype=torch.long)
        return noise.index_select(0, col_idx)

    def q_sample(self, x_1, t, noise=None, restoration_residual=None):
        """
        Flow Matching Forward Process.
        x_1 is the clean target image.
        t is a float tensor in [0, 1].
        """
        if noise is None:
            if self.use_ot_coupling:
                noise = self.ot_coupled_noise(x_1)
            else:
                noise = torch.randn_like(x_1)

        # t must be broadcastable to x_1
        t_expand = t.view(-1, 1, 1, 1).to(x_1.device)
        
        # Rectified Flow: straight line from noise to x_1
        x_t = (1.0 - t_expand) * noise + t_expand * x_1

        # Calculate the ground truth velocity vector
        v_target = x_1 - noise
        
        return x_t, v_target

    # ── Reverse process (Inference) ──

    @torch.no_grad()
    def sample(self, model, condition, shape, device="cuda",
               he_init_alpha=None, use_restoration=True, use_noise=True, progress=True):
        """
        Flow Matching sampling from t=0 (noise/source blend) to t=1 (clean image).
        """
        model.eval()
        if he_init_alpha is None:
            he_init_alpha = self.he_init_alpha

        noise = torch.randn(shape, device=device)
        condition_resized = condition
        if condition.shape[-2:] != (shape[-2], shape[-1]):
            condition_resized = F.interpolate(condition, size=(shape[-2], shape[-1]), mode="bilinear", align_corners=False)
        x = (1.0 - he_init_alpha) * noise + he_init_alpha * condition_resized

        steps = self.timesteps.to(device)
        
        iterator = range(self.num_timesteps)
        if progress:
            iterator = tqdm(iterator, desc="Sampling (Flow Matching)")

        for i in iterator:
            t_cur = steps[i]
            t_next = steps[i+1]
            dt = t_next - t_cur

            t_batch = torch.full((shape[0],), t_cur.item(), device=device, dtype=torch.float32)

            # Predict components
            # Both heads predict clean image representation x_1 to leverage ImageNet pretraining
            x1_noise_pred, x1_rest_pred = model(x, t_batch, condition)

            # Combine paths
            # StarDiff concept: v_eff = v_noise + gamma_t * v_restoration
            gamma_t = self.restoration_schedule[i].item()

            # Numerical stabilization near t -> 1: direct x1 blending instead of velocity division
            if t_cur.item() >= self.late_t_threshold:
                x1_eff = torch.zeros_like(x)
                if use_noise:
                    x1_eff += x1_noise_pred
                if use_restoration:
                    x1_eff += gamma_t * x1_rest_pred

                alpha = dt / (1.0 - t_cur + 1e-6)
                alpha = torch.clamp(alpha, 0.0, 1.0)
                x = (1.0 - alpha) * x + alpha * x1_eff
                continue

            # Predictor at current time
            denom = (1.0 - t_batch.view(-1, 1, 1, 1)).clamp_min(1e-3)
            velocity_noise = (x1_noise_pred - x) / denom
            velocity_rest = (x1_rest_pred - x) / denom

            v1 = torch.zeros_like(x)
            if use_noise:
                v1 += velocity_noise
            if use_restoration:
                v1 += gamma_t * velocity_rest

            if self.integration_method == "euler" or i == self.num_timesteps - 1:
                x = x + v1 * dt
                continue

            if self.integration_method != "heun":
                raise ValueError(f"Unknown integration_method '{self.integration_method}'. Choose from ['euler', 'heun']")

            # Heun corrector at next time
            x_pred = x + v1 * dt
            t_batch_next = torch.full((shape[0],), t_next.item(), device=device, dtype=torch.float32)
            x1_noise_pred2, x1_rest_pred2 = model(x_pred, t_batch_next, condition)
            denom2 = (1.0 - t_batch_next.view(-1, 1, 1, 1)).clamp_min(1e-3)
            velocity_noise2 = (x1_noise_pred2 - x_pred) / denom2
            velocity_rest2 = (x1_rest_pred2 - x_pred) / denom2
            gamma_next = self.restoration_schedule[i + 1].item()

            v2 = torch.zeros_like(x)
            if use_noise:
                v2 += velocity_noise2
            if use_restoration:
                v2 += gamma_next * velocity_rest2

            x = x + 0.5 * (v1 + v2) * dt

        return x
