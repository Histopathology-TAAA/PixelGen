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
from typing import Tuple
import torch.nn.functional as F
from tqdm import tqdm


class StarDiffScheduler:
    """
    Flow Matching (Rectified Flow) scheduler for StarDiff Dual-Path.
    Timesteps t range from 0.0 (pure noise) to 1.0 (clean image).
    """

    def __init__(
        self,
        num_timesteps: int = 50,  # Flow Matching needs far fewer steps!
        restoration_weight: float = 0.5,
        he_init_alpha:float = 0.3,
        **kwargs # Ignore DDPM kwargs
    ):
        self.num_timesteps = num_timesteps
        self.restoration_weight = restoration_weight

        # Timesteps sequence for inference: [0.0, ..., 1.0]
        self.timesteps = torch.linspace(0.0, 1.0, num_timesteps + 1)
        
        # Restoration schedule: gamma_t grows linearly from 0 to restoration_weight
        self.restoration_schedule = torch.linspace(0.0, restoration_weight, num_timesteps + 1)
        self.he_init_alpha = he_init_alpha

    def to(self, device):
        """Move scheduler tensors to device."""
        self.timesteps = self.timesteps.to(device)
        self.restoration_schedule = self.restoration_schedule.to(device)
        return self

    # ── Forward process (Training) ──

    def q_sample(self, x_1, t, noise=None, restoration_residual=None):
        """
        Flow Matching Forward Process.
        x_1 is the clean target image.
        t is a float tensor in [0, 1].
        """
        if noise is None:
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
               use_restoration=True, use_noise=True, progress=True, he_init_alpha=None):
        """
        Euler Sampling from t=0 (noise) to t=1 (clean image).
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
            
            # Convert both to velocity using the Flow Matching formula: v = (x_1 - x_t) / (1 - t)
            # This ensures the model correctly uses its pretrained image-generation priors.
            denom = (1.0 - t_batch.view(-1, 1, 1, 1)).clamp_min(1e-3)
            velocity_noise = (x1_noise_pred - x) / denom
            velocity_rest = (x1_rest_pred - x) / denom
            
            # Combine paths
            # StarDiff concept: v_eff = v_noise + gamma_t * v_restoration
            gamma_t = self.restoration_schedule[i].item()
            
            v_eff = torch.zeros_like(x)
            if use_noise:
                v_eff += velocity_noise
            if use_restoration:
                v_eff += gamma_t * velocity_rest
                
            # Euler step
            x = x + v_eff * dt

        return x
