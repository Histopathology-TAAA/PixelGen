# --------------------------------------------------------
# UNet_I2I: UNet denoiser for Image-to-Image translation
# Drop-in replacement for JiT_I2I with the I2IREPATrainer
# - Same forward(x, t, y, return_layer, return_last) signature
# - REPA-compatible bottleneck feature extraction
# - final_layer.linear.weight for adaptive perceptual weight
# - 6-channel input: 3 noisy target + 3 condition (H&E)
# --------------------------------------------------------
import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding followed by a 2-layer MLP."""

    def __init__(self, dim, hidden_dim, frequency_embedding_size=256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @staticmethod
    def sinusoidal_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        emb = self.sinusoidal_embedding(t, self.frequency_embedding_size)
        return self.mlp(emb)


class ResBlock(nn.Module):
    """Residual block with timestep conditioning (scale + shift via AdaGN)."""

    def __init__(self, in_ch, out_ch, time_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, 2 * out_ch),
        )
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.skip_proj = (
            nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x, t_emb):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        # AdaGN: scale & shift from timestep
        scale, shift = self.time_proj(t_emb)[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.skip_proj(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention for 2-D spatial features (QKV via Linear)."""

    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Linear(channels, 3 * channels)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).reshape(B, C, -1).transpose(1, 2)  # (B, N, C)
        qkv = (
            self.qkv(h)
            .reshape(B, -1, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # each (B, heads, N, head_dim)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, -1, C)  # (B, N, C)
        out = self.proj(out).transpose(1, 2).reshape(B, C, H, W)
        return x + out


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class FinalLayer(nn.Module):
    """
    Output layer.  Exposes `.linear.weight` so the I2IREPATrainer can compute
    adaptive perceptual weights via ``net.final_layer.linear.weight``.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, in_channels)
        self.act = nn.SiLU()
        # Named ``linear`` for trainer compatibility
        self.linear = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x):
        return self.linear(self.act(self.norm(x)))


# ---------------------------------------------------------------------------
# UNet_I2I
# ---------------------------------------------------------------------------

class UNet_I2I(nn.Module):
    """
    UNet denoiser for Image-to-Image flow-matching.

    * Condition image (H&E) is concatenated channel-wise with the noisy target
      (IHC) at the input – same semantics as JiT_I2I.
    * Timestep conditioning via AdaGN (scale + shift in every ResBlock).
    * Self-attention at configurable spatial resolutions.
    * REPA-compatible: when ``return_layer`` is not None, the bottleneck
      features are adaptively pooled to match DINOv2's spatial grid (16×16
      for 256 px input with patch_size 16) and returned as (B, N, C).
    * ``final_layer.linear.weight`` exposed for the adaptive perceptual
      weight computation in I2IREPATrainer.

    Parameters
    ----------
    input_size : int
        Spatial resolution of input images (default 256).
    in_channels : int
        Channels of the noisy target image (default 3).
    cond_channels : int
        Channels of the condition image (default 3).
    base_channels : int
        Base channel count; multiplied by ``channel_mult`` at each level.
    channel_mult : tuple of int
        Channel multiplier per encoder level.  Length = number of levels.
        Downsampling happens between levels (so len − 1 down/up-samples).
    num_res_blocks : int
        Number of ResBlocks per encoder level (decoder gets +1 for skip).
    attention_resolutions : tuple of int
        Spatial resolutions at which self-attention is applied.
    num_heads : int
        Number of heads for multi-head self-attention.
    dropout : float
        Dropout rate inside ResBlocks.
    repa_patch_size : int
        Patch size of the DINO teacher.  Used only to determine the target
        spatial size when pooling bottleneck features for REPA alignment.
    """

    def __init__(
        self,
        input_size: int = 256,
        in_channels: int = 3,
        cond_channels: int = 3,
        base_channels: int = 128,
        channel_mult: tuple = (1, 2, 3, 4),
        num_res_blocks: int = 2,
        attention_resolutions: tuple = (32,),
        num_heads: int = 8,
        dropout: float = 0.1,
        repa_patch_size: int = 16,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.cond_channels = cond_channels
        self.out_channels = in_channels
        self.input_size = input_size
        self.base_channels = base_channels
        self.channel_mult = list(channel_mult)
        self.num_res_blocks = num_res_blocks
        self.repa_patch_size = repa_patch_size

        time_dim = base_channels * 4
        total_in_ch = in_channels + cond_channels

        # ---- timestep embedding ----
        self.time_embed = TimestepEmbedding(base_channels, time_dim)

        # ---- input conv ----
        self.input_conv = nn.Conv2d(total_in_ch, base_channels, 3, padding=1)

        # ---- encoder (flat ModuleLists) ----
        self.encoder_blocks = nn.ModuleList()
        self.encoder_attns = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        ch = base_channels
        self._skip_channels = [ch]  # used only at build time
        current_res = input_size

        for level_idx, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.encoder_blocks.append(ResBlock(ch, out_ch, time_dim, dropout))
                self.encoder_attns.append(
                    SelfAttention(out_ch, num_heads)
                    if current_res in attention_resolutions
                    else nn.Identity()
                )
                ch = out_ch
                self._skip_channels.append(ch)
            if level_idx < len(channel_mult) - 1:
                self.downsamples.append(Downsample(ch))
                self._skip_channels.append(ch)
                current_res //= 2

        # ---- bottleneck ----
        self.mid_res1 = ResBlock(ch, ch, time_dim, dropout)
        self.mid_attn = SelfAttention(ch, num_heads)
        self.mid_res2 = ResBlock(ch, ch, time_dim, dropout)

        self.bottleneck_channels = ch
        self.bottleneck_res = current_res

        # ---- decoder (flat ModuleLists) ----
        self.decoder_blocks = nn.ModuleList()
        self.decoder_attns = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for level_idx in reversed(range(len(channel_mult))):
            out_ch = base_channels * channel_mult[level_idx]
            for _ in range(num_res_blocks + 1):
                skip_ch = self._skip_channels.pop()
                self.decoder_blocks.append(
                    ResBlock(ch + skip_ch, out_ch, time_dim, dropout)
                )
                self.decoder_attns.append(
                    SelfAttention(out_ch, num_heads)
                    if current_res in attention_resolutions
                    else nn.Identity()
                )
                ch = out_ch
            if level_idx > 0:
                self.upsamples.append(Upsample(ch))
                current_res *= 2

        assert (
            len(self._skip_channels) == 0
        ), f"Skip channel mismatch: {len(self._skip_channels)} remaining"

        # ---- output ----
        self.final_layer = FinalLayer(ch, self.out_channels)

        self._initialize_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-init final output for residual-like startup (model predicts 0 initially)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

        # Zero-init AdaGN scale/shift projections for stable start
        all_res = (
            list(self.encoder_blocks)
            + [self.mid_res1, self.mid_res2]
            + list(self.decoder_blocks)
        )
        for block in all_res:
            if hasattr(block, "time_proj"):
                nn.init.zeros_(block.time_proj[-1].weight)
                nn.init.zeros_(block.time_proj[-1].bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _extract_repa_features(self, h):
        """Pool bottleneck map to DINO's spatial grid → (B, N_tokens, C)."""
        repa_size = self.input_size // self.repa_patch_size  # 256//16 = 16
        feat = F.adaptive_avg_pool2d(h, (repa_size, repa_size))
        return feat.flatten(2).transpose(1, 2)  # (B, 256, C)

    def forward(self, x, t, y, return_layer=None, return_last=False):
        """
        Forward pass – same signature as JiT_I2I.

        Args:
            x: (B, C, H, W) noisy target image
            t: (B,) diffusion timesteps
            y: (B, C, H, W) condition image (H&E)
            return_layer: if not None, also return bottleneck features
                          for REPA alignment (value is ignored; kept for
                          interface compat with JiT_I2I)
            return_last: additionally return pre-output features

        Returns:
            output: (B, out_C, H, W) predicted clean target
            feat  : (B, N, bottleneck_C) only when return_layer is not None
            last  : (B, N, base_C)       only when return_last is True
        """
        # --- concatenate condition with noisy target ---
        h = torch.cat([x, y], dim=1)  # (B, 6, H, W)
        h = self.input_conv(h)
        t_emb = self.time_embed(t)

        # --- encoder ---
        skips = [h]
        enc_idx = 0
        down_idx = 0
        for level_idx in range(len(self.channel_mult)):
            for _ in range(self.num_res_blocks):
                h = self.encoder_blocks[enc_idx](h, t_emb)
                h = self.encoder_attns[enc_idx](h)
                skips.append(h)
                enc_idx += 1
            if level_idx < len(self.channel_mult) - 1:
                h = self.downsamples[down_idx](h)
                skips.append(h)
                down_idx += 1

        # --- bottleneck ---
        h = self.mid_res1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_res2(h, t_emb)

        if return_layer is not None:
            feat = self._extract_repa_features(h)

        # --- decoder ---
        dec_idx = 0
        up_idx = 0
        for level_idx in reversed(range(len(self.channel_mult))):
            for _ in range(self.num_res_blocks + 1):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = self.decoder_blocks[dec_idx](h, t_emb)
                h = self.decoder_attns[dec_idx](h)
                dec_idx += 1
            if level_idx > 0:
                h = self.upsamples[up_idx](h)
                up_idx += 1

        if return_last:
            pre_out = h.flatten(2).transpose(1, 2)

        # --- output ---
        output = self.final_layer(h)

        if return_layer is not None:
            if return_last:
                return output, feat, pre_out
            return output, feat
        return output


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def UNet_I2I_B(**kwargs):
    """
    Base UNet I2I (~90M params).
    4 encoder levels, attention at 32×32, bottleneck at 32×32.
    Good for small-to-medium datasets (< 10 K images).
    """
    return UNet_I2I(
        base_channels=128,
        channel_mult=(1, 2, 3, 4),
        num_res_blocks=2,
        attention_resolutions=(32,),
        num_heads=8,
        dropout=0.1,
        **kwargs,
    )


def UNet_I2I_S(**kwargs):
    """
    Small UNet I2I (~35M params).
    Lighter variant for very small datasets (< 5 K images).
    """
    return UNet_I2I(
        base_channels=96,
        channel_mult=(1, 2, 3, 4),
        num_res_blocks=2,
        attention_resolutions=(32,),
        num_heads=8,
        dropout=0.1,
        **kwargs,
    )


def UNet_I2I_L(**kwargs):
    """
    Large UNet I2I (~200M params).
    5 encoder levels, attention at 32×32, bottleneck at 16×16.
    """
    return UNet_I2I(
        base_channels=128,
        channel_mult=(1, 1, 2, 3, 4),
        num_res_blocks=2,
        attention_resolutions=(32, 16),
        num_heads=8,
        dropout=0.1,
        **kwargs,
    )
