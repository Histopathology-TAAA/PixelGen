"""
GAN DABPixelGen Generator.

Conditional UNet: H&E [3ch] → DAB density [1ch] + IHC RGB [3ch]

Architecture:
  Stem:          Conv(3 → C), GroupNorm, SiLU
  Encoder:       `len(channel_mult)` levels, each with `num_res_blocks`
                 GenResBlocks + optional SelfAttn, then a strided conv
                 downsample between levels
  Bottleneck:    2 GenResBlocks + SelfAttention (always)
                 → global pool → HNormHead for (a, b) scalars
  Decoder:       Mirror of encoder with UNet skip connections
  DAB head:      GroupNorm → SiLU → Conv(C → 1) → ReLU (density ≥ 0)
  CombinationNet (DAB 1ch + H&E 3ch) → IHC RGB 3ch in [-1, 1]

No timestep conditioning — the GAN generator is purely conditional on H&E.
Reuses CombinationNet and HNormHead from dab_pixelgen.model.
"""
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

PIXELGEN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PIXELGEN_ROOT not in sys.path:
    sys.path.insert(0, PIXELGEN_ROOT)

from dab_pixelgen.model import CombinationNet, HNormHead


# ── Helpers ───────────────────────────────────────────────────────────────────

def _gn(channels: int) -> int:
    """Largest divisor of `channels` that is ≤ 32 (for GroupNorm)."""
    for g in (32, 16, 8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


# ── Building blocks ───────────────────────────────────────────────────────────

class GenResBlock(nn.Module):
    """
    Residual block with GroupNorm + SiLU (no timestep conditioning).
    Zero-initialises the second conv for a stable residual-identity start.
    """

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(_gn(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(_gn(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.act   = nn.SiLU()
        self.drop  = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.skip  = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.act(self.norm2(h))
        h = self.drop(h)
        h = self.conv2(h)
        return h + self.skip(x)


class SelfAttn(nn.Module):
    """
    Multi-head self-attention over 2-D spatial feature maps.
    GroupNorm pre-norm; uses scaled_dot_product_attention for efficiency.
    """

    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        # Ensure divisibility
        while channels % num_heads != 0 and num_heads > 1:
            num_heads //= 2
        self.num_heads = num_heads
        self.head_dim  = channels // num_heads
        self.norm = nn.GroupNorm(_gn(channels), channels)
        self.qkv  = nn.Linear(channels, 3 * channels, bias=True)
        self.proj = nn.Linear(channels, channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h   = self.norm(x).reshape(B, C, -1).transpose(1, 2)      # (B, HW, C)
        qkv = (
            self.qkv(h)
            .reshape(B, -1, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)                                # (3, B, heads, HW, d)
        )
        q, k, v = qkv.unbind(0)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, -1, C)
        out = self.proj(out).transpose(1, 2).reshape(B, C, H, W)
        return x + out


# ── Generator ─────────────────────────────────────────────────────────────────

class DABUNetGenerator(nn.Module):
    """
    Conditional UNet generator for H&E → IHC virtual staining.

    forward(he) → (dab_pred, ihc_pred, h_norm_params)

      he:            [B, 3, H, W]   H&E in [-1, 1]
      ──────────────────────────────────────────────
      dab_pred:      [B, 1, H, W]   raw DAB density (≥ 0)
      ihc_pred:      [B, 3, H, W]   IHC RGB in [-1, 1]
      h_norm_params: [B, 2]          (a_raw, b_raw) for analytical fallback

    Parameters
    ----------
    base_channels          : int   = 64    base channel width
    channel_mult           : tuple = (1,2,4,8)  channel multiplier per level
    num_res_blocks         : int   = 2     ResBlocks per encoder/decoder level
    attention_resolutions  : tuple = (32,) spatial resolutions with SelfAttn
    input_size             : int   = 256
    dropout                : float = 0.0
    combination_channels   : int   = 32   CombinationNet base channels
    """

    def __init__(
        self,
        base_channels:          int             = 64,
        channel_mult:           Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks:         int             = 2,
        attention_resolutions:  Tuple[int, ...] = (32,),
        input_size:             int             = 256,
        dropout:                float           = 0.0,
        combination_channels:   int             = 32,
    ):
        super().__init__()

        # Store for use in forward()
        self._channel_mult    = tuple(channel_mult)
        self._num_res_blocks  = num_res_blocks

        C      = base_channels
        n_levs = len(channel_mult)
        attn   = set(attention_resolutions)

        # ── Stem ──────────────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, 3, padding=1),
            nn.GroupNorm(_gn(C), C),
            nn.SiLU(),
        )

        # ── Encoder ───────────────────────────────────────────────────────────
        enc_blocks  = []
        enc_attns   = []
        downsamples = []
        skip_chans  = [C]   # stem output

        ch  = C
        res = input_size

        for i, mult in enumerate(channel_mult):
            out_ch = C * mult
            for _ in range(num_res_blocks):
                enc_blocks.append(GenResBlock(ch, out_ch, dropout))
                enc_attns.append(SelfAttn(out_ch) if res in attn else nn.Identity())
                ch = out_ch
                skip_chans.append(ch)
            # Strided downsample between levels (skip after last level)
            if i < n_levs - 1:
                downsamples.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                skip_chans.append(ch)
                res //= 2

        self.enc_blocks  = nn.ModuleList(enc_blocks)
        self.enc_attns   = nn.ModuleList(enc_attns)
        self.downsamples = nn.ModuleList(downsamples)
        self._skip_chans = list(skip_chans)   # reference copy for decoder build

        # ── Bottleneck ────────────────────────────────────────────────────────
        self.bot_res1 = GenResBlock(ch, ch)
        self.bot_attn = SelfAttn(ch)          # unconditional attn at bottleneck
        self.bot_res2 = GenResBlock(ch, ch)
        self._bottleneck_ch = ch

        # HNormHead: global-pooled bottleneck → (a_raw, b_raw)
        self.h_norm_head = HNormHead(ch)

        # ── Decoder ───────────────────────────────────────────────────────────
        dec_blocks = []
        dec_attns  = []
        upsamples  = []

        rem_skips = list(skip_chans)    # pop from end for reverse order
        dec_res   = res                 # resolution at bottleneck

        for i in reversed(range(n_levs)):
            out_ch = C * channel_mult[i]
            for _ in range(num_res_blocks + 1):   # extra block to consume downsample skip
                skip_ch = rem_skips.pop()
                dec_blocks.append(GenResBlock(ch + skip_ch, out_ch, dropout))
                dec_attns.append(SelfAttn(out_ch) if dec_res in attn else nn.Identity())
                ch = out_ch
            if i > 0:
                upsamples.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(ch, ch, 3, padding=1),
                ))
                dec_res *= 2

        assert len(rem_skips) == 0, (
            f"DABUNetGenerator: skip channel mismatch, {len(rem_skips)} remaining"
        )

        self.dec_blocks = nn.ModuleList(dec_blocks)
        self.dec_attns  = nn.ModuleList(dec_attns)
        self.upsamples  = nn.ModuleList(upsamples)

        # ── DAB head ──────────────────────────────────────────────────────────
        self.dab_head = nn.Sequential(
            nn.GroupNorm(_gn(ch), ch),
            nn.SiLU(),
            nn.Conv2d(ch, 1, 1),
        )
        nn.init.zeros_(self.dab_head[-1].weight)
        nn.init.zeros_(self.dab_head[-1].bias)

        # ── CombinationNet ────────────────────────────────────────────────────
        self.combination_net = CombinationNet(base_channels=combination_channels)

        # ── Summary ───────────────────────────────────────────────────────────
        total   = sum(p.numel() for p in self.parameters())
        combo_p = sum(p.numel() for p in self.combination_net.parameters())
        print(
            f"DABUNetGenerator: {total/1e6:.1f}M params  "
            f"(UNet {(total-combo_p)/1e6:.1f}M + CombinationNet {combo_p/1e6:.2f}M)\n"
            f"  base={base_channels}  mult={channel_mult}  "
            f"nrb={num_res_blocks}  attn@{sorted(attn)}"
        )

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(
        self, he: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            he: [B, 3, H, W] H&E image in [-1, 1]
        Returns:
            dab_pred:      [B, 1, H, W]  predicted DAB density (≥ 0)
            ihc_pred:      [B, 3, H, W]  IHC RGB in [-1, 1]
            h_norm_params: [B, 2]         (a_raw, b_raw) for analytical recomposition
        """
        channel_mult   = self._channel_mult
        num_res_blocks = self._num_res_blocks

        # ── Encode ────────────────────────────────────────────────────────────
        x     = self.stem(he)
        skips = [x]

        enc_idx = down_idx = 0
        for i, _ in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                x = self.enc_blocks[enc_idx](x)
                x = self.enc_attns[enc_idx](x)
                skips.append(x)
                enc_idx += 1
            if i < len(channel_mult) - 1:
                x = self.downsamples[down_idx](x)
                skips.append(x)
                down_idx += 1

        # ── Bottleneck ────────────────────────────────────────────────────────
        x = self.bot_res1(x)
        x = self.bot_attn(x)
        x = self.bot_res2(x)

        # H-norm params from global-pooled bottleneck: [B, C] → [B, 1, C] → [B, 2]
        h_norm_params = self.h_norm_head(x.mean(dim=(2, 3)).unsqueeze(1))

        # ── Decode ────────────────────────────────────────────────────────────
        dec_idx = up_idx = 0
        for i in reversed(range(len(channel_mult))):
            for _ in range(num_res_blocks + 1):
                skip = skips.pop()
                x = torch.cat([x, skip], dim=1)
                x = self.dec_blocks[dec_idx](x)
                x = self.dec_attns[dec_idx](x)
                dec_idx += 1
            if i > 0:
                x = self.upsamples[up_idx](x)
                up_idx += 1

        # ── Outputs ───────────────────────────────────────────────────────────
        dab_pred = F.relu(self.dab_head(x))           # [B, 1, H, W]  density ≥ 0
        ihc_pred = self.combination_net(dab_pred, he)  # [B, 3, H, W]  in [-1, 1]

        return dab_pred, ihc_pred, h_norm_params


# ── EMA ───────────────────────────────────────────────────────────────────────

class GeneratorEMA:
    """
    CPU-offloaded Exponential Moving Average for the generator.
    Mirrors the pattern from dab_pixelgen.model.SimpleEMA.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay  = decay
        self.shadow = {
            name: param.clone().detach().cpu()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self._backup: dict = {}
        print(f"GeneratorEMA: decay={decay}, tracking {len(self.shadow)} params")

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data.cpu(), alpha=1 - self.decay
                )

    def apply_shadow(self, model: nn.Module):
        self._backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self._backup[name] = param.data.clone().cpu()
                param.data.copy_(self.shadow[name].to(param.device))

    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if name in self._backup:
                param.data.copy_(self._backup[name].to(param.device))
        self._backup = {}

    def state_dict(self):
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, sd: dict):
        self.shadow = sd["shadow"]
        self.decay  = sd.get("decay", self.decay)


# ── Factory ───────────────────────────────────────────────────────────────────

def create_generator(config, device: str = "cuda") -> DABUNetGenerator:
    """Instantiate DABUNetGenerator from GANDABConfig and move to device."""
    import gc
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return DABUNetGenerator(
        base_channels         = config.gen_base_channels,
        channel_mult          = tuple(config.gen_channel_mult),
        num_res_blocks        = config.gen_num_res_blocks,
        attention_resolutions = tuple(config.gen_attention_resolutions),
        input_size            = config.image_size,
        dropout               = config.gen_dropout,
        combination_channels  = config.combination_channels,
    ).to(device)
