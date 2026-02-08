# --------------------------------------------------------
# JiT_I2I: Image-to-Image variant of Just Image Transformer
# Based on JiT.py - adapted for paired image translation
# Condition image is concatenated channel-wise with noisy target
# No class/text embedding - only timestep conditioning
# Supports loading pretrained JiT weights (skips y_embedder, in_context)
# --------------------------------------------------------
import torch
import math
import torch.nn.functional as F
from math import pi
import logging

from torch import nn
import numpy as np
from einops import rearrange, repeat

logger = logging.getLogger(__name__)


def broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d r -> ... (d r)')


class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len=16,
        ft_seq_len=None,
        custom_freqs=None,
        freqs_for='lang',
        theta=10000,
        max_freq=10,
        num_freqs=1,
        num_cls_token=0,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs = torch.einsum('..., f -> ... f', t, freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r=2)
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim=-1)

        if num_cls_token > 0:
            freqs_flat = freqs.view(-1, freqs.shape[-1])
            cos_img = freqs_flat.cos()
            sin_img = freqs_flat.sin()
            N_img, D = cos_img.shape
            cos_pad = torch.ones(num_cls_token, D, dtype=cos_img.dtype, device=cos_img.device)
            sin_pad = torch.zeros(num_cls_token, D, dtype=sin_img.dtype, device=sin_img.device)
            self.freqs_cos = torch.cat([cos_pad, cos_img], dim=0).cuda()
            self.freqs_sin = torch.cat([sin_pad, sin_img], dim=0).cuda()
        else:
            self.freqs_cos = freqs.cos().view(-1, freqs.shape[-1]).cuda()
            self.freqs_sin = freqs.sin().view(-1, freqs.shape[-1]).cuda()

    def forward(self, t):
        if self.freqs_cos.device != t.device:
            self.freqs_cos = self.freqs_cos.to(t.device)
            self.freqs_sin = self.freqs_sin.to(t.device)
        return t * self.freqs_cos + rotate_half(t) * self.freqs_sin


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class PatchEmbed(nn.Module):
    """Image to Patch Embedding (non-bottleneck, matches original JiT XL)."""
    def __init__(self, img_size=256, patch_size=16, in_chans=6, pca_dim=768, embed_dim=768, bias=True):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj1 = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj1(x).flatten(2).transpose(1, 2)
        return x


class BottleneckPatchEmbed(nn.Module):
    """Image to Patch Embedding with bottleneck."""
    def __init__(self, img_size=256, patch_size=16, in_chans=6, pca_dim=128, embed_dim=768, bias=True):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj1 = nn.Conv2d(in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)
        return x


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class SwiGLUFFN(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.0, bias=True):
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


from torch.nn.functional import scaled_dot_product_attention

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = rope(q)
        k = rope(k)
        x = scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FinalLayer(nn.Module):
    """The final layer of JiT_I2I."""
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class JiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    @torch.compile
    def forward(self, x, c, feat_rope=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class JiT_I2I(nn.Module):
    """
    Just Image Transformer for Image-to-Image translation.

    Condition image (e.g., H&E) is concatenated channel-wise with the noisy
    target image (e.g., IHC) inside the forward method.

    No class/text embedding - only timestep conditioning via adaLN.
    Supports loading pretrained JiT weights (y_embedder & in_context skipped,
    x_embedder conv expanded from 3ch to 6ch with zero-init for condition half).
    """
    def __init__(
        self,
        input_size=256,
        patch_size=16,
        in_channels=3,
        cond_channels=3,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        bottleneck_dim=128,
        use_bottleneck=False,
        use_compile=False,
        weight_path=None,
        load_ema=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.cond_channels = cond_channels
        self.out_channels = in_channels  # Output only the target channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.use_bottleneck = use_bottleneck
        self.use_compile = use_compile
        self.weight_path = weight_path
        self.load_ema = load_ema

        # Timestep embedding only (no class embedding)
        self.t_embedder = TimestepEmbedder(hidden_size)

        # Patch embedding takes concatenated input: in_channels + cond_channels
        total_in_channels = in_channels + cond_channels
        if self.use_bottleneck:
            self.x_embedder = BottleneckPatchEmbed(
                input_size, patch_size, total_in_channels, bottleneck_dim, hidden_size, bias=True
            )
        else:
            self.x_embedder = PatchEmbed(
                input_size, patch_size, total_in_channels, bottleneck_dim, hidden_size, bias=True
            )

        # Fixed sin-cos positional embedding
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # RoPE (no in-context tokens, so only one rope)
        half_head_dim = hidden_size // num_heads // 2
        hw_seq_len = input_size // patch_size
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=0,
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            JiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                     attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                     proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0)
            for i in range(depth)
        ])

        # Final prediction layer (outputs only target channels)
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

        # Load pretrained weights after initialization
        if self.weight_path is not None:
            self._load_pretrained(self.weight_path, self.load_ema)

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize pos_embed with sin-cos
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed
        if self.use_bottleneck:
            w1 = self.x_embedder.proj1.weight.data
            nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
            w2 = self.x_embedder.proj2.weight.data
            nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
            nn.init.constant_(self.x_embedder.proj2.bias, 0)
        else:
            w1 = self.x_embedder.proj1.weight.data
            nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
            nn.init.constant_(self.x_embedder.proj1.bias, 0)

        # Initialize timestep embedding
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def _load_pretrained(self, weight_path, load_ema=True):
        """
        Load pretrained JiT weights, adapting x_embedder for 6-channel input.
        Skips y_embedder, in_context_posemb, feat_rope_incontext.
        For x_embedder.proj1: copies 3ch pretrained weights to first 3 channels,
        zero-initializes the condition (last 3) channels.
        """
        logger.info(f"Loading pretrained JiT weights from {weight_path}")
        ckpt = torch.load(weight_path, map_location='cpu')

        # Handle lightning checkpoint format
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt

        # Determine prefix (ema_denoiser or denoiser)
        prefix = "ema_denoiser." if load_ema else "denoiser."

        # Strip prefix to get bare model keys
        pretrained = {}
        for k, v in state_dict.items():
            if k.startswith(prefix):
                bare_key = k[len(prefix):]
                pretrained[bare_key] = v

        if not pretrained:
            logger.warning(f"No weights found with prefix '{prefix}'. Trying without prefix.")
            pretrained = state_dict

        # Keys to skip (not present in I2I model)
        skip_prefixes = ('y_embedder.', 'in_context_posemb', 'feat_rope_incontext.')

        loaded, skipped = 0, 0
        my_state = self.state_dict()

        for key, param in pretrained.items():
            # Skip class-conditioning and in-context related weights
            if any(key.startswith(sp) for sp in skip_prefixes):
                logger.info(f"  Skipping: {key}")
                skipped += 1
                continue

            # Handle x_embedder.proj1 channel expansion (3ch -> 6ch)
            if key == 'x_embedder.proj1.weight' and key in my_state:
                my_shape = my_state[key].shape  # (embed_dim, 6, P, P)
                pt_shape = param.shape           # (embed_dim, 3, P, P)
                if my_shape[1] != pt_shape[1]:
                    logger.info(f"  Expanding x_embedder.proj1.weight: {pt_shape} -> {my_shape}")
                    new_weight = torch.zeros_like(my_state[key])
                    new_weight[:, :pt_shape[1], :, :] = param  # Copy 3ch to first 3
                    # Last 3 channels (condition) stay zero-initialized
                    my_state[key].copy_(new_weight)
                    loaded += 1
                    continue

            # Load matching keys
            if key in my_state:
                if my_state[key].shape == param.shape:
                    my_state[key].copy_(param)
                    loaded += 1
                else:
                    logger.warning(f"  Shape mismatch for {key}: "
                                   f"pretrained {param.shape} vs model {my_state[key].shape}. Skipping.")
                    skipped += 1
            else:
                logger.info(f"  Key not in I2I model: {key}")
                skipped += 1

        logger.info(f"Pretrained weight loading: {loaded} loaded, {skipped} skipped")

    def compile(self):
        """Optionally compile transformer blocks."""
        if self.use_compile:
            for block in self.blocks:
                block.forward = torch.compile(block.forward)

    def unpatchify(self, x, p):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        c = self.out_channels
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y, return_layer=None, return_last=False):
        """
        Forward pass for image-to-image translation.

        Args:
            x: (N, C, H, W) - noisy target image
            t: (N,) - timesteps
            y: (N, C, H, W) - condition image (e.g., H&E)
            return_layer: int or None - return intermediate features at this layer
            return_last: bool - also return last layer features
        """
        # Concatenate condition image with noisy target
        x_concat = torch.cat([x, y], dim=1)  # (N, in_channels + cond_channels, H, W)

        # Timestep embedding only
        t_emb = self.t_embedder(t)
        c = t_emb  # No class embedding

        # Patch embedding
        x = self.x_embedder(x_concat)
        x += self.pos_embed

        # Transformer blocks
        for i, block in enumerate(self.blocks):
            if return_layer is not None and i == return_layer:
                feat = x
            x = block(x, c, self.feat_rope)

        if return_last:
            last_out = x

        # Final layer
        x = self.final_layer(x, c)
        output = self.unpatchify(x, self.patch_size)

        if return_layer is not None:
            if return_last:
                return output, feat, last_out
            else:
                return output, feat
        else:
            return output


def JiT_I2I_XL(**kwargs):
    """XL I2I model (~671M params, matching PixelGen XL)"""
    return JiT_I2I(
        depth=28, hidden_size=1152, num_heads=16,
        use_bottleneck=False, patch_size=16, **kwargs
    )


def JiT_I2I_S(**kwargs):
    """Small I2I model (~195M params)"""
    return JiT_I2I(
        depth=18, hidden_size=768, num_heads=12,
        bottleneck_dim=128, use_bottleneck=True, patch_size=16, **kwargs
    )
