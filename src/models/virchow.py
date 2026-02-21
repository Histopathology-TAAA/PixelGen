"""
Virchow2 encoder for extracting pathology embeddings from H&E images.

Virchow2 is a ViT-H/14 vision transformer pretrained on 3.1M histopathology slides.
Output embedding dimension: 2560 (1280 class token + 1280 mean patch token)
or 256x1280 patch tokens for spatial conditioning.

Reference: https://huggingface.co/paige-ai/Virchow2
"""

import torch
import torch.nn as nn
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked


class Virchow2Encoder(nn.Module):
    """
    Wrapper for Virchow2 model to extract embeddings from H&E images.
    
    Args:
        use_patch_tokens: If True, return patch tokens (B, 256, 1280) for spatial conditioning.
                         If False, return concatenated class + mean patch (B, 2560).
        freeze: If True, freeze the encoder weights (recommended for conditioning).
    """
    
    def __init__(self, use_patch_tokens: bool = True, freeze: bool = True):
        super().__init__()
        self.use_patch_tokens = use_patch_tokens
        self.freeze = freeze
        
        # Load Virchow2 model with proper configuration
        self.model = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU
        )
        self.model.eval()
        
        # Get the preprocessing transform from model config
        self.transform = create_transform(
            **resolve_data_config(self.model.pretrained_cfg, model=self.model)
        )
        
        # Freeze if specified
        if self.freeze:
            for param in self.model.parameters():
                param.requires_grad = False
    
    @property
    def embed_dim(self) -> int:
        """Return the embedding dimension."""
        if self.use_patch_tokens:
            return 1280  # Per-patch dimension
        else:
            return 2560  # Class token + mean patch token
    
    @property
    def num_tokens(self) -> int:
        """Return the number of tokens (sequence length)."""
        if self.use_patch_tokens:
            return 256  # 16x16 patches from 224x224 image with patch_size=14
        else:
            return 1  # Single embedding
    
    def forward(self, x: torch.Tensor, resize: bool = True) -> torch.Tensor:
        """
        Extract embeddings from images.
        
        Args:
            x: Input images of shape (B, C, H, W) in range [0, 1].
               Will be resized to 224x224 and normalized internally.
            resize: Whether to resize input to 224x224 (default: True).
        
        Returns:
            If use_patch_tokens=True: (B, 256, 1280) patch token embeddings
            If use_patch_tokens=False: (B, 2560) concatenated class + mean patch embedding
        """
        # Resize to Virchow2 expected size (224x224)
        if resize and x.shape[-2:] != (224, 224):
            x = torch.nn.functional.interpolate(
                x, size=(224, 224), mode='bilinear', align_corners=False
            )
        
        # Normalize using ImageNet stats (what timm transform does)
        # Mean and std from ImageNet
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        
        # Run inference with autocast if CUDA is available
        if x.device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                output = self.model(x)  # (B, 261, 1280) = 1 class + 4 register + 256 patch
        else:
            output = self.model(x)  # CPU fallback
        
        class_token = output[:, 0]      # (B, 1280)
        patch_tokens = output[:, 5:]    # (B, 256, 1280) - skip register tokens 1-4
        
        if self.use_patch_tokens:
            return patch_tokens  # (B, 256, 1280)
        else:
            # Concatenate class token and mean pooled patches
            mean_patch = patch_tokens.mean(dim=1)  # (B, 1280)
            embedding = torch.cat([class_token, mean_patch], dim=-1)  # (B, 2560)
            return embedding
    
    def get_intermediate_feats(self, x: torch.Tensor, resize: bool = True, n: list = None, 
                               reshape: bool = False, return_class_token: bool = False) -> list:
        """
        Get intermediate features compatible with DINO interface.
        
        Args:
            x: Input images of shape (B, C, H, W) in range [0, 1].
            resize: Whether to resize input (default: True).
            n: List of layer indices (ignored for Virchow2, kept for compatibility).
            reshape: Whether to reshape features (ignored for Virchow2).
            return_class_token: Whether to return class token (ignored for Virchow2).
        
        Returns:
            List containing patch token features (B, 256, 1280) for compatibility with DINO.
            Since Virchow2 doesn't have multiple layers, returns a single-element list.
        """
        # Resize to Virchow2 expected size (224x224)
        if resize and x.shape[-2:] != (224, 224):
            x = torch.nn.functional.interpolate(
                x, size=(224, 224), mode='bilinear', align_corners=False
            )
        
        # Normalize using ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        
        # Run inference
        if x.device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                output = self.model(x)  # (B, 261, 1280)
        else:
            output = self.model(x)  # CPU fallback
        
        # Extract patch tokens (skip class token and register tokens)
        patch_tokens = output[:, 5:]  # (B, 256, 1280)
        
        # Return as list for compatibility with DINO interface
        # DINO returns list of features from different layers, we return single element
        return [patch_tokens]

