"""
GAN DABPixelGen — Conditional GAN for H&E → IHC virtual staining.

Replaces the flow-matching diffusion backbone with a conditional GAN:
  - Generator:     DABUNetGenerator  (UNet + DAB head + CombinationNet)
  - Discriminator: MultiScalePatchGAN (2-scale PatchGAN with spectral norm)
  - Training:      TTUR, R1 penalty, EMA, feature matching + L1 + DAB + LPIPS

Quick start:
    from gan_dab_pixelgen.run import main
    main()
"""
from .config        import GANDABConfig
from .generator     import DABUNetGenerator, GeneratorEMA, create_generator
from .discriminator import MultiScalePatchGAN, NLayerDiscriminator
from .losses        import GANLoss, FeatureMatchingLoss, GANDABLoss, DABExpressionLoss
from .inference     import run_inference, PracticalMetrics

__all__ = [
    "GANDABConfig",
    "DABUNetGenerator",
    "GeneratorEMA",
    "create_generator",
    "MultiScalePatchGAN",
    "NLayerDiscriminator",
    "GANLoss",
    "FeatureMatchingLoss",
    "GANDABLoss",
    "DABExpressionLoss",
    "run_inference",
    "PracticalMetrics",
]
