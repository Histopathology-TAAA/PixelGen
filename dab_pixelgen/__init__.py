"""
DABPixelGen: H&E -> IHC virtual staining via DAB density prediction.

Single PixelGen JiT_I2I backbone that:
  1. Predicts 1-channel DAB optical density from noisy DAB + H&E condition
  2. Predicts per-image H-channel normalization parameters (a, b)
  3. Analytically recomposes IHC RGB via Beer-Lambert law

Quick start:
    from dab_pixelgen.run import main
    main(pretrained_weight_path="./PixelGen_XL_80ep.ckpt")
"""
from dab_pixelgen.config    import DABPixelGenConfig
from dab_pixelgen.model     import DABPixelGenModel, create_dab_model
from dab_pixelgen.scheduler import DABFlowScheduler
from dab_pixelgen.stain_utils import (
    StainDeconvolution, StainRecomposer, PSPStainDABExtractor,
    analytical_recompose, normalize_h_density,
)

__all__ = [
    "DABPixelGenConfig",
    "DABPixelGenModel",
    "create_dab_model",
    "DABFlowScheduler",
    "StainDeconvolution",
    "StainRecomposer",
    "PSPStainDABExtractor",
    "analytical_recompose",
    "normalize_h_density",
]
