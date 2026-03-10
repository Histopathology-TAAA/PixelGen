# PixelGen Virtual Staining Project

You are assisting with the PixelGen I2I virtual staining project. Key context:
- Model: JiT_I2I XL, flow-matching diffusion for H&E → IHC staining
- Data: MIST ER/Ki67 paired 256×256 histology images
- Current focus: Comparing noise-flow vs source-flow paradigms

For detailed context, see CONTEXT.md in the root of this workspace.

Key facts:
- Source flow (H&E→IHC) has better SSIM but worse FID than noise flow
- Model receives concat(x_t, H&E condition) in JiT_I2I denoiser
- Training uses REPA + LPIPS + optional DINO perceptual losses
- Inference: 50-step Euler ODE sampling from t=0 to t=1

When discussing experiments, refer to configs in configs_i2i/ and workdirs in mist_er_workdirs/.