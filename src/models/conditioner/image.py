import torch
from src.models.conditioner.base import BaseConditioner


class ImageConditioner(BaseConditioner):
    """
    Conditioner for image-to-image tasks.
    Extracts the condition image from metadata (training) or from stacked tensors.
    Returns zero tensor as the null/unconditional signal.
    """
    def __init__(self):
        super().__init__()

    def _impl_condition(self, y, metadata):
        # metadata['condition_image'] is already stacked by the collate function
        condition = metadata['condition_image']
        if not isinstance(condition, torch.Tensor):
            condition = torch.stack(condition)
        return condition.cuda()

    def _impl_uncondition(self, y, metadata):
        condition = self._impl_condition(y, metadata)
        return torch.zeros_like(condition)

    @torch.no_grad()
    def __call__(self, y, metadata: dict = {}):
        condition = self._impl_condition(y, metadata)
        uncondition = self._impl_uncondition(y, metadata)
        if condition.dtype in [torch.float64, torch.float32, torch.float16]:
            condition = condition.to(torch.bfloat16)
        if uncondition.dtype in [torch.float64, torch.float32, torch.float16]:
            uncondition = uncondition.to(torch.bfloat16)
        return condition, uncondition

