import numpy as np

from .common import CommonInpainter
from ..config import InpainterConfig


class OriginalInpainter(CommonInpainter):

    async def _inpaint(self, image: np.ndarray, mask: np.ndarray, config: InpainterConfig, inpainting_size: int = 1024, verbose: bool = False, time_stats=None) -> np.ndarray:
        return np.copy(image)
