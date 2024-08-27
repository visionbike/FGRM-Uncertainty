import numpy as np
import cv2

__all__ = ["resize_image"]


def resize_image(x: np.ndarray, img_size: int = 256) -> np.ndarray:
    return cv2.resize(x, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
