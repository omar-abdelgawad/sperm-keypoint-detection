"""contains the preprocess_image function. Sharpens input image."""

import numpy as np
import cv2

BLUR_KERNEL = (5, 5)
SIGMA_X = 1


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """apply all filters and preprocessing functions for better quality."""
    # blurred = cv2.GaussianBlur(img, BLUR_KERNEL, SIGMA_X)
    # alpha = 2.5  # Weight for the original image
    # beta = -1.5  # Weight for the sharpened image
    # img = cv2.addWeighted(img, alpha, blurred, beta, 0)
    return img
