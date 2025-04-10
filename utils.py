"""Utility functions for HDR image grouping pipeline."""

import os
from pathlib import Path
import logging
import uuid
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError, ImageFile

# Allow truncated images to be loaded
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('hdr_grouping')

# Type definitions
ImageData = Dict[str, Union[str, np.ndarray, Dict]]
PathLike = Union[str, Path]


def setup_output_dir(base_dir: PathLike) -> Path:
    """Create and return output directory with timestamp."""
    base_path = Path(base_dir)
    output_dir = base_path / "grouped_output"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def is_valid_image_file(file_path: Path) -> bool:
    """Check if a file is a valid image file (JPEG/PNG)."""
    if not file_path.is_file():
        return False

    valid_extensions = {'.jpg', '.jpeg', '.png'}
    if file_path.suffix.lower() not in valid_extensions:
        return False

    try:
        with Image.open(file_path) as img:
            # Attempt to access image properties to verify it's valid
            img.verify()
        return True
    except (UnidentifiedImageError, OSError, Exception):
        logger.warning(f"Invalid image file: {file_path}")
        return False


def generate_unique_id() -> str:
    """Generate a unique ID for an image."""
    return str(uuid.uuid4())


def estimate_exposure_value(image: np.ndarray) -> float:
    """
    Estimate relative exposure value from image luminance.

    Args:
        image: Grayscale or color image as numpy array

    Returns:
        Relative EV (exposure value) - higher means brighter
    """
    # Convert to grayscale if image is color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Get median brightness as a more robust measure than mean
    # (less affected by extreme highlights/shadows)
    median_brightness = np.median(gray)

    # Calculate log2 of the median for a more perceptual scale
    # Add small epsilon to avoid log(0)
    ev = np.log2(median_brightness + 1e-5)

    return ev


def normalize_image_size(image: np.ndarray, target_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """
    Resize image to target dimensions while preserving aspect ratio.

    Args:
        image: Input image as numpy array
        target_size: Target (width, height) tuple

    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size

    # Calculate aspect ratios
    aspect = w / h
    target_aspect = target_w / target_h

    # Resize preserving aspect ratio
    if aspect > target_aspect:
        # Image is wider than target
        new_w = target_w
        new_h = int(target_w / aspect)
    else:
        # Image is taller than target
        new_h = target_h
        new_w = int(target_h * aspect)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create a black canvas of target size
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # Calculate padding
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2

    # Place the resized image on the canvas
    if len(resized.shape) == 3:
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    else:
        # If grayscale, expand to 3 channels
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = np.expand_dims(resized, axis=2).repeat(3, axis=2)

    return canvas


def enhance_local_contrast(image: np.ndarray) -> np.ndarray:
    """
    Enhance local contrast using CLAHE to reveal details in extreme exposures.

    Args:
        image: Input grayscale image

    Returns:
        Enhanced grayscale image
    """
    # Ensure image is grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    return enhanced