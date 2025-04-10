"""Step 1: Ingest & Preprocess module for HDR image grouping."""

from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import cv2
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from utils import (
    is_valid_image_file,
    generate_unique_id,
    normalize_image_size,
    enhance_local_contrast,
    logger,
    ImageData,
    PathLike
)


class ImageIngestor:
    """Handles image ingestion, validation, and preprocessing."""

    def __init__(
            self,
            target_size: Tuple[int, int] = (512, 512),
            max_workers: int = 4
    ):
        """
        Initialize the ingestor.

        Args:
            target_size: Target image size for normalization
            max_workers: Maximum number of worker threads for parallel processing
        """
        self.target_size = target_size
        self.max_workers = max_workers
        self.logger = logging.getLogger('hdr_grouping.ingest')

    def load_image(self, file_path: Path) -> Optional[ImageData]:
        """
        Load a single image file and preprocess it.

        Args:
            file_path: Path to the image file

        Returns:
            Dictionary with image data or None if loading fails
        """
        try:
            # Use OpenCV for better HDR handling
            img = cv2.imread(str(file_path))
            if img is None:
                self.logger.warning(f"Failed to load image: {file_path}")
                return None

            # Generate a unique ID
            image_id = generate_unique_id()

            # Store original image dimensions
            original_height, original_width = img.shape[:2]

            # Normalize image size
            normalized = normalize_image_size(img, self.target_size)

            # Convert to grayscale
            grayscale = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)

            # Enhance local contrast to reveal details in all exposure ranges
            enhanced = enhance_local_contrast(grayscale)

            # Package image data
            return {
                'id': image_id,
                'file_path': str(file_path),
                'original_image': img,
                'normalized_image': normalized,
                'grayscale_image': grayscale,
                'enhanced_image': enhanced,
                'metadata': {
                    'original_width': original_width,
                    'original_height': original_height,
                    'filename': file_path.name
                }
            }
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {str(e)}")
            return None

    def ingest_directory(self, directory_path: PathLike) -> List[ImageData]:
        """
        Scan a directory and ingest all valid image files.

        Args:
            directory_path: Path to directory containing images

        Returns:
            List of processed image data dictionaries
        """
        dir_path = Path(directory_path)
        if not dir_path.exists() or not dir_path.is_dir():
            self.logger.error(f"Directory does not exist or is not a directory: {directory_path}")
            return []

        # Get all files in directory and subdirectories
        all_files = list(dir_path.glob('**/*'))

        # Filter for valid image files
        image_files = [f for f in all_files if is_valid_image_file(f)]

        if not image_files:
            self.logger.warning(f"No valid image files found in {directory_path}")
            return []

        self.logger.info(f"Found {len(image_files)} valid image files in {directory_path}")

        # Process images in parallel
        processed_images = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.load_image, f) for f in image_files]

            # Use tqdm for progress tracking
            for future in tqdm(futures, desc="Ingesting images", total=len(futures)):
                result = future.result()
                if result is not None:
                    processed_images.append(result)

        self.logger.info(f"Successfully processed {len(processed_images)} images")
        return processed_images