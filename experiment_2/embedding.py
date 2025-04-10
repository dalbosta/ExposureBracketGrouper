"""Step 2: Robust Scene Embedding module for HDR image grouping."""

import numpy as np
import torch
import cv2
from typing import List, Dict, Tuple, Optional, Union, Any
from pathlib import Path
import pickle
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import logging
from sklearn.decomposition import PCA
import umap

# Import transformers only if available
try:
    from transformers import AutoFeatureExtractor, AutoModel

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from utils import ImageData, logger, PathLike


class SceneEmbedder:
    """Creates exposure-agnostic image fingerprints using deep and traditional features."""

    def __init__(
            self,
            model_name: str = "facebook/dinov2-small",
            use_gpu: bool = torch.cuda.is_available(),
            use_traditional_features: bool = True,
            dimensionality_reduction: str = "pca",  # 'pca', 'umap', or None
            target_dims: int = 128,
            cache_dir: Optional[PathLike] = None,
            max_workers: int = 2  # Fewer workers for GPU memory considerations
    ):
        """
        Initialize the scene embedder.

        Args:
            model_name: Name of the pretrained model to use
            use_gpu: Whether to use GPU if available
            use_traditional_features: Whether to also extract traditional CV features
            dimensionality_reduction: Method for dimensionality reduction
            target_dims: Target embedding dimensions after reduction
            cache_dir: Directory to cache embeddings
            max_workers: Maximum number of worker threads
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.use_traditional_features = use_traditional_features
        self.dimensionality_reduction = dimensionality_reduction
        self.target_dims = target_dims
        self.max_workers = max_workers
        self.logger = logging.getLogger('hdr_grouping.embedding')

        # Initialize cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
        else:
            self.cache_dir = None

        # Initialize dimensionality reduction model
        self.dim_reducer = None

        # Initialize deep model if transformers is available
        if TRANSFORMERS_AVAILABLE:
            self._init_deep_model()
        else:
            self.logger.warning("Transformers library not available, falling back to traditional features only")
            self.extractor = None
            self.model = None
            self.use_traditional_features = True  # Force traditional features if deep model not available

        # Initialize traditional feature extractor if needed
        if self.use_traditional_features:
            self._init_traditional_extractor()

    def _init_deep_model(self):
        """Initialize the deep learning model for feature extraction."""
        try:
            self.logger.info(f"Loading model: {self.model_name}")
            self.extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()  # Set to evaluation mode
            self.logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            self.logger.error(f"Error loading deep model: {str(e)}")
            self.extractor = None
            self.model = None
            self.use_traditional_features = True  # Fall back to traditional features

    def _init_traditional_extractor(self):
        """Initialize the traditional CV feature extractors."""
        # Initialize SIFT for keypoint detection and feature extraction
        self.sift = cv2.SIFT_create()

        # Initialize ORB as backup/complement
        self.orb = cv2.ORB_create()

    def _extract_deep_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract deep features from an image using the loaded model.

        Args:
            image: Preprocessed image as numpy array

        Returns:
            Feature vector as numpy array
        """
        if self.extractor is None or self.model is None:
            return np.array([])

        try:
            # Prepare the image for the model
            inputs = self.extractor(images=image, return_tensors="pt").to(self.device)

            # Extract features
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Get embeddings from the [CLS] token from the last hidden state
            features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            return features.flatten()
        except Exception as e:
            self.logger.error(f"Error extracting deep features: {str(e)}")
            return np.array([])

    def _extract_traditional_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract traditional computer vision features from an image.

        Args:
            image: Enhanced grayscale image as numpy array

        Returns:
            Feature vector as numpy array
        """
        try:
            # Ensure image is grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Convert to uint8 if needed
            if gray.dtype != np.uint8:
                gray = np.clip(gray, 0, 255).astype(np.uint8)

            # Extract SIFT features
            keypoints, descriptors = self.sift.detectAndCompute(gray, None)

            if descriptors is None or len(keypoints) < 5:
                # Fall back to ORB if SIFT fails or finds too few keypoints
                keypoints, descriptors = self.orb.detectAndCompute(gray, None)

            if descriptors is None or len(descriptors) == 0:
                # If still no features found, return empty array
                return np.array([])

            # Compute the bag of visual words representation by averaging descriptors
            global_descriptor = np.mean(descriptors, axis=0)

            # Normalize the descriptor
            norm = np.linalg.norm(global_descriptor)
            if norm > 0:
                global_descriptor = global_descriptor / norm

            return global_descriptor
        except Exception as e:
            self.logger.error(f"Error extracting traditional features: {str(e)}")
            return np.array([])

    def extract_features(self, image_data: ImageData) -> np.ndarray:
        """
        Extract combined features from an image.

        Args:
            image_data: Dictionary containing image data

        Returns:
            Combined feature vector
        """
        features = []

        # Extract deep features if model is available
        if self.extractor is not None and self.model is not None:
            deep_features = self._extract_deep_features(image_data['normalized_image'])
            if len(deep_features) > 0:
                features.append(deep_features)

        # Extract traditional features if enabled
        if self.use_traditional_features:
            trad_features = self._extract_traditional_features(image_data['enhanced_image'])
            if len(trad_features) > 0:
                features.append(trad_features)

        # Combine features if we have multiple types
        if len(features) > 0:
            # Concatenate features
            combined = np.concatenate([f.flatten() for f in features])

            # Normalize the combined feature vector
            norm = np.linalg.norm(combined)
            if norm > 0:
                combined = combined / norm

            return combined
        else:
            self.logger.warning(f"No features extracted for image {image_data['id']}")
            return np.array([])

    def _get_cache_path(self, image_id: str) -> Optional[Path]:
        """Get the path for cached embeddings."""
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{image_id}.pkl"

    def _check_cache(self, image_id: str) -> Optional[np.ndarray]:
        """Check if embeddings are cached for this image."""
        cache_path = self._get_cache_path(image_id)
        if cache_path and cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                return None
        return None

    def _save_to_cache(self, image_id: str, embedding: np.ndarray) -> bool:
        """Save embeddings to cache."""
        if self.cache_dir is None:
            return False

        cache_path = self._get_cache_path(image_id)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(embedding, f)
            return True
        except Exception as e:
            self.logger.error(f"Error saving to cache: {str(e)}")
            return False

    def _process_single_image(self, image_data: ImageData) -> Tuple[str, np.ndarray]:
        """Process a single image for embedding."""
        image_id = image_data['id']

        # Check cache first
        cached_embedding = self._check_cache(image_id)
        if cached_embedding is not None:
            return image_id, cached_embedding

        # Extract features
        embedding = self.extract_features(image_data)

        # Save to cache if valid
        if len(embedding) > 0:
            self._save_to_cache(image_id, embedding)

        return image_id, embedding

    def create_embeddings(self, images_data: List[ImageData]) -> Dict[str, np.ndarray]:
        """
        Create embeddings for a batch of images.

        Args:
            images_data: List of image data dictionaries

        Returns:
            Dictionary mapping image IDs to embeddings
        """
        if not images_data:
            return {}

        self.logger.info(f"Creating embeddings for {len(images_data)} images")

        # Process images in parallel
        embeddings = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._process_single_image, img_data) for img_data in images_data]

            for future in tqdm(futures, desc="Extracting features", total=len(futures)):
                image_id, embedding = future.result()
                if len(embedding) > 0:
                    embeddings[image_id] = embedding

        # Check if we have enough embeddings
        if len(embeddings) < len(images_data):
            self.logger.warning(f"Only created {len(embeddings)} embeddings from {len(images_data)} images")

        # Skip dimensionality reduction if we don't have enough embeddings
        if len(embeddings) <= 1:
            return embeddings

        # Apply dimensionality reduction if specified
        if self.dimensionality_reduction and len(embeddings) >= 2:
            embeddings = self._reduce_dimensions(embeddings)

        return embeddings

    def _reduce_dimensions(self, embeddings: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Reduce dimensions of embeddings using the specified method.

        Args:
            embeddings: Dictionary mapping image IDs to embeddings

        Returns:
            Dictionary with reduced embeddings
        """
        # Stack embeddings for processing
        ids = list(embeddings.keys())
        embedding_matrix = np.vstack([embeddings[id_] for id_ in ids])

        if embedding_matrix.shape[0] <= 1:
            return embeddings

        # Choose dimensionality reduction method
        if self.dimensionality_reduction == 'pca':
            # Use PCA for reduction
            n_components = min(self.target_dims, embedding_matrix.shape[0] - 1, embedding_matrix.shape[1])
            self.dim_reducer = PCA(n_components=n_components)
        elif self.dimensionality_reduction == 'umap':
            # Use UMAP for non-linear reduction
            n_components = min(self.target_dims, embedding_matrix.shape[0] - 1)
            self.dim_reducer = umap.UMAP(n_components=n_components)
        else:
            # No reduction
            return embeddings

        # Fit and transform
        try:
            reduced = self.dim_reducer.fit_transform(embedding_matrix)

            # Reconstruct the dictionary
            reduced_embeddings = {}
            for i, id_ in enumerate(ids):
                reduced_embeddings[id_] = reduced[i]

            self.logger.info(f"Reduced embeddings from {embedding_matrix.shape[1]} to {reduced.shape[1]} dimensions")
            return reduced_embeddings
        except Exception as e:
            self.logger.error(f"Error in dimensionality reduction: {str(e)}")
            return embeddings  # Return original embeddings on error