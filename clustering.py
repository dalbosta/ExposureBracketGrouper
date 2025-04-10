"""Step 3: Scene-Level Clustering module for HDR image grouping."""

import numpy as np
import hdbscan
from sklearn.metrics.pairwise import cosine_similarity
import logging
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass

from utils import ImageData


@dataclass
class ClusterResult:
    """Results from clustering operation."""
    cluster_id: int  # -1 for noise points
    confidence: float  # Cluster membership confidence


class SceneClusterer:
    """Groups images into scenes based on visual content, ignoring exposure differences."""

    def __init__(
            self,
            min_cluster_size: int = 2,
            min_samples: int = 1,
            cluster_selection_epsilon: float = 0.1,
            metric: str = 'eucledian',
            cluster_selection_method: str = 'leaf'
    ):
        """
        Initialize the scene clusterer.

        Args:
            min_cluster_size: Minimum size for a cluster
            min_samples: Min samples for a point to be a core point
            cluster_selection_epsilon: Distance threshold for cluster merging
            metric: Distance metric to use
            cluster_selection_method: Method to select flat clusters
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.metric = metric
        self.cluster_selection_method = cluster_selection_method
        self.logger = logging.getLogger('hdr_grouping.clustering')
        self.clusterer = None

    def cluster_embeddings(
            self,
            embeddings: Dict[str, np.ndarray],
            images_data: List[ImageData]
    ) -> Dict[int, List[Tuple[str, float]]]:
        """
        Cluster image embeddings into scene groups.

        Args:
            embeddings: Dictionary mapping image IDs to embeddings
            images_data: List of image data dictionaries

        Returns:
            Dictionary mapping cluster IDs to lists of (image_ID, confidence) tuples
        """
        if not embeddings:
            self.logger.warning("No embeddings provided for clustering")
            return {}

        # Create ID mapping for ordered access
        id_to_index = {id_: i for i, id_ in enumerate(embeddings.keys())}
        index_to_id = {i: id_ for id_, i in id_to_index.items()}

        # Create embedding matrix
        embedding_matrix = np.vstack([embeddings[id_] for id_ in embeddings.keys()])

        # Initialize and fit HDBSCAN
        self.logger.info(f"Clustering {len(embeddings)} embeddings with HDBSCAN")

        # If using cosine distance, precompute the distance matrix
        if self.metric == 'cosine':
            # Convert cosine similarity to distance (1 - similarity)
            self.logger.info("Using precomputed cosine distance matrix")
            # Explicitly convert to double (float64) to avoid dtype mismatch
            distance_matrix = (1 - cosine_similarity(embedding_matrix)).astype(np.float64)

            self.clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                metric='precomputed',  # Use precomputed distances
                cluster_selection_epsilon=self.cluster_selection_epsilon,
                cluster_selection_method=self.cluster_selection_method,
                prediction_data=True
            )

            self.clusterer.fit(distance_matrix)
        else:
            # For other metrics, let's use 'euclidean' as a safe default
            actual_metric = self.metric if self.metric != 'cosine' else 'euclidean'
            self.logger.info(f"Using metric: {actual_metric}")
            self.clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                metric=actual_metric,
                cluster_selection_epsilon=self.cluster_selection_epsilon,
                cluster_selection_method=self.cluster_selection_method,
                prediction_data=True
            )

            self.clusterer.fit(embedding_matrix)

        # Get cluster labels and probabilities
        labels = self.clusterer.labels_
        if hasattr(self.clusterer, 'probabilities_'):
            probabilities = self.clusterer.probabilities_
        else:
            # If probabilities not available, use a default value of 1.0
            probabilities = np.ones_like(labels, dtype=float)

        # Organize results by cluster
        clusters = {}
        for i, (label, prob) in enumerate(zip(labels, probabilities)):
            if label not in clusters:
                clusters[label] = []

            img_id = index_to_id[i]
            clusters[label].append((img_id, prob))

        # Sort images within each cluster by confidence
        for label in clusters:
            clusters[label].sort(key=lambda x: x[1], reverse=True)

        # Get statistics about clustering
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_count = list(labels).count(-1)

        self.logger.info(f"Found {n_clusters} clusters and {noise_count} noise points")

        return clusters

    def assign_outliers(
            self,
            clusters: Dict[int, List[Tuple[str, float]]],
            embeddings: Dict[str, np.ndarray],
            threshold: float = 0.8
    ) -> Dict[int, List[Tuple[str, float]]]:
        """
        Try to assign noise points to the nearest cluster if similarity is high enough.

        Args:
            clusters: Dictionary mapping cluster IDs to (image_ID, confidence) lists
            embeddings: Dictionary mapping image IDs to embeddings
            threshold: Similarity threshold for assignment

        Returns:
            Updated clusters dictionary with reassigned noise points
        """
        # Skip if no noise points or no proper clusters
        if -1 not in clusters or len(clusters) <= 1:
            return clusters

        # Make a copy to modify
        updated_clusters = {k: v[:] for k, v in clusters.items()}

        # Get noise points
        noise_points = updated_clusters.pop(-1, [])

        if not noise_points:
            return updated_clusters

        self.logger.info(f"Attempting to assign {len(noise_points)} noise points to clusters")

        # For each noise point
        for img_id, _ in noise_points:
            noise_embedding = embeddings[img_id]

            best_cluster = -1
            best_similarity = -1

            # Compute average embedding for each cluster
            for cluster_id, cluster_items in updated_clusters.items():
                # Skip empty clusters
                if not cluster_items:
                    continue

                # Get embeddings for this cluster
                cluster_embeddings = np.vstack([
                    embeddings[cluster_img_id]
                    for cluster_img_id, _ in cluster_items
                    if cluster_img_id in embeddings
                ])

                # Calculate mean embedding
                cluster_centroid = np.mean(cluster_embeddings, axis=0)

                # Calculate similarity to noise point
                similarity = cosine_similarity(
                    noise_embedding.reshape(1, -1),
                    cluster_centroid.reshape(1, -1)
                )[0][0]

                # Update best match if better
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_cluster = cluster_id

            # Assign to best cluster if similarity exceeds threshold
            if best_similarity >= threshold and best_cluster != -1:
                updated_clusters[best_cluster].append((img_id, best_similarity))
            else:
                # Keep as noise
                if -1 not in updated_clusters:
                    updated_clusters[-1] = []
                updated_clusters[-1].append((img_id, 0.0))

        return updated_clusters

    def get_cluster_statistics(
            self,
            clusters: Dict[int, List[Tuple[str, float]]],
            embeddings: Dict[str, np.ndarray]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Calculate statistics for each cluster.

        Args:
            clusters: Dictionary mapping cluster IDs to (image_ID, confidence) lists
            embeddings: Dictionary mapping image IDs to embeddings

        Returns:
            Dictionary with cluster statistics
        """
        stats = {}

        for cluster_id, items in clusters.items():
            # Skip empty clusters
            if not items:
                continue

            # Extract image IDs and confidences
            img_ids = [item[0] for item in items]
            confidences = [item[1] for item in items]

            # Get embeddings for this cluster
            cluster_embeddings = np.vstack([
                embeddings[img_id] for img_id in img_ids if img_id in embeddings
            ])

            # Calculate statistics
            stats[cluster_id] = {
                'size': len(items),
                'avg_confidence': np.mean(confidences),
                'min_confidence': np.min(confidences),
                'is_noise': cluster_id == -1,
                'coherence': self._calculate_cluster_coherence(cluster_embeddings)
            }

        return stats

    def _calculate_cluster_coherence(self, embeddings: np.ndarray) -> float:
        """
        Calculate the coherence of a cluster based on pairwise similarities.

        Args:
            embeddings: Matrix of embeddings for the cluster

        Returns:
            Coherence score (higher is more coherent)
        """
        if embeddings.shape[0] <= 1:
            return 1.0  # A single point is perfectly coherent

        # Calculate pairwise similarities
        similarities = cosine_similarity(embeddings)

        # Calculate average similarity (excluding self-similarity on diagonal)
        np.fill_diagonal(similarities, 0)
        total_similarity = similarities.sum()
        num_pairs = embeddings.shape[0] * (embeddings.shape[0] - 1)

        if num_pairs == 0:
            return 1.0

        return total_similarity / num_pairs