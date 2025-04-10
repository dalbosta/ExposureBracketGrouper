"""Step 5: Final Cleanup & Validation module for HDR image grouping."""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
from sklearn.metrics.pairwise import cosine_similarity

from utils import ImageData


class ClusterValidator:
    """Validates and refines clustering results for high confidence groupings."""

    def __init__(
            self,
            similarity_threshold: float = 0.7,
            min_hdr_score: float = 0.4,
            geometry_threshold: float = 0.6,
            min_cluster_size: int = 2,
            max_cluster_size: int = 20
    ):
        """
        Initialize the cluster validator.

        Args:
            similarity_threshold: Minimum feature similarity for valid clusters
            min_hdr_score: Minimum HDR quality score for a valid sequence
            geometry_threshold: Threshold for geometric consistency
            min_cluster_size: Minimum size for valid clusters (after cleanup)
            max_cluster_size: Maximum size for a single cluster
        """
        self.similarity_threshold = similarity_threshold
        self.min_hdr_score = min_hdr_score
        self.geometry_threshold = geometry_threshold
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        self.logger = logging.getLogger('hdr_grouping.validation')

    def check_feature_similarity(
            self,
            cluster_embeddings: Dict[str, np.ndarray]
    ) -> Tuple[float, bool]:
        """
        Check feature similarity within a cluster.

        Args:
            cluster_embeddings: Dictionary mapping image IDs to embeddings

        Returns:
            Tuple of (similarity score, is_valid)
        """
        if len(cluster_embeddings) <= 1:
            return 1.0, True

        # Create embedding matrix
        embeddings = np.vstack(list(cluster_embeddings.values()))

        # Calculate pairwise cosine similarities
        similarities = cosine_similarity(embeddings)

        # Calculate mean similarity (excluding self-similarity)
        np.fill_diagonal(similarities, 0)
        n = similarities.shape[0]
        if n <= 1:
            return 1.0, True

        mean_similarity = similarities.sum() / (n * (n - 1))

        # Check if mean similarity exceeds threshold
        is_valid = mean_similarity >= self.similarity_threshold

        return mean_similarity, is_valid

    def check_geometric_consistency(
            self,
            images: List[ImageData]
    ) -> Tuple[float, bool]:
        """
        Check geometric consistency within a cluster using homography estimation.

        Args:
            images: List of image data dictionaries

        Returns:
            Tuple of (consistency score, is_valid)
        """
        if len(images) <= 1:
            return 1.0, True

        # Use first image as reference
        reference_img = images[0]['enhanced_image']
        reference_gray = reference_img if len(reference_img.shape) == 2 else cv2.cvtColor(reference_img,
                                                                                          cv2.COLOR_BGR2GRAY)

        # Initialize SIFT detector
        sift = cv2.SIFT_create()

        # Extract keypoints from reference image
        kp1, des1 = sift.detectAndCompute(reference_gray, None)

        # Minimum number of good matches to consider it consistent
        min_match_count = 10

        scores = []

        # Compare each image to the reference
        for img_data in images[1:]:
            img = img_data['enhanced_image']
            gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Extract keypoints
            kp2, des2 = sift.detectAndCompute(gray, None)

            # Handle case where no keypoints are found
            if des1 is None or des2 is None or len(kp1) < min_match_count or len(kp2) < min_match_count:
                scores.append(0)
                continue

            # Match keypoints
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)

            # Use FLANN matcher
            flann = cv2.FlannBasedMatcher(index_params, search_params)

            # Get top 2 matches for each keypoint
            matches = flann.knnMatch(des1, des2, k=2)

            # Apply ratio test to find good matches
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

            # Calculate score based on number of good matches
            if len(good_matches) > min_match_count:
                try:
                    # Extract matched keypoints
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                    # Find homography matrix
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                    if M is not None:
                        # Calculate percentage of inliers
                        inliers = np.sum(mask)
                        inlier_ratio = inliers / len(good_matches)
                        scores.append(inlier_ratio)
                    else:
                        scores.append(0)
                except Exception:
                    # Homography estimation failed
                    scores.append(0)
            else:
                # Not enough good matches
                scores.append(0)

        # Calculate average score
        avg_score = np.mean(scores) if scores else 0

        # Determine if the cluster is geometrically consistent
        is_valid = avg_score >= self.geometry_threshold

        return avg_score, is_valid

    def validate_cluster(
            self,
            cluster_items: List[Tuple[str, float]],
            images_data: Dict[str, ImageData],
            embeddings: Dict[str, np.ndarray],
            hdr_score: float
    ) -> Dict[str, Any]:
        """
        Validate a cluster based on multiple criteria.

        Args:
            cluster_items: List of (image_ID, confidence) tuples
            images_data: Dictionary mapping image IDs to image data
            embeddings: Dictionary mapping image IDs to embeddings
            hdr_score: HDR sequence quality score

        Returns:
            Validation results
        """
        # Extract cluster image data and embeddings
        cluster_images = [images_data[img_id] for img_id, _ in cluster_items if img_id in images_data]
        cluster_embeddings = {img_id: embeddings[img_id] for img_id, _ in cluster_items if img_id in embeddings}

        # Check feature similarity
        similarity_score, similarity_valid = self.check_feature_similarity(cluster_embeddings)

        # Check geometric consistency
        geometry_score, geometry_valid = self.check_geometric_consistency(cluster_images)

        # Check HDR sequence quality
        hdr_valid = hdr_score >= self.min_hdr_score

        # Check cluster size
        size_valid = self.min_cluster_size <= len(cluster_items) <= self.max_cluster_size

        # Overall validation
        is_valid = similarity_valid and geometry_valid and hdr_valid and size_valid

        # Confidence score (weighted combination of all factors)
        confidence = (
                0.3 * similarity_score +
                0.3 * geometry_score +
                0.3 * min(1.0, hdr_score) +
                0.1 * (1.0 if size_valid else 0.0)
        )

        return {
            'is_valid': is_valid,
            'confidence': confidence,
            'details': {
                'similarity_score': similarity_score,
                'similarity_valid': similarity_valid,
                'geometry_score': geometry_score,
                'geometry_valid': geometry_valid,
                'hdr_score': hdr_score,
                'hdr_valid': hdr_valid,
                'size': len(cluster_items),
                'size_valid': size_valid
            }
        }

    def suggest_split(
            self,
            cluster_items: List[Tuple[str, float]],
            images_data: Dict[str, ImageData],
            embeddings: Dict[str, np.ndarray]
    ) -> List[List[Tuple[str, float]]]:
        """
        Suggest splitting a large or incoherent cluster into smaller ones.

        Args:
            cluster_items: List of (image_ID, confidence) tuples
            images_data: Dictionary mapping image IDs to image data
            embeddings: Dictionary mapping image IDs to embeddings

        Returns:
            List of split sub-clusters
        """
        if len(cluster_items) <= self.min_cluster_size:
            return [cluster_items]

        # Check if cluster is already valid and not too large
        if len(cluster_items) <= self.max_cluster_size:
            feature_sim, sim_valid = self.check_feature_similarity({
                img_id: embeddings[img_id] for img_id, _ in cluster_items if img_id in embeddings
            })

            if sim_valid:
                return [cluster_items]

        # If too large or not coherent, split using hierarchical clustering
        # Extract embeddings for the cluster
        cluster_embeddings = np.vstack([
            embeddings[img_id] for img_id, _ in cluster_items if img_id in embeddings
        ])

        # Map embeddings to original items
        valid_indices = [i for i, (img_id, _) in enumerate(cluster_items) if img_id in embeddings]
        valid_items = [cluster_items[i] for i in valid_indices]

        if len(valid_items) <= 1:
            return [cluster_items]

        # Perform hierarchical clustering
        from scipy.cluster.hierarchy import linkage, fcluster

        # Compute linkage
        Z = linkage(cluster_embeddings, method='ward')

        # Try different number of clusters, from 2 to k
        k = max(2, min(len(valid_items) // self.min_cluster_size, 5))

        best_split = None
        best_score = -1

        for n_clusters in range(2, k + 1):
            # Get cluster labels
            labels = fcluster(Z, n_clusters, criterion='maxclust')

            # Form sub-clusters
            subclusters = [[] for _ in range(n_clusters)]
            for i, label in enumerate(labels):
                subclusters[label - 1].append(valid_items[i])

            # Filter out empty subclusters
            subclusters = [sc for sc in subclusters if sc]

            # Score the split
            score = self._evaluate_split(subclusters, embeddings)

            if score > best_score:
                best_score = score
                best_split = subclusters

        # If we couldn't find a good split, return the original cluster
        if best_split is None or best_score < 0.5:
            return [cluster_items]

        return best_split

    def _evaluate_split(
            self,
            subclusters: List[List[Tuple[str, float]]],
            embeddings: Dict[str, np.ndarray]
    ) -> float:
        """
        Evaluate the quality of a cluster split.

        Args:
            subclusters: List of split sub-clusters
            embeddings: Dictionary mapping image IDs to embeddings

        Returns:
            Split quality score
        """
        # If any subcluster is too small, penalize
        if any(len(sc) < self.min_cluster_size for sc in subclusters):
            return 0.0

        # Calculate intra-cluster similarity for each subcluster
        intra_similarities = []

        for sc in subclusters:
            sc_embeddings = {img_id: embeddings[img_id] for img_id, _ in sc if img_id in embeddings}

            if len(sc_embeddings) <= 1:
                intra_similarities.append(1.0)
                continue

            # Create embedding matrix
            sc_emb_matrix = np.vstack(list(sc_embeddings.values()))

            # Calculate pairwise similarities
            similarities = cosine_similarity(sc_emb_matrix)

            # Calculate mean similarity (excluding self-similarity)
            np.fill_diagonal(similarities, 0)
            n = similarities.shape[0]
            if n <= 1:
                intra_similarities.append(1.0)
                continue

            mean_similarity = similarities.sum() / (n * (n - 1))
            intra_similarities.append(mean_similarity)

        # Calculate inter-cluster separation
        if len(subclusters) <= 1:
            return 0.0

        # Get centroids for each subcluster
        centroids = []
        for sc in subclusters:
            sc_embeddings = [embeddings[img_id] for img_id, _ in sc if img_id in embeddings]
            if sc_embeddings:
                centroids.append(np.mean(sc_embeddings, axis=0))

        # Calculate pairwise distances between centroids
        inter_similarities = []
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                sim = cosine_similarity(centroids[i].reshape(1, -1), centroids[j].reshape(1, -1))[0][0]
                inter_similarities.append(sim)

        # Average inter-cluster similarity (lower is better)
        avg_inter_sim = np.mean(inter_similarities) if inter_similarities else 1.0

        # Average intra-cluster similarity (higher is better)
        avg_intra_sim = np.mean(intra_similarities) if intra_similarities else 0.0

        # Final score: combination of high intra-cluster similarity and low inter-cluster similarity
        score = avg_intra_sim * (1 - avg_inter_sim)

        return score

    def suggest_merge(
            self,
            clusters: Dict[int, List[Tuple[str, float]]],
            embeddings: Dict[str, np.ndarray]
    ) -> Dict[int, List[int]]:
        """
        Suggest merging similar small clusters.

        Args:
            clusters: Dictionary mapping cluster IDs to (image_ID, confidence) tuples
            embeddings: Dictionary mapping image IDs to embeddings

        Returns:
            Dictionary mapping original cluster IDs to lists of clusters to merge with
        """
        # Skip if fewer than 2 clusters
        if len(clusters) < 2:
            return {}

        # Calculate cluster centroids
        centroids = {}
        for cluster_id, items in clusters.items():
            # Skip very large clusters
            if len(items) > self.max_cluster_size // 2:
                continue

            # Get embeddings for this cluster
            cluster_embeddings = [
                embeddings[img_id] for img_id, _ in items if img_id in embeddings
            ]

            if not cluster_embeddings:
                continue

            # Calculate centroid
            centroids[cluster_id] = np.mean(cluster_embeddings, axis=0)

        # Calculate pairwise similarities between centroids
        merge_candidates = {}

        for id1 in centroids:
            for id2 in centroids:
                if id1 >= id2:
                    continue

                # Calculate similarity
                sim = cosine_similarity(
                    centroids[id1].reshape(1, -1),
                    centroids[id2].reshape(1, -1)
                )[0][0]

                # Check if clusters are similar enough to merge
                if sim >= self.similarity_threshold * 1.1:  # Higher threshold for merging
                    # Check if combined size is acceptable
                    combined_size = len(clusters[id1]) + len(clusters[id2])
                    if combined_size <= self.max_cluster_size:
                        # Add to merge candidates
                        if id1 not in merge_candidates:
                            merge_candidates[id1] = []
                        merge_candidates[id1].append(id2)

        return merge_candidates