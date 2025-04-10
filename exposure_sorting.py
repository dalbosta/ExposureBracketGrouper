"""Step 4: Internal Exposure Sequence Identification for HDR image grouping."""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
import imagehash
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from utils import ImageData, estimate_exposure_value


class ExposureSorter:
    """Identifies exposure sequences within each scene cluster."""

    def __init__(
            self,
            ssim_threshold: float = 0.7,
            hash_threshold: int = 10,
            min_ev_difference: float = 0.3
    ):
        """
        Initialize the exposure sorter.

        Args:
            ssim_threshold: Threshold for structural similarity (higher means more similar)
            hash_threshold: Threshold for perceptual hash difference (lower means more similar)
            min_ev_difference: Minimum EV difference for distinct exposure levels
        """
        self.ssim_threshold = ssim_threshold
        self.hash_threshold = hash_threshold
        self.min_ev_difference = min_ev_difference
        self.logger = logging.getLogger('hdr_grouping.exposure_sorting')

    def sort_by_exposure(
            self,
            cluster_items: List[Tuple[str, float]],
            images_data: Dict[str, ImageData]
    ) -> List[Dict[str, Any]]:
        """
        Sort images in a cluster by exposure level.

        Args:
            cluster_items: List of (image_ID, confidence) tuples
            images_data: Dictionary mapping image IDs to image data

        Returns:
            List of image data sorted by exposure (dark to bright)
        """
        # Prepare list of images with exposure values
        exposure_info = []

        for img_id, confidence in cluster_items:
            if img_id not in images_data:
                continue

            img_data = images_data[img_id]

            # Calculate relative exposure value
            ev = estimate_exposure_value(img_data['original_image'])

            exposure_info.append({
                'id': img_id,
                'data': img_data,
                'ev': ev,
                'cluster_confidence': confidence
            })

        # Sort by exposure value (dark to bright)
        exposure_info.sort(key=lambda x: x['ev'])

        return exposure_info

    def identify_duplicates(
            self,
            exposure_info: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Identify and mark duplicate or nearly identical images.

        Args:
            exposure_info: List of image info dictionaries with exposure values

        Returns:
            Updated list with duplicate flags
        """
        if not exposure_info:
            return []

        n_images = len(exposure_info)
        if n_images <= 1:
            # Single image can't have duplicates
            exposure_info[0]['is_duplicate'] = False
            return exposure_info

        # Compute perceptual hashes for each image
        for info in exposure_info:
            # Get original image
            img = info['data']['original_image']
            # Convert to PIL format for hashing
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # Compute perceptual hash
            info['phash'] = imagehash.phash(pil_img)

        # Compare each pair of images
        for i in range(n_images):
            exposure_info[i]['is_duplicate'] = False

            # Skip if already marked as duplicate
            if exposure_info[i].get('is_duplicate', False):
                continue

            for j in range(i + 1, n_images):
                # Skip if already marked as duplicate
                if exposure_info[j].get('is_duplicate', False):
                    continue

                # Check if exposure values are similar
                ev_diff = abs(exposure_info[i]['ev'] - exposure_info[j]['ev'])

                # Skip if exposures are notably different
                if ev_diff > self.min_ev_difference:
                    continue

                # Compare hashes
                hash_diff = exposure_info[i]['phash'] - exposure_info[j]['phash']

                if hash_diff <= self.hash_threshold:
                    # Images are likely duplicates
                    # Keep the one with higher cluster confidence
                    if exposure_info[i]['cluster_confidence'] < exposure_info[j]['cluster_confidence']:
                        exposure_info[i]['is_duplicate'] = True
                    else:
                        exposure_info[j]['is_duplicate'] = True

                # Additional SSIM check if hash comparison is close
                elif hash_diff <= self.hash_threshold * 2:
                    # Compute SSIM between images
                    img1_gray = cv2.cvtColor(exposure_info[i]['data']['normalized_image'], cv2.COLOR_BGR2GRAY)
                    img2_gray = cv2.cvtColor(exposure_info[j]['data']['normalized_image'], cv2.COLOR_BGR2GRAY)

                    similarity = ssim(img1_gray, img2_gray)

                    if similarity >= self.ssim_threshold:
                        # Images are likely duplicates
                        # Keep the one with higher cluster confidence
                        if exposure_info[i]['cluster_confidence'] < exposure_info[j]['cluster_confidence']:
                            exposure_info[i]['is_duplicate'] = True
                        else:
                            exposure_info[j]['is_duplicate'] = True

        return exposure_info

    def identify_accidental_shots(
            self,
            exposure_info: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Identify likely accidental shots within a sequence.

        Args:
            exposure_info: List of image info dictionaries

        Returns:
            Updated list with accidental shot flags
        """
        if not exposure_info:
            return []

        # Initialize accidental flag for all images
        for info in exposure_info:
            info['is_accidental'] = False

        n_images = len(exposure_info)
        if n_images <= 2:
            # Not enough images to identify accidental shots
            return exposure_info

        # Group images by similar exposure
        ev_groups = []
        current_group = [0]

        for i in range(1, n_images):
            prev_ev = exposure_info[current_group[-1]]['ev']
            curr_ev = exposure_info[i]['ev']

            ev_diff = abs(curr_ev - prev_ev)

            if ev_diff <= self.min_ev_difference:
                # Similar exposure level, add to current group
                current_group.append(i)
            else:
                # Different exposure level, start a new group
                ev_groups.append(current_group)
                current_group = [i]

        # Add the last group
        if current_group:
            ev_groups.append(current_group)

        # Check for singleton groups that might be accidental shots
        for group in ev_groups:
            if len(group) == 1:
                # This is a singleton group
                idx = group[0]

                # If it's at the extreme end of exposure range, it's probably intentional
                if idx == 0 or idx == n_images - 1:
                    continue

                # Calculate exposure differences to adjacent groups
                prev_idx = idx - 1
                next_idx = idx + 1

                prev_ev_diff = abs(exposure_info[idx]['ev'] - exposure_info[prev_idx]['ev'])
                next_ev_diff = abs(exposure_info[idx]['ev'] - exposure_info[next_idx]['ev'])

                # If both differences are large, likely an accidental shot
                if min(prev_ev_diff, next_ev_diff) > self.min_ev_difference * 2:
                    exposure_info[idx]['is_accidental'] = True

        return exposure_info

    def score_hdr_sequence(
            self,
            exposure_info: List[Dict[str, Any]]
    ) -> float:
        """
        Score how well the image sequence works as an HDR stack.

        Args:
            exposure_info: List of image info dictionaries

        Returns:
            HDR quality score (0.0 to 1.0)
        """
        if not exposure_info:
            return 0.0

        # Remove duplicates and accidental shots from consideration
        valid_exposures = [
            info for info in exposure_info
            if not info.get('is_duplicate', False) and not info.get('is_accidental', False)
        ]

        if not valid_exposures:
            return 0.0

        n_valid = len(valid_exposures)

        # If only a single exposure, it's not a proper HDR sequence
        if n_valid == 1:
            return 0.3  # Low but non-zero score

        # Extract exposure values
        evs = [info['ev'] for info in valid_exposures]

        # Calculate exposure range
        ev_range = max(evs) - min(evs)

        # Calculate exposure distribution uniformity
        if n_valid > 2:
            # Sort EVs
            sorted_evs = sorted(evs)
            # Calculate gaps between adjacent EVs
            gaps = [sorted_evs[i + 1] - sorted_evs[i] for i in range(n_valid - 1)]
            # Calculate uniformity as 1 - (std_dev / mean) of gaps
            mean_gap = np.mean(gaps)
            if mean_gap > 0:
                uniformity = 1 - min(1.0, np.std(gaps) / mean_gap)
            else:
                uniformity = 0.0
        else:
            uniformity = 1.0  # Perfect uniformity for 2 exposures

        # Number of exposures score (more is better, up to a point)
        n_score = min(1.0, (n_valid - 1) / 6)  # Up to 7 exposures gets full score

        # Exposure range score (wider range is better for HDR)
        range_score = min(1.0, ev_range / 8)  # 8 EV range gets full score

        # Final score is weighted combination
        final_score = (0.4 * n_score) + (0.4 * range_score) + (0.2 * uniformity)

        return final_score

    def process_cluster(
            self,
            cluster_items: List[Tuple[str, float]],
            images_data: Dict[str, ImageData]
    ) -> Dict[str, Any]:
        """
        Process a cluster to identify and sort exposure sequences.

        Args:
            cluster_items: List of (image_ID, confidence) tuples
            images_data: Dictionary mapping image IDs to image data

        Returns:
            Dictionary with exposure sequence information
        """
        # Sort by exposure
        exposure_info = self.sort_by_exposure(cluster_items, images_data)

        if not exposure_info:
            return {
                'exposure_sequence': [],
                'hdr_score': 0.0,
                'flags': {
                    'has_duplicates': False,
                    'has_accidental_shots': False
                }
            }

        # Identify duplicates
        exposure_info = self.identify_duplicates(exposure_info)

        # Identify accidental shots
        exposure_info = self.identify_accidental_shots(exposure_info)

        # Score HDR sequence
        hdr_score = self.score_hdr_sequence(exposure_info)

        # Check for duplicates and accidental shots
        has_duplicates = any(info.get('is_duplicate', False) for info in exposure_info)
        has_accidental_shots = any(info.get('is_accidental', False) for info in exposure_info)

        return {
            'exposure_sequence': exposure_info,
            'hdr_score': hdr_score,
            'flags': {
                'has_duplicates': has_duplicates,
                'has_accidental_shots': has_accidental_shots
            }
        }