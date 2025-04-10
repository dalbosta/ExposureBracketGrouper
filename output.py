"""Step 6: Output & Logging module for HDR image grouping."""

import os
import json
import shutil
import logging
from pathlib import Path
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from datetime import datetime

from utils import ImageData, PathLike, setup_output_dir


class ResultsExporter:
    """Handles exporting and logging of pipeline results."""

    def __init__(
            self,
            output_base_dir: PathLike,
            create_thumbnails: bool = True,
            thumbnail_size: Tuple[int, int] = (256, 256),
            export_json: bool = True,
            copy_images: bool = True
    ):
        """
        Initialize the results exporter.

        Args:
            output_base_dir: Base directory for outputs
            create_thumbnails: Whether to create thumbnail previews
            thumbnail_size: Size for thumbnails
            export_json: Whether to export JSON metadata
            copy_images: Whether to copy images to output folders
        """
        self.output_dir = setup_output_dir(output_base_dir)
        self.create_thumbnails = create_thumbnails
        self.thumbnail_size = thumbnail_size
        self.export_json = export_json
        self.copy_images = copy_images
        self.logger = logging.getLogger('hdr_grouping.output')

        # Create thumbnails directory if needed
        if self.create_thumbnails:
            self.thumbnails_dir = self.output_dir / "thumbnails"
            os.makedirs(self.thumbnails_dir, exist_ok=True)

        # Create directory for unassigned images
        self.unassigned_dir = self.output_dir / "unassigned"
        os.makedirs(self.unassigned_dir, exist_ok=True)

    def _create_thumbnail(self, image: np.ndarray, output_path: Path) -> bool:
        """Create a thumbnail of an image."""
        try:
            # Resize image to thumbnail size
            h, w = image.shape[:2]
            target_w, target_h = self.thumbnail_size

            # Calculate aspect ratio
            aspect = w / h
            target_aspect = target_w / target_h

            if aspect > target_aspect:
                # Image is wider than target
                new_w = target_w
                new_h = int(target_w / aspect)
            else:
                # Image is taller than target
                new_h = target_h
                new_w = int(target_h * aspect)

            # Resize image
            thumbnail = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Create a black canvas of target size
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

            # Calculate padding
            pad_x = (target_w - new_w) // 2
            pad_y = (target_h - new_h) // 2

            # Place the resized image on the canvas
            canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = thumbnail

            # Save the thumbnail
            cv2.imwrite(str(output_path), canvas)
            return True
        except Exception as e:
            self.logger.error(f"Error creating thumbnail: {str(e)}")
            return False

    def export_clusters(
            self,
            valid_clusters: Dict[int, Dict[str, Any]],
            images_data: Dict[str, ImageData]
    ) -> Dict[str, Any]:
        """
        Export validated clusters to output directory.

        Args:
            valid_clusters: Dictionary mapping cluster IDs to cluster information
            images_data: Dictionary mapping image IDs to image data

        Returns:
            Export statistics
        """
        self.logger.info(f"Exporting {len(valid_clusters)} valid clusters")

        stats = {
            'total_clusters': len(valid_clusters),
            'total_images': 0,
            'exported_clusters': 0,
            'exported_images': 0,
            'skipped_images': 0
        }

        # Check if we have any valid clusters
        if not valid_clusters:
            self.logger.warning("No valid clusters to export")
            return stats

        # Prepare export metadata
        export_metadata = {
            'timestamp': datetime.now().isoformat(),
            'clusters': {}
        }

        # Process each cluster
        for cluster_id, cluster_info in valid_clusters.items():
            # Skip invalid clusters
            if not cluster_info.get('validation', {}).get('is_valid', False):
                continue

            # Get clean exposure sequence (no duplicates or accidental shots)
            exposure_sequence = [
                info for info in cluster_info.get('exposure_info', {}).get('exposure_sequence', [])
                if not info.get('is_duplicate', False) and not info.get('is_accidental', False)
            ]

            # Skip if no valid images
            if not exposure_sequence:
                continue

            # Create cluster directory
            cluster_dir = self.output_dir / f"cluster_{cluster_id}"
            os.makedirs(cluster_dir, exist_ok=True)

            # Prepare cluster metadata
            cluster_metadata = {
                'cluster_id': cluster_id,
                'confidence': cluster_info.get('validation', {}).get('confidence', 0),
                'hdr_score': cluster_info.get('exposure_info', {}).get('hdr_score', 0),
                'image_count': len(exposure_sequence),
                'images': []
            }

            # Copy images to cluster directory
            for i, info in enumerate(exposure_sequence):
                img_id = info['id']
                img_data = images_data.get(img_id)

                if img_data is None:
                    stats['skipped_images'] += 1
                    continue

                # Determine original file path
                src_path = Path(img_data['file_path'])

                # Format relative EV as +/-X.X
                ev_str = f"{info['ev']:+.1f}".replace('+0.0', '0.0')

                # Create destination filename with exposure indication
                dst_filename = f"{i:02d}_EV{ev_str}_{src_path.name}"
                dst_path = cluster_dir / dst_filename

                # Copy the image file if requested
                copied = False
                if self.copy_images:
                    try:
                        shutil.copy2(src_path, dst_path)
                        copied = True
                    except Exception as e:
                        self.logger.error(f"Error copying image {src_path}: {str(e)}")

                # Create thumbnail if requested
                thumbnail_path = None
                if self.create_thumbnails:
                    thumb_filename = f"cluster_{cluster_id}_{i:02d}.jpg"
                    thumbnail_path = self.thumbnails_dir / thumb_filename
                    self._create_thumbnail(img_data['original_image'], thumbnail_path)

                # Add to metadata
                image_meta = {
                    'id': img_id,
                    'source_path': str(src_path),
                    'target_path': str(dst_path) if copied else None,
                    'thumbnail_path': str(thumbnail_path) if thumbnail_path else None,
                    'exposure_value': info['ev'],
                    'is_duplicate': info.get('is_duplicate', False),
                    'is_accidental': info.get('is_accidental', False),
                    'sequence_position': i
                }

                cluster_metadata['images'].append(image_meta)
                stats['exported_images'] += 1

            # Save cluster metadata if requested
            if self.export_json:
                meta_path = cluster_dir / "metadata.json"
                with open(meta_path, 'w') as f:
                    json.dump(cluster_metadata, f, indent=2)

            # Add to export metadata
            export_metadata['clusters'][str(cluster_id)] = cluster_metadata
            stats['exported_clusters'] += 1
            stats['total_images'] += len(exposure_sequence)

        # Export overall metadata if requested
        if self.export_json:
            meta_path = self.output_dir / "metadata.json"
            with open(meta_path, 'w') as f:
                json.dump(export_metadata, f, indent=2)

        self.logger.info(f"Export complete: {stats['exported_clusters']} clusters, {stats['exported_images']} images")
        return stats

    def export_unassigned_images(
            self,
            unassigned_images: List[str],
            images_data: Dict[str, ImageData]
    ) -> int:
        """
        Export unassigned images to a separate directory.

        Args:
            unassigned_images: List of unassigned image IDs
            images_data: Dictionary mapping image IDs to image data

        Returns:
            Number of exported unassigned images
        """
        if not unassigned_images:
            return 0

        self.logger.info(f"Exporting {len(unassigned_images)} unassigned images")

        # Create unassigned directory if it doesn't exist
        os.makedirs(self.unassigned_dir, exist_ok=True)

        # Export metadata
        unassigned_meta = {
            'count': len(unassigned_images),
            'images': []
        }

        # Copy each unassigned image
        exported_count = 0
        for img_id in unassigned_images:
            img_data = images_data.get(img_id)

            if img_data is None:
                continue

            # Determine original file path
            src_path = Path(img_data['file_path'])

            # Create destination path
            dst_path = self.unassigned_dir / src_path.name

            # Copy the image file if requested
            copied = False
            if self.copy_images:
                try:
                    shutil.copy2(src_path, dst_path)
                    copied = True
                    exported_count += 1
                except Exception as e:
                    self.logger.error(f"Error copying unassigned image {src_path}: {str(e)}")

            # Create thumbnail if requested
            thumbnail_path = None
            if self.create_thumbnails:
                thumb_filename = f"unassigned_{img_id[-8:]}.jpg"
                thumbnail_path = self.thumbnails_dir / thumb_filename
                self._create_thumbnail(img_data['original_image'], thumbnail_path)

            # Add to metadata
            unassigned_meta['images'].append({
                'id': img_id,
                'source_path': str(src_path),
                'target_path': str(dst_path) if copied else None,
                'thumbnail_path': str(thumbnail_path) if thumbnail_path else None
            })

        # Save metadata if requested
        if self.export_json:
            meta_path = self.unassigned_dir / "metadata.json"
            with open(meta_path, 'w') as f:
                json.dump(unassigned_meta, f, indent=2)

        self.logger.info(f"Exported {exported_count} unassigned images")
        return exported_count

    def generate_report(
            self,
            stats: Dict[str, Any],
            unassigned_count: int,
            pipeline_stats: Dict[str, Any]
    ) -> None:
        """
        Generate a summary report of the HDR grouping process.

        Args:
            stats: Export statistics
            unassigned_count: Number of unassigned images
            pipeline_stats: Additional pipeline statistics
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'output_directory': str(self.output_dir),
            'clusters': {
                'total': stats['total_clusters'],
                'exported': stats['exported_clusters']
            },
            'images': {
                'total_processed': pipeline_stats.get('total_images', 0),
                'in_valid_clusters': stats['total_images'],
                'exported': stats['exported_images'],
                'unassigned': unassigned_count,
                'skipped': stats['skipped_images']
            },
            'pipeline_stats': pipeline_stats
        }

        # Save report
        report_path = self.output_dir / "report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Also generate a text summary
        summary = [
            "HDR Image Grouping Summary",
            "=========================",
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Output directory: {self.output_dir}",
            "",
            "Cluster Statistics:",
            f"  Total clusters found: {stats['total_clusters']}",
            f"  Valid clusters exported: {stats['exported_clusters']}",
            "",
            "Image Statistics:",
            f"  Total images processed: {pipeline_stats.get('total_images', 0)}",
            f"  Images in valid clusters: {stats['total_images']}",
            f"  Images successfully exported: {stats['exported_images']}",
            f"  Unassigned images: {unassigned_count}",
            "",
            "Pipeline Performance:",
            f"  Ingest time: {pipeline_stats.get('ingest_time', 0):.2f}s",
            f"  Embedding time: {pipeline_stats.get('embedding_time', 0):.2f}s",
            f"  Clustering time: {pipeline_stats.get('clustering_time', 0):.2f}s",
            f"  Validation time: {pipeline_stats.get('validation_time', 0):.2f}s",
            f"  Export time: {pipeline_stats.get('export_time', 0):.2f}s",
            f"  Total time: {pipeline_stats.get('total_time', 0):.2f}s",
        ]

        # Save text summary
        summary_path = self.output_dir / "summary.txt"
        with open(summary_path, 'w') as f:
            f.write("\n".join(summary))

        self.logger.info(f"Report generated at {report_path}")