"""Main HDR image grouping pipeline coordinator."""

import time
import logging
from pathlib import Path
from typing import Dict, Optional, Any
import os

from ingest import ImageIngestor
from embedding import SceneEmbedder
from clustering import SceneClusterer
from exposure_sorting import ExposureSorter
from validation import ClusterValidator
from output import ResultsExporter
from utils import PathLike, logger


class HDRGroupingPipeline:
    """Coordinates the HDR image grouping process."""

    def __init__(
            self,
            output_dir: Optional[PathLike] = None,
            config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the HDR grouping pipeline.

        Args:
            output_dir: Directory for output files
            config: Configuration parameters for pipeline components
        """
        self.logger = logging.getLogger('hdr_grouping.pipeline')

        # Set default output directory if not provided
        if output_dir is None:
            output_dir = Path.cwd() / "hdr_grouping_output"
        self.output_dir = Path(output_dir)

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize configuration with defaults if not provided
        if config is None:
            config = self._get_default_config()
        self.config = config

        # Initialize pipeline components
        self._init_components()

        # Statistics storage
        self.stats = {
            'total_images': 0,
            'ingest_time': 0,
            'embedding_time': 0,
            'clustering_time': 0,
            'validation_time': 0,
            'export_time': 0,
            'total_time': 0
        }

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for pipeline components."""
        return {
            'ingest': {
                'target_size': (512, 512),
                'max_workers': 4
            },
            'embedding': {
                'model_name': "facebook/dinov2-small",
                'use_gpu': True,
                'use_traditional_features': True,
                'dimensionality_reduction': "pca",
                'target_dims': 128,
                'cache_dir': None,
                'max_workers': 2
            },
            'clustering': {
                'min_cluster_size': 2,
                'min_samples': 1,
                'cluster_selection_epsilon': 0.1,
                'metric': 'cosine',
                'cluster_selection_method': 'leaf'
            },
            'exposure_sorting': {
                'ssim_threshold': 0.7,
                'hash_threshold': 10,
                'min_ev_difference': 0.3
            },
            'validation': {
                'similarity_threshold': 0.7,
                'min_hdr_score': 0.4,
                'geometry_threshold': 0.6,
                'min_cluster_size': 2,
                'max_cluster_size': 20
            },
            'output': {
                'create_thumbnails': True,
                'thumbnail_size': (256, 256),
                'export_json': True,
                'copy_images': True
            }
        }

    def _init_components(self) -> None:
        """Initialize all pipeline components with config."""
        self.ingestor = ImageIngestor(**self.config['ingest'])
        self.embedder = SceneEmbedder(**self.config['embedding'])
        self.clusterer = SceneClusterer(**self.config['clustering'])
        self.exposure_sorter = ExposureSorter(**self.config['exposure_sorting'])
        self.validator = ClusterValidator(**self.config['validation'])
        self.exporter = ResultsExporter(
            output_base_dir=self.output_dir,
            **self.config['output']
        )

    def process_directory(self, directory_path: PathLike) -> Dict[str, Any]:
        """
        Process a directory of images for HDR grouping.

        Args:
            directory_path: Path to directory containing images

        Returns:
            Dictionary with processing results and statistics
        """
        start_time = time.time()

        # Step 1: Ingest & Preprocess
        self.logger.info("Step 1: Ingesting and preprocessing images")
        ingest_start = time.time()
        images_data = self.ingestor.ingest_directory(directory_path)
        ingest_time = time.time() - ingest_start
        self.stats['ingest_time'] = ingest_time
        self.stats['total_images'] = len(images_data)

        if not images_data:
            self.logger.error(f"No valid images found in {directory_path}")
            self.stats['total_time'] = time.time() - start_time
            return {
                'success': False,
                'error': "No valid images found",
                'stats': self.stats
            }

        # Create a dictionary for faster lookup by ID
        images_dict = {img['id']: img for img in images_data}

        # Step 2: Robust Scene Embedding
        self.logger.info("Step 2: Creating scene embeddings")
        embedding_start = time.time()
        embeddings = self.embedder.create_embeddings(images_data)
        embedding_time = time.time() - embedding_start
        self.stats['embedding_time'] = embedding_time

        if not embeddings:
            self.logger.error("Failed to create embeddings")
            self.stats['total_time'] = time.time() - start_time
            return {
                'success': False,
                'error': "Failed to create embeddings",
                'stats': self.stats
            }

        # Step 3: Scene-Level Clustering
        self.logger.info("Step 3: Clustering images by scene")
        clustering_start = time.time()
        clusters = self.clusterer.cluster_embeddings(embeddings, images_data)

        # Try to assign outliers to existing clusters
        clusters = self.clusterer.assign_outliers(clusters, embeddings)

        # Calculate cluster statistics
        cluster_stats = self.clusterer.get_cluster_statistics(clusters, embeddings)
        clustering_time = time.time() - clustering_start
        self.stats['clustering_time'] = clustering_time

        # Check for empty clustering results
        if not clusters or (len(clusters) == 1 and -1 in clusters):
            self.logger.warning("No meaningful clusters found")

        # Step 4 & 5: Exposure Sorting and Validation
        self.logger.info("Step 4 & 5: Sorting by exposure and validating clusters")
        validation_start = time.time()

        # Process each cluster
        processed_clusters = {}
        unassigned_images = []

        for cluster_id, cluster_items in clusters.items():
            # Skip noise cluster
            if cluster_id == -1:
                unassigned_images.extend([img_id for img_id, _ in cluster_items])
                continue

            # Process exposure sequence
            exposure_info = self.exposure_sorter.process_cluster(cluster_items, images_dict)

            # Validate cluster
            validation_result = self.validator.validate_cluster(
                cluster_items,
                images_dict,
                embeddings,
                exposure_info['hdr_score']
            )

            # If cluster is invalid but large, try splitting
            if not validation_result['is_valid'] and len(cluster_items) > self.config['validation'][
                'min_cluster_size'] * 2:
                subclusters = self.validator.suggest_split(cluster_items, images_dict, embeddings)

                if len(subclusters) > 1:
                    # Process each subcluster
                    for i, subcluster in enumerate(subclusters):
                        sub_id = f"{cluster_id}.{i}"
                        sub_exposure_info = self.exposure_sorter.process_cluster(subcluster, images_dict)
                        sub_validation = self.validator.validate_cluster(
                            subcluster,
                            images_dict,
                            embeddings,
                            sub_exposure_info['hdr_score']
                        )

                        processed_clusters[sub_id] = {
                            'items': subcluster,
                            'exposure_info': sub_exposure_info,
                            'validation': sub_validation,
                            'parent_cluster': cluster_id
                        }
                else:
                    # Keep original cluster
                    processed_clusters[cluster_id] = {
                        'items': cluster_items,
                        'exposure_info': exposure_info,
                        'validation': validation_result
                    }
            else:
                # Store processed cluster
                processed_clusters[cluster_id] = {
                    'items': cluster_items,
                    'exposure_info': exposure_info,
                    'validation': validation_result
                }

        # Check for potential merges
        valid_clusters = {
            cid: cdata for cid, cdata in processed_clusters.items()
            if cdata['validation']['is_valid']
        }

        merge_candidates = self.validator.suggest_merge(
            {cid: cdata['items'] for cid, cdata in valid_clusters.items()},
            embeddings
        )

        # Apply merges if any
        if merge_candidates:
            merged_clusters = {}
            merged_ids = set()

            for primary_id, secondary_ids in merge_candidates.items():
                if primary_id in merged_ids:
                    continue

                # Mark primary and secondary as merged
                merged_ids.add(primary_id)
                merged_ids.update(secondary_ids)

                # Combine items
                merged_items = valid_clusters[primary_id]['items'].copy()
                for sec_id in secondary_ids:
                    merged_items.extend(valid_clusters[sec_id]['items'])

                # Process merged cluster
                merged_exposure_info = self.exposure_sorter.process_cluster(merged_items, images_dict)
                merged_validation = self.validator.validate_cluster(
                    merged_items,
                    images_dict,
                    embeddings,
                    merged_exposure_info['hdr_score']
                )

                # Store if valid
                if merged_validation['is_valid']:
                    merge_id = f"M{primary_id}"
                    merged_clusters[merge_id] = {
                        'items': merged_items,
                        'exposure_info': merged_exposure_info,
                        'validation': merged_validation,
                        'merged_from': [primary_id] + secondary_ids
                    }

            # Update processed clusters
            for cid, cdata in merged_clusters.items():
                processed_clusters[cid] = cdata

        validation_time = time.time() - validation_start
        self.stats['validation_time'] = validation_time

        # Step 6: Output & Logging
        self.logger.info("Step 6: Exporting results")
        export_start = time.time()

        # Get valid clusters
        valid_clusters = {
            cid: cdata for cid, cdata in processed_clusters.items()
            if cdata['validation']['is_valid']
        }

        # Export valid clusters
        export_stats = self.exporter.export_clusters(valid_clusters, images_dict)

        # Export unassigned images
        # Add images from invalid clusters to unassigned list
        for cid, cdata in processed_clusters.items():
            if not cdata['validation']['is_valid']:
                unassigned_images.extend([img_id for img_id, _ in cdata['items']])

        unassigned_count = self.exporter.export_unassigned_images(unassigned_images, images_dict)

        # Generate report
        self.exporter.generate_report(export_stats, unassigned_count, self.stats)

        export_time = time.time() - export_start
        self.stats['export_time'] = export_time

        # Calculate total time
        total_time = time.time() - start_time
        self.stats['total_time'] = total_time

        # Prepare final results
        results = {
            'success': True,
            'output_dir': str(self.output_dir),
            'total_images': len(images_data),
            'total_clusters': len(processed_clusters),
            'valid_clusters': len(valid_clusters),
            'unassigned_images': unassigned_count,
            'stats': self.stats
        }

        return results


def main():
    """Command line interface for HDR grouping pipeline."""
    import argparse
    import sys

    # Configure argument parser
    parser = argparse.ArgumentParser(description='HDR Image Grouping Pipeline')
    parser.add_argument('input_dir', type=str, help='Directory containing HDR images')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                        help='Directory for output files (default: ./hdr_grouping_output)')
    parser.add_argument('--config', '-c', type=str, default=None,
                        help='Path to JSON configuration file')
    parser.add_argument('--log-level', '-l', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level (default: INFO)')

    # Parse arguments
    args = parser.parse_args()

    # Configure logging
    log_level = getattr(logging, args.log_level)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load configuration if provided
    config = None
    if args.config:
        import json
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"Error loading configuration file: {str(e)}")
            sys.exit(1)

    # Create pipeline
    pipeline = HDRGroupingPipeline(output_dir=args.output_dir, config=config)

    # Process directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        logger.error(f"Input directory does not exist or is not a directory: {args.input_dir}")
        sys.exit(1)

    logger.info(f"Processing directory: {input_dir}")
    results = pipeline.process_directory(input_dir)

    if not results['success']:
        logger.error(f"Pipeline failed: {results.get('error', 'Unknown error')}")
        sys.exit(1)

    logger.info(f"Pipeline completed successfully in {results['stats']['total_time']:.2f}s")
    logger.info(f"Output directory: {results['output_dir']}")
    logger.info(f"Processed {results['total_images']} images into {results['valid_clusters']} valid clusters")


if __name__ == "__main__":
    main()