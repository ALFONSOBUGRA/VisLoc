#!/usr/bin/env python
"""Main script for running visual localization pipelines sequentially."""

import argparse
import logging
import cv2 # cv2 import'u ekleyelim (ihtiyaç olabilir)
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, Optional, List, Tuple
import gc # Garbage collector'ı import edelim

# PyTorch import'u (GPU varsa cache temizlemek için)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# TensorFlow import'u (Keras backend temizlemek için)
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


from visloc.config import ConfigHandler
# VisLoc Pipeline components
from visloc.keypoint_pipeline.detection_and_description import SuperPointAlgorithm
from visloc.keypoint_pipeline.matcher import SuperGlueMatcher
from visloc.keypoint_pipeline.typing import SuperGlueConfig, SuperPointConfig
from visloc.localization.superglue_pipeline import Pipeline as SuperGluePipeline
# OmniGlue Pipeline components
from visloc.localization.omniglue_pipeline import OmniGluePipeline
# Common components
from visloc.localization.drone_streamer import DroneImageStreamer
from visloc.localization.map_reader import SatelliteMapReader
from visloc.localization.superglue_pipeline import PipelineConfig # Base config reused
from visloc.localization.preprocessing import QueryProcessor
from visloc.tms.data_structures import CameraModel
from visloc.utils.constants import (
    DEFAULT_DEVICE,
    DEFAULT_FOCAL_LENGTH, DEFAULT_HFOV_DEG, DEFAULT_KEYPOINT_THRESHOLD,
    DEFAULT_LOGGING_FORMAT, DEFAULT_LOGGING_DATE_FORMAT, DEFAULT_LOGGING_LEVEL,
    DEFAULT_MAP_PATH, DEFAULT_MAX_KEYPOINTS, DEFAULT_NMS_RADIUS,
    DEFAULT_OUTPUT_PATH, DEFAULT_QUERY_PATH, DEFAULT_RESIZE_SIZE,
    DEFAULT_RESOLUTION_HEIGHT, DEFAULT_RESOLUTION_WIDTH, DEFAULT_SINKHORN_ITERATIONS,
    DEFAULT_MATCH_THRESHOLD, DEFAULT_WEIGHTS, DEFAULT_OG_EXPORT_PATH,
    DEFAULT_SP_EXPORT_PATH, DEFAULT_DINO_EXPORT_PATH, DEFAULT_OG_MATCH_THRESHOLD
)

class VisualLocalizationRunner:
    """
    Manages and runs visual localization pipelines sequentially to save memory.
    Initializes common components first, then loads/unloads pipeline-specific
    models during the run phase.
    """

    def __init__(self, config_path: Optional[str] = None, output_path: Optional[str] = None, device_preference: Optional[str] = None) -> None:
        """
        Initializes the runner, loading common components only.

        Args:
            config_path: Path to the configuration file.
            output_path: Path to the base output directory.
            device_preference: Preferred device for Torch models (cuda or cpu).
        """
        self.config_handler = ConfigHandler(config_path)
        self.output_path_base = Path(output_path or self.config_handler.get("pipeline.output_path", DEFAULT_OUTPUT_PATH))
        # Determine device, default to CPU if Torch/CUDA not available or specified
        if TORCH_AVAILABLE and torch.cuda.is_available() and device_preference != 'cpu':
             default_torch_device = 'cuda'
        else:
             default_torch_device = 'cpu'
        self.device = device_preference or self.config_handler.get("superpoint.device", default_torch_device)

        self.logger = self._setup_logger()
        self.logger.info(f"Using device: {self.device} for PyTorch components.")

        # --- Initialize Common Components Only ---
        self.logger.info("Initializing common components (MapReader, Streamer, Processor)...")
        self.map_reader = self._init_map_reader()
        self.streamer = self._init_streamer()
        self.query_processor = self._init_query_processor()
        self.pipeline_config = PipelineConfig()
        self.logger.info("Common components initialized.")

        # Flag to track if map images have been described by SuperPoint
        self._map_described_sp = False

    def _setup_logger(self) -> logging.Logger:
        """Sets up logging, including TensorFlow log level."""
        log_format = self.config_handler.get("logging.format", DEFAULT_LOGGING_FORMAT)
        log_level_str = self.config_handler.get("logging.level", DEFAULT_LOGGING_LEVEL).upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        log_datefmt = self.config_handler.get("logging.datefmt", DEFAULT_LOGGING_DATE_FORMAT)

        logging.basicConfig(format=log_format, level=log_level, datefmt=log_datefmt, force=True)

        # Configure TF logging level
        tf_log_level_env = '3' # ERROR
        if log_level <= logging.DEBUG: tf_log_level_env = '0'
        elif log_level <= logging.INFO: tf_log_level_env = '1'
        elif log_level <= logging.WARNING: tf_log_level_env = '2'
        import os
        # Set TF log level via environment variable BEFORE importing TF (if possible)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = tf_log_level_env
        # Suppress TF warnings specifically if TF is available
        if TF_AVAILABLE:
             tf.get_logger().setLevel(logging.ERROR) # Or use higher level like WARNING/ERROR

        # Also set Python's TF logger level (less effective for C++ logs)
        logging.getLogger('tensorflow').setLevel(logging.ERROR) # Or map log_level

        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(log_level)
        return logger

    # --- Common Initializers (No changes needed here) ---
    def _init_map_reader(self) -> SatelliteMapReader:
        self.logger.debug("Initializing SatelliteMapReader...")
        resize_conf = self.config_handler.get("map_reader.resize_size", DEFAULT_RESIZE_SIZE)
        resize_tuple = tuple(resize_conf) if isinstance(resize_conf, list) else resize_conf
        map_reader = SatelliteMapReader(
            db_path=Path(self.config_handler.get("map_reader.db_path", DEFAULT_MAP_PATH)),
            cv2_read_mode=cv2.IMREAD_COLOR, # Always read color
            resize_size=resize_tuple,
            logger=logging.getLogger(f"{self.__class__.__name__}.SatelliteMapReader"),
        )
        map_reader.setup_db()
        map_reader.resize_db_images()
        self.logger.debug("SatelliteMapReader initialized.")
        return map_reader

    def _init_streamer(self) -> DroneImageStreamer:
        self.logger.debug("Initializing DroneImageStreamer...")
        streamer = DroneImageStreamer(
            image_folder=Path(self.config_handler.get("drone_streamer.image_folder", DEFAULT_QUERY_PATH)),
            has_gt=self.config_handler.get("drone_streamer.has_gt", True),
            cv2_read_mode=cv2.IMREAD_COLOR, # Always read color
            logger=logging.getLogger(f"{self.__class__.__name__}.DroneImageStreamer"),
        )
        self.logger.debug("DroneImageStreamer initialized.")
        return streamer

    def _init_query_processor(self) -> QueryProcessor:
        self.logger.debug("Initializing QueryProcessor...")
        camera_model = CameraModel(
            focal_length=self.config_handler.get("camera_model.focal_length", DEFAULT_FOCAL_LENGTH),
            resolution_height=self.config_handler.get("camera_model.resolution_height", DEFAULT_RESOLUTION_HEIGHT),
            resolution_width=self.config_handler.get("camera_model.resolution_width", DEFAULT_RESOLUTION_WIDTH),
            hfov_deg=self.config_handler.get("camera_model.hfov_deg", DEFAULT_HFOV_DEG),
        )
        size_conf = self.config_handler.get("query_processor.size", DEFAULT_RESIZE_SIZE)
        size_tuple = tuple(size_conf) if isinstance(size_conf, list) else size_conf
        query_processor = QueryProcessor(
            processings=self.config_handler.get("query_processor.processings", ["resize"]),
            camera_model=camera_model,
            satellite_resolution=self.config_handler.get("query_processor.satellite_resolution", None),
            size=size_tuple,
        )
        self.logger.debug("QueryProcessor initialized.")
        return query_processor

    # --- Pipeline Specific Loaders/Unloaders ---

    def _load_superglue_components(self) -> Tuple[Optional[SuperGluePipeline], Optional[SuperPointAlgorithm], Optional[SuperGlueMatcher]]:
        """Loads SuperPoint, SuperGlue, describes map if needed, and creates SuperGlue pipeline."""
        self.logger.info("Loading SuperGlue (SuperPoint/SuperGlue) components...")
        try:
            superpoint_config = SuperPointConfig(
                device=self.device,
                nms_radius=self.config_handler.get("superpoint.nms_radius", DEFAULT_NMS_RADIUS),
                keypoint_threshold=self.config_handler.get("superpoint.keypoint_threshold", DEFAULT_KEYPOINT_THRESHOLD),
                max_keypoints=self.config_handler.get("superpoint.max_keypoints", DEFAULT_MAX_KEYPOINTS),
            )
            sp_algo = SuperPointAlgorithm(superpoint_config)
            self.logger.info(f"SuperPoint model loaded on {self.device}.")

            superglue_config = SuperGlueConfig(
                device=self.device,
                weights=self.config_handler.get("superglue.weights", DEFAULT_WEIGHTS),
                sinkhorn_iterations=self.config_handler.get("superglue.sinkhorn_iterations", DEFAULT_SINKHORN_ITERATIONS),
                match_threshold=self.config_handler.get("superglue.match_threshold", DEFAULT_MATCH_THRESHOLD),
            )
            sg_matcher = SuperGlueMatcher(superglue_config)
            self.logger.info(f"SuperGlue model loaded on {self.device}.")

            # Describe map images ONLY if not already described
            if not self._map_described_sp:
                 self.logger.info("Describing map images with SuperPoint for SuperGlue...")
                 self.map_reader.describe_db_images(sp_algo)
                 self._map_described_sp = True # Set flag
                 self.logger.info("Map images described.")
            else:
                 self.logger.info("Map images already described by SuperPoint, skipping.")


            pipeline = SuperGluePipeline(
                 map_reader=self.map_reader,
                 drone_streamer=self.streamer,
                 detector=sp_algo,
                 matcher=sg_matcher,
                 query_processor=self.query_processor,
                 config=self.pipeline_config,
                 logger=logging.getLogger(f"{self.__class__.__name__}.SuperGluePipeline"),
            )
            self.logger.info("SuperGlue pipeline instance created.")
            return pipeline, sp_algo, sg_matcher

        except Exception as e:
            self.logger.error(f"Failed to load SuperGlue components: {e}", exc_info=True)
            return None, None, None

    def _unload_superglue_components(self, pipeline, sp_algo, sg_matcher):
        """Releases SuperGlue components from memory."""
        self.logger.info("Unloading SuperGlue components...")
        del pipeline, sp_algo, sg_matcher
        gc.collect() # Trigger garbage collection
        if self.device == 'cuda' and TORCH_AVAILABLE:
            torch.cuda.empty_cache()
            self.logger.info("PyTorch CUDA cache cleared.")
        self.logger.info("SuperGlue components unloaded.")


    def _load_omniglue_components(self) -> Optional[OmniGluePipeline]:
        """Loads OmniGlue matcher and creates OmniGlue pipeline."""
        self.logger.info("Loading OmniGlue components...")
        try:
             og_export = self.config_handler.get("omniglue.og_export_path", DEFAULT_OG_EXPORT_PATH)
             sp_export = self.config_handler.get("omniglue.sp_export_path", DEFAULT_SP_EXPORT_PATH)
             dino_export = self.config_handler.get("omniglue.dino_export_path", DEFAULT_DINO_EXPORT_PATH)
             og_threshold = self.config_handler.get("omniglue.match_threshold", DEFAULT_OG_MATCH_THRESHOLD)

             # Path checks (OmniGluePipeline init also does this, but good to check early)
             if not Path(og_export).exists(): raise FileNotFoundError(f"OmniGlue TF model path not found: {og_export}")
             # Warnings for SP/DINO handled inside OmniGluePipeline init

             pipeline = OmniGluePipeline(
                 map_reader=self.map_reader,
                 drone_streamer=self.streamer,
                 query_processor=self.query_processor,
                 config=self.pipeline_config,
                 logger=logging.getLogger(f"{self.__class__.__name__}.OmniGluePipeline"),
                 og_export_path=og_export,
                 sp_export_path=sp_export,
                 dino_export_path=dino_export,
                 match_threshold=og_threshold,
             )
             # Check if matcher loaded successfully inside the pipeline init
             if pipeline.og_matcher is None:
                 raise RuntimeError("OmniGluePipeline initialized, but og_matcher failed to load.")

             self.logger.info("OmniGlue pipeline instance created.")
             return pipeline

        except Exception as e:
            self.logger.error(f"Failed to load OmniGlue components: {e}", exc_info=True)
            return None

    def _unload_omniglue_components(self, pipeline):
        """Releases OmniGlue components from memory."""
        self.logger.info("Unloading OmniGlue components...")
        # Explicitly delete components held by the pipeline if possible
        if hasattr(pipeline, 'og_matcher'): delattr(pipeline, 'og_matcher')
        # Delete internal TF sessions if the wrapper allows? OmniGlue's doesn't seem to expose it directly.
        del pipeline
        gc.collect()
        # Clear Keras session if TF uses it (safer approach)
        if TF_AVAILABLE and hasattr(tf, 'keras') and hasattr(tf.keras, 'backend'):
             tf.keras.backend.clear_session()
             self.logger.info("TensorFlow Keras session cleared.")
        # Clear PyTorch cache again (for DINOv2 used internally by OmniGlue)
        if self.device == 'cuda' and TORCH_AVAILABLE:
            torch.cuda.empty_cache()
            self.logger.info("PyTorch CUDA cache cleared again.")
        self.logger.info("OmniGlue components unloaded.")


    def run(self) -> Dict[str, Any]:
        """
        Runs the localization pipelines sequentially.
        """
        self.output_path_base.mkdir(parents=True, exist_ok=True)
        superglue_output_path = self.output_path_base / "superglue"
        omniglue_output_path = self.output_path_base / "omniglue"
        # No need to create dirs here, pipelines handle it in run_on_image

        self.logger.info(f"Number of drone images: {len(self.streamer)}")
        self.logger.info(f"Number of map images: {len(self.map_reader)}")
        all_metrics = {}

        # --- Run SuperGlue (SuperPoint/SuperGlue) Pipeline ---
        superglue_pipeline, sp_algo, sg_matcher = self._load_superglue_components()
        if superglue_pipeline:
            self.logger.info("--- Starting SuperGlue Pipeline Run ---")
            try:
                superglue_preds = superglue_pipeline.run(output_path=superglue_output_path)
                superglue_metrics = superglue_pipeline.compute_metrics(superglue_preds)
                all_metrics["superglue"] = superglue_metrics
                pprint({"superglue_metrics": superglue_metrics})
            except Exception as e:
                self.logger.error(f"SuperGlue Pipeline run failed: {e}", exc_info=True)
                all_metrics["superglue"] = {"error": str(e)}
            finally:
                self._unload_superglue_components(superglue_pipeline, sp_algo, sg_matcher)
            self.logger.info("--- SuperGlue Pipeline Run Finished ---")
        else:
            self.logger.error("Skipping SuperGlue pipeline run due to loading failure.")
            all_metrics["superglue"] = {"error": "Pipeline loading failed"}


        # --- Run OmniGlue Pipeline ---
        omniglue_pipeline = self._load_omniglue_components()
        if omniglue_pipeline:
            self.logger.info("--- Starting OmniGlue Pipeline Run ---")
            try:
                # Reset streamer iterator
                if hasattr(self.streamer, '__iter__'): self.streamer.__iter__()
                omniglue_preds = omniglue_pipeline.run(output_path=omniglue_output_path)
                omniglue_metrics = omniglue_pipeline.compute_metrics(omniglue_preds)
                all_metrics["omniglue"] = omniglue_metrics
                pprint({"omniglue_metrics": omniglue_metrics})
            except Exception as e:
                self.logger.error(f"OmniGlue Pipeline run failed: {e}", exc_info=True)
                all_metrics["omniglue"] = {"error": str(e)}
            finally:
                self._unload_omniglue_components(omniglue_pipeline)
            self.logger.info("--- OmniGlue Pipeline Run Finished ---")
        else:
             self.logger.error("Skipping OmniGlue pipeline run due to loading failure.")
             all_metrics["omniglue"] = {"error": "Pipeline loading failed"}

        return all_metrics


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Visual Localization Pipelines (SuperGlue & OmniGlue)")
    parser.add_argument("--config", "-c", type=str, help="Path to the configuration file", default=None)
    parser.add_argument("--output", "-o", type=str, help="Path to the base output directory", default=None)
    parser.add_argument("--device", "-d", type=str, help="Preferred device for Torch models (cuda or cpu)", default=None)
    return parser.parse_args()


def main() -> None:
    """Main entry point for running the visual localization pipelines."""
    args = parse_args()

    # Optional: Set TF GPU memory growth globally here if preferred
    # import os
    # os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    runner = VisualLocalizationRunner(
        config_path=args.config,
        output_path=args.output,
        device_preference=args.device
    )

    metrics = runner.run()
    print("\n--- Combined Metrics ---")
    pprint(metrics)
    print("--- Execution Finished ---")


if __name__ == "__main__":
    main()