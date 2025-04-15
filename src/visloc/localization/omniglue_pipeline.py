import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from tqdm import tqdm
# matplotlib cm import'u gereksiz, og_utils kendi renklerini kullanıyor
# import matplotlib.cm as cm

# OmniGlue components
from omniglue.omniglue_extract import OmniGlue as OGMatcher
from omniglue import utils as og_utils

# VisLoc components
from visloc.localization.base import BasePipeline, PipelineConfig
from visloc.localization.drone_streamer import DroneImageStreamer
from visloc.localization.map_reader import BaseMapReader, SatelliteMapReader, GeoSatelliteImage
from visloc.localization.preprocessing import QueryProcessor
from visloc.tms.data_structures import DroneImage
from visloc.tms.geo import haversine_distance
from visloc.tms.schemas import GpsCoordinate

class OmniGluePipeline(BasePipeline):
    """
    Class to run the localization pipeline using OmniGlue.

    Handles matching (OmniGlue), pose estimation, and result visualization/logging.
    Requires COLOR images.
    """

    def __init__(
        self,
        map_reader: BaseMapReader,
        drone_streamer: DroneImageStreamer,
        config: PipelineConfig,
        query_processor: QueryProcessor,
        logger: logging.Logger,
        og_export_path: str,
        sp_export_path: str,
        dino_export_path: str,
        match_threshold: float = 0.02,
    ) -> None:
        """Initializes the OmniGlue Pipeline."""
        super().__init__(
            map_reader=map_reader,
            drone_streamer=drone_streamer,
            detector=None, matcher=None, # Not used directly
            config=config,
            query_processor=query_processor,
            logger=logger,
        )
        self.logger.info("[OmniGlue] Initializing OmniGlue Matcher...")
        self.og_matcher = None # Start as None
        try:
             if not Path(og_export_path).exists(): raise FileNotFoundError(f"OmniGlue TF model not found: {og_export_path}")
             # SP and DINO paths are needed by OGMatcher init
             self.og_matcher = OGMatcher(
                 og_export=og_export_path,
                 sp_export=sp_export_path,
                 dino_export=dino_export_path,
             )
             self.match_threshold = match_threshold
             self.logger.info("[OmniGlue] OmniGlue Matcher initialized.")
        except Exception as e:
             self.logger.error(f"[OmniGlue] Failed to initialize OmniGlue Matcher: {e}", exc_info=True)
             # self.og_matcher remains None

    def _convert_to_rgb(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Ensure image is in RGB format (H, W, 3) for OmniGlue."""
        if image is None: return None
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 3:
             # Assume BGR if read by OpenCV
             return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        self.logger.warning(f"[OmniGlue] Unexpected image shape {image.shape} in _convert_to_rgb.")
        return None

    def _save_match_summary(
        self,
        drone_name: str,
        matched_map_name: Optional[str],
        num_raw_matches: int,
        num_filtered_matches: int,
        num_inliers: int,
        predicted_coords: Optional[GpsCoordinate],
        gt_coords: Optional[GpsCoordinate],
        distance_m: Optional[float],
        output_txt_path: Path
    ) -> None:
        """Saves the OmniGlue match summary to a text file."""
        num_outliers = 0
        if matched_map_name and num_inliers >= 0:
            num_outliers = num_raw_matches - num_inliers
        elif matched_map_name is None and num_inliers <= 0:
             num_inliers = 0
             num_outliers = num_raw_matches

        try:
            with open(output_txt_path, 'w') as f:
                f.write(f"Matcher: OmniGlue\n")
                f.write(f"Drone Image: {drone_name}\n")
                if matched_map_name:
                    f.write(f"Matched Map Image: {matched_map_name}\n")
                    f.write(f"Raw Matches (Pre-RANSAC): {num_raw_matches}\n")
                    f.write(f"Filtered Matches (Post-RANSAC): {num_inliers}\n")
                    f.write(f"Outliers: {num_outliers}\n")
                    f.write(f"Predicted Coordinates: {predicted_coords}\n")
                    f.write(f"Ground Truth Coordinates: {gt_coords if gt_coords else 'N/A'}\n")
                    f.write(f"Distance (m): {distance_m:.2f}" if distance_m is not None else "Distance (m): N/A\n")
                else:
                    f.write("Result: No Match Found\n")
                    f.write(f"Total Raw Matches Tried (Sum over maps): {num_raw_matches}\n")
                    f.write(f"Total Filtered Matches Tried (Sum over maps): {num_filtered_matches}\n")
        except Exception as e:
            self.logger.error(f"[OmniGlue] Failed to write summary file {output_txt_path}: {e}", exc_info=True)

    def run_on_image(
        self, drone_image: DroneImage, output_path: Union[str, Path] = None
    ) -> Dict[str, Any]:
        """Runs the OmniGlue pipeline on a single drone image."""
        self.logger.info(f"[OmniGlue] Processing image {drone_image.name}")

        if not self.og_matcher:
             self.logger.error("[OmniGlue] Matcher not initialized. Skipping.")
             # Save summary indicating initialization failure
             if output_path:
                  txt_path = Path(output_path) / f"{drone_image.name}_omniglue_summary.txt"
                  Path(output_path).mkdir(parents=True, exist_ok=True)
                  self._save_match_summary(drone_image.name, None, 0, 0, -1, None, None, None, txt_path)
             return {"is_match": False, "distance": None, "error": "OmniGlue Matcher not initialized", "num_inliers": 0}

        # Initialize variables
        best_dst = None; matched_image_obj = None; center = None; distance = None
        best_mkpts0 = None; best_mkpts1 = None; num_inliers_best = -1
        is_match = False; predicted_coordinates = None
        num_raw_matches_best = 0; num_filtered_matches_best = 0
        total_raw_matches_tried = 0; total_filtered_matches_tried = 0
        error_message = None

        # --- Prepare Drone Image ---
        if drone_image.image is None:
            self.logger.error(f"[OmniGlue] Drone image {drone_image.name} not loaded.")
            error_message = "Drone image not loaded"
        else:
            drone_image_rgb = self._convert_to_rgb(drone_image.image)
            if drone_image_rgb is None:
                 self.logger.error(f"[OmniGlue] Could not convert drone image {drone_image.name} to RGB.")
                 error_message = "Drone image RGB conversion failed"

        # --- Get Ground Truth Coordinates ---
        gt_coordinates = None
        if drone_image.geo_point:
            try:
                gt_coordinates = GpsCoordinate(lat=drone_image.geo_point.latitude, long=drone_image.geo_point.longitude)
            except (AttributeError, TypeError) as e:
                self.logger.error(f"[OmniGlue] Error creating GpsCoordinate from geo_point for {drone_image.name}: {e}")

        # --- Matching Loop (Only if drone image is ready) ---
        if error_message is None:
            for idx in tqdm(
                range(len(self.map_reader)),
                desc=f"[OmniGlue] Matching {drone_image.name}",
                total=len(self.map_reader),
                leave=False
            ):
                if not isinstance(self.map_reader, SatelliteMapReader):
                     error_message = "Incorrect Map Reader Type for OmniGlue"; break # Exit loop

                satellite_image: GeoSatelliteImage = self.map_reader[idx]
                if satellite_image.image is None: continue
                satellite_image_rgb = self._convert_to_rgb(satellite_image.image)
                if satellite_image_rgb is None: continue

                try:
                    # 1. Find matches using OmniGlue
                    mkpts0_raw, mkpts1_raw, confidences_raw = self.og_matcher.FindMatches(drone_image_rgb, satellite_image_rgb)
                    num_raw_matches_current = len(mkpts0_raw)
                    total_raw_matches_tried += num_raw_matches_current

                    # 2. Filter matches by confidence
                    keep_idx = confidences_raw > self.match_threshold
                    mkpts0_f = mkpts0_raw[keep_idx]
                    mkpts1_f = mkpts1_raw[keep_idx]
                    num_filtered_matches_current = len(mkpts0_f)
                    total_filtered_matches_tried += num_filtered_matches_current

                    if num_filtered_matches_current < 4: continue

                    # 3. Estimate Homography
                    ret, num_inliers_current, dst = self.estimate_and_apply_geometric_transform(mkpts0_f, mkpts1_f, drone_image_rgb.shape[:2])
                    num_inliers_current = int(num_inliers_current)

                    if ret and num_inliers_current > num_inliers_best:
                         denormalized_center = self.compute_center(dst)
                         sat_img_shape = satellite_image_rgb.shape
                         center_norm = self.normalize_center(denormalized_center, sat_img_shape[:2])

                         if not (0 <= center_norm[0] <= 1 and 0 <= center_norm[1] <= 1):
                              self.logger.warning(f"[OmniGlue] Center {center_norm} out of bounds for {satellite_image.name}. Discarding match.")
                              continue

                         num_inliers_best = num_inliers_current
                         num_raw_matches_best = num_raw_matches_current
                         num_filtered_matches_best = num_filtered_matches_current
                         best_dst = dst
                         matched_image_obj = satellite_image
                         center = center_norm
                         best_mkpts0 = mkpts0_f
                         best_mkpts1 = mkpts1_f

                except cv2.error as e:
                    self.logger.warning(f"[OmniGlue] cv2 error during RANSAC for {satellite_image.name}: {e}")
                    continue
                except Exception as e:
                    self.logger.error(f"[OmniGlue] Error during matching/processing {drone_image.name} vs {satellite_image.name}: {e}", exc_info=True)
                    # Set error message if a critical error happened during matching?
                    if error_message is None: error_message = f"Matching error: {e}"
                    continue

        # --- Post-Loop Processing ---
        final_pred_coords = None
        final_distance_m = None

        if best_dst is not None and matched_image_obj is not None:
            try:
                 if not (matched_image_obj.top_left and matched_image_obj.bottom_right):
                      raise ValueError(f"Matched map image {matched_image_obj.name} missing GPS bounds.")
                 final_pred_coords = self.compute_geo_pose(matched_image_obj, center)
                 if gt_coordinates and final_pred_coords:
                     final_distance_m = haversine_distance(gt_coordinates, final_pred_coords) * 1000
                 self.logger.info(f"[OmniGlue] Match Found: {drone_image.name} -> {matched_image_obj.name} ({num_inliers_best} inliers from {num_filtered_matches_best} filtered | {num_raw_matches_best} raw)")
                 self.logger.info(f"[OmniGlue] Predicted: {final_pred_coords}, GT: {gt_coordinates if gt_coordinates else 'N/A'}")
                 if final_distance_m is not None:
                     self.logger.info(f"[OmniGlue] Haversine distance (m): {final_distance_m:.2f}")
                 is_match = True
            except Exception as e:
                self.logger.error(f"[OmniGlue] Error computing final pose/distance for {drone_image.name}: {e}", exc_info=True)
                if error_message is None: error_message = f"Pose/distance calculation failed: {e}"
                is_match = False

        # --- Create Outputs (Visualization & TXT) ---
        if output_path:
            output_dir = Path(output_path)
            txt_path = output_dir / f"{drone_image.name}_omniglue_summary.txt"
            viz_path = output_dir / f"{drone_image.name}_omniglue_viz.png" # Use PNG
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save Summary Text (always)
            self._save_match_summary(
                drone_name=drone_image.name,
                matched_map_name=matched_image_obj.name if is_match else None,
                num_raw_matches=num_raw_matches_best if is_match else total_raw_matches_tried,
                num_filtered_matches=num_filtered_matches_best if is_match else total_filtered_matches_tried,
                num_inliers=num_inliers_best if is_match else -1,
                predicted_coords=final_pred_coords if is_match else None,
                gt_coords=gt_coordinates,
                distance_m=final_distance_m if is_match else None,
                output_txt_path=txt_path
            )

            # Create and Save Visualization (only if match found)
            if is_match:
                try:
                    viz_drone_img = drone_image_rgb.copy()
                    viz_sat_img = self._convert_to_rgb(matched_image_obj.image).copy()
                    denorm_center_viz = self.compute_center(best_dst)
                    # Çizimleri temiz uydu görüntüsü üzerine yapalım
                    viz_sat_img_clean = viz_sat_img.copy()
                    viz_sat_img_drawn = self.draw_transform_polygon_on_image(viz_sat_img_clean, best_dst)
                    viz_sat_img_drawn = self.draw_center(viz_sat_img_drawn, denorm_center_viz)

                    # Use OmniGlue's visualizer with the *inlier* keypoints and DRAWN satellite image
                    raw_out_img = og_utils.visualize_matches(
                        image0=viz_drone_img,
                        image1=viz_sat_img_drawn, # Use the image with drawings
                        kp0=best_mkpts0,
                        kp1=best_mkpts1,
                        match_matrix=np.eye(len(best_mkpts0)),
                        show_keypoints=True, title="", line_width=1, circle_radius=3
                    )

                    # Add only Matcher Name Text
                    matcher_name = "OmniGlue"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.5
                    font_thickness = 3
                    text_color = (0, 255, 255)  # Cyan
                    outline_color = (0, 0, 0)
                    outline_thickness = font_thickness + 2
                    text_size, _ = cv2.getTextSize(matcher_name, font, font_scale, font_thickness)
                    text_x = raw_out_img.shape[1] - text_size[0] - 20
                    text_y = text_size[1] + 20
                    # Draw outline for matcher name
                    cv2.putText(raw_out_img, matcher_name, (text_x, text_y), font, font_scale, outline_color, outline_thickness, cv2.LINE_AA)
                    cv2.putText(raw_out_img, matcher_name, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

                    # Ensure uint8 and contiguous before saving
                    if raw_out_img.dtype != np.uint8: raw_out_img = raw_out_img.astype(np.uint8)
                    if not raw_out_img.flags['C_CONTIGUOUS']: raw_out_img = np.ascontiguousarray(raw_out_img)

                    self.save_viz(raw_out_img, viz_path) # Save the final image
                    self.logger.info(f"[OmniGlue] Saved visualization: {viz_path}")

                except Exception as e_viz:
                    self.logger.error(f"[OmniGlue] Error creating/saving visualization for {drone_image.name}: {e_viz}", exc_info=True)
                    if error_message is None: error_message = f"Visualization failed: {e_viz}"

        elif error_message is None and not is_match: # Explicitly log no match if no other error occurred
             self.logger.warning(f"[OmniGlue] No suitable match found for {drone_image.name}")


        # --- Return Results ---
        return {
            "is_match": is_match,
            "gt_coordinate": gt_coordinates,
            "predicted_coordinate": final_pred_coords,
            "center": center,
            "best_dst": best_dst.tolist() if isinstance(best_dst, np.ndarray) else None,
            "num_inliers": num_inliers_best if num_inliers_best > -1 else 0,
            "matched_image": matched_image_obj.name if matched_image_obj else None,
            "distance": final_distance_m,
            "error": error_message
        }

    def run(self, output_path: Union[str, Path] = None) -> List[Dict[str, Any]]:
        """Runs the OmniGlue pipeline on all drone images."""
        if not self.og_matcher:
             self.logger.error("[OmniGlue] Matcher not initialized. Cannot run pipeline.")
             return [{"is_match": False, "distance": None, "error": "OmniGlue Matcher not initialized", "num_inliers": 0}] * len(self.drone_streamer or [])

        self.logger.info(f"[OmniGlue] Running pipeline on {len(self.drone_streamer)} images")
        preds = []
        num_matches = 0

        viz_output_path = None
        if output_path:
             viz_output_path = Path(output_path)
             viz_output_path.mkdir(parents=True, exist_ok=True)

        if hasattr(self.drone_streamer, '__iter__'):
             self.drone_streamer.__iter__() # Reset iterator

        for i, drone_image_orig in enumerate(self.drone_streamer):
            self.logger.info(f"[OmniGlue] Processing image {i+1}/{len(self.drone_streamer)}: {drone_image_orig.name}")
            processed_query = self.query_processor(drone_image_orig)

            try:
                pred = self.run_on_image(processed_query, viz_output_path)
            except Exception as e:
                self.logger.error(f"[OmniGlue] Unhandled error processing {drone_image_orig.name}: {e}", exc_info=True)
                pred = {"is_match": False, "distance": None, "error": f"Unhandled error: {e}", "num_inliers": 0}

            if isinstance(pred.get("best_dst"), np.ndarray):
                 pred["best_dst"] = pred["best_dst"].tolist()
            if isinstance(pred.get("gt_coordinate"), GpsCoordinate):
                 pred["gt_coordinate"] = {"lat": pred["gt_coordinate"].lat, "long": pred["gt_coordinate"].long}
            if isinstance(pred.get("predicted_coordinate"), GpsCoordinate):
                 pred["predicted_coordinate"] = {"lat": pred["predicted_coordinate"].lat, "long": pred["predicted_coordinate"].long}

            preds.append(pred)
            if pred.get("is_match", False):
                num_matches += 1
                if pred.get("distance") is not None and pred["distance"] > 50:
                    self.logger.warning(f"[OmniGlue] Large distance: {pred['distance']:.2f}m for {drone_image_orig.name}")

        self.logger.info(f"[OmniGlue] Pipeline finished. Found {num_matches} matches for {len(self.drone_streamer)} images.")
        return preds

    def compute_geo_pose(
        self, satellite_image: GeoSatelliteImage, matching_center: Tuple[float, float]
    ) -> Optional[GpsCoordinate]:
        """Compute the GPS coordinates from a satellite image and normalized center."""
        # Bu metod BasePipeline'daki ile aynı olabilir veya buraya kopyalanabilir.
        # Önceki OmniGlue versiyonundaki gibi bırakıyorum, gerekli kontrolleri içeriyor.
        if not (satellite_image and satellite_image.top_left and satellite_image.bottom_right):
             self.logger.error(f"[OmniGlue] Cannot compute geo pose for {getattr(satellite_image, 'name', 'unknown')}, missing GPS bounds.")
             return None
        if not (matching_center and len(matching_center) == 2 and
                all(isinstance(c, (float, int)) for c in matching_center)):
             self.logger.error(f"[OmniGlue] Invalid matching_center format: {matching_center}")
             return None
        # ... (kalan hesaplama ve kontroller aynı) ...
        try:
            center_x, center_y = float(matching_center[0]), float(matching_center[1])
            center_x = max(0.0, min(1.0, center_x)); center_y = max(0.0, min(1.0, center_y)) # Clamp
            lat_span = abs(satellite_image.bottom_right.lat - satellite_image.top_left.lat)
            lon_span = abs(satellite_image.bottom_right.long - satellite_image.top_left.long)
            pred_lat = satellite_image.top_left.lat - (center_y * lat_span)
            pred_long = satellite_image.top_left.long + (center_x * lon_span)
            return GpsCoordinate(lat=pred_lat, long=pred_long)
        except Exception as e:
            self.logger.error(f"[OmniGlue] Error calculating geo pose for {satellite_image.name}: {e}", exc_info=True)
            return None