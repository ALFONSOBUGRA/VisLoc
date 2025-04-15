import logging
from pathlib import Path
from typing import Any, Dict, List, Union, Optional, Tuple

import cv2
import matplotlib.cm as cm
import numpy as np
from tqdm import tqdm

# SuperGlue ve SuperPoint bileşenleri
from superglue.utils import make_matching_plot_fast # Düzeltilmiş halini kullandığını varsayıyoruz
from visloc.keypoint_pipeline.base import CombinedKeyPointAlgorithm, KeyPointMatcher
from visloc.keypoint_pipeline.detection_and_description import SuperPointAlgorithm
from visloc.localization.base import BasePipeline, PipelineConfig
from visloc.localization.drone_streamer import DroneImageStreamer
from visloc.localization.map_reader import SatelliteMapReader, GeoSatelliteImage
from visloc.localization.preprocessing import QueryProcessor
from visloc.tms.data_structures import DroneImage
from visloc.tms.geo import haversine_distance
from visloc.tms.schemas import GpsCoordinate

class Pipeline(BasePipeline): # Orijinal SuperGlue pipeline'ı
    """
    Class to run the localization pipeline based on keypoint matching.
    (SuperPoint/SuperGlue version)

    Handles feature extraction (SuperPoint), matching (SuperGlue), pose estimation,
    and result visualization/logging.
    """

    def __init__(
        self,
        map_reader: SatelliteMapReader,
        drone_streamer: DroneImageStreamer,
        detector: CombinedKeyPointAlgorithm, # SuperPointAlgorithm örneği bekleniyor
        matcher: KeyPointMatcher,           # SuperGlueMatcher örneği bekleniyor
        config: PipelineConfig,
        query_processor: QueryProcessor,
        logger: logging.Logger,
    ) -> None:
        """Initializes the SuperGlue Pipeline."""
        super().__init__(
            map_reader=map_reader,
            drone_streamer=drone_streamer,
            detector=detector,
            matcher=matcher,
            config=config,
            query_processor=query_processor,
            logger=logger,
        )
        if not getattr(map_reader,'_is_described', False): # map_reader'da bu özellik varsa kontrol et
             logger.warning("[SuperGlue] Map reader images might not have been described with keypoints yet.")

    def _save_match_summary(
        self,
        drone_name: str,
        matched_map_name: Optional[str],
        num_potential_matches: int,
        num_inliers: int,
        predicted_coords: Optional[GpsCoordinate],
        gt_coords: Optional[GpsCoordinate],
        distance_m: Optional[float],
        output_txt_path: Path
    ) -> None:
        """Saves the SuperGlue match summary to a text file."""
        num_outliers = 0
        if matched_map_name and num_inliers >= 0:
            num_outliers = num_potential_matches - num_inliers
        elif matched_map_name is None and num_inliers <= 0: # Eşleşme yok veya RANSAC başarısız
             num_inliers = 0
             num_outliers = num_potential_matches

        try:
            with open(output_txt_path, 'w') as f:
                f.write(f"Matcher: SuperGlue\n")
                f.write(f"Drone Image: {drone_name}\n")
                if matched_map_name:
                    f.write(f"Matched Map Image: {matched_map_name}\n")
                    f.write(f"Raw Matches (Pre-RANSAC): {num_potential_matches}\n")
                    f.write(f"Filtered Matches (Post-RANSAC): {num_inliers}\n")
                    f.write(f"Outliers: {num_outliers}\n")
                    f.write(f"Predicted Coordinates: {predicted_coords}\n")
                    f.write(f"Ground Truth Coordinates: {gt_coords if gt_coords else 'N/A'}\n")
                    f.write(f"Distance (m): {distance_m:.2f}" if distance_m is not None else "Distance (m): N/A\n")
                else:
                    f.write("Result: No Match Found\n")
                    f.write(f"Total Raw Matches Tried (Sum over maps): {num_potential_matches}\n")
                    f.write(f"Total Filtered Matches Tried (Sum over maps): {num_inliers}\n")
        except Exception as e:
            self.logger.error(f"[SuperGlue] Failed to write summary file {output_txt_path}: {e}", exc_info=True)

    def run_on_image(
        self, drone_image: DroneImage, output_path: Union[str, Path] = None
    ) -> Dict[str, Any]:
        """Runs the [SuperGlue] pipeline on a single drone image."""
        self.logger.info(f"[SuperGlue] Processing image {drone_image.name}")

        # Initialize variables
        best_dst = None
        matched_image = None
        center = None
        distance = None
        matched_kpts0 = None
        matched_kpts1 = None
        matched_confidence = None
        num_inliers_best = -1
        is_match = False
        predicted_coordinates = None
        num_potential_matches_best = 0
        total_potential_matches_tried = 0
        error_message = None # Hata mesajını tutmak için

        # --- Ensure Image Loaded ---
        if drone_image.image is None:
             self.logger.error(f"[SuperGlue] Drone image {drone_image.name} not loaded.")
             error_message = "Drone image not loaded"
             # Fall through to save summary if output path is provided

        # --- Grayscale Conversion for SuperPoint ---
        image_for_detector = None
        if error_message is None: # Sadece görüntü yüklendiyse devam et
            image_for_detector = drone_image.image.copy()
            if isinstance(self.detector, SuperPointAlgorithm):
                if len(image_for_detector.shape) == 3 and image_for_detector.shape[2] == 3:
                    self.logger.debug(f"[SuperGlue] Converting drone image {drone_image.name} to grayscale for SuperPoint.")
                    image_for_detector = cv2.cvtColor(image_for_detector, cv2.COLOR_BGR2GRAY)
                elif len(image_for_detector.shape) != 2:
                     self.logger.error(f"[SuperGlue] Drone image {drone_image.name} has unexpected shape {image_for_detector.shape} for SuperPoint.")
                     error_message = "Invalid image shape for SuperPoint"
            # else: Handle other detectors if needed

        # --- Extract Drone Keypoints ---
        if error_message is None: # Sadece önceki adımlar başarılıysa devam et
            try:
                drone_image.key_points = self.detector.detect_and_describe_keypoints(image_for_detector)
                if drone_image.key_points is None or len(drone_image.key_points.keypoints) == 0:
                     raise ValueError("No keypoints detected in drone image.")
            except Exception as e:
                 self.logger.error(f"[SuperGlue] Error extracting drone features for {drone_image.name}: {e}", exc_info=True)
                 error_message = f"Drone feature extraction failed: {e}"

        # --- Get Ground Truth Coordinates ---
        gt_coordinates = None
        if drone_image.geo_point:
            try:
                gt_coordinates = GpsCoordinate(
                    lat=drone_image.geo_point.latitude,
                    long=drone_image.geo_point.longitude
                )
            except (AttributeError, TypeError) as e:
                self.logger.error(f"[SuperGlue] Error creating GpsCoordinate from geo_point for {drone_image.name}: {e}")

        # --- Matching Loop (Only if drone keypoints were extracted) ---
        if error_message is None:
            for idx in tqdm(
                range(len(self.map_reader)),
                desc=f"[SuperGlue] Matching {drone_image.name}",
                total=len(self.map_reader),
                leave=False
            ):
                satellite_image: GeoSatelliteImage = self.map_reader[idx]

                if satellite_image.key_points is None or len(satellite_image.key_points.keypoints) == 0:
                    continue

                try:
                    matches, confidence = self.matcher.match_keypoints(
                        drone_image.key_points, satellite_image.key_points
                    )
                except Exception as e:
                     self.logger.error(f"[SuperGlue] Error matching {drone_image.name} vs {satellite_image.name}: {e}", exc_info=True)
                     continue

                valid = matches > -1
                num_potential_matches_current = int(np.sum(valid))
                total_potential_matches_tried += num_potential_matches_current

                if num_potential_matches_current < 4:
                    continue

                mkpts0 = drone_image.key_points.keypoints[valid]
                mkpts1 = satellite_image.key_points.keypoints[matches[valid]]
                confidences_f = confidence[valid]

                try:
                     ret, num_inliers_current, dst = self.estimate_and_apply_geometric_transform(
                         mkpts0, mkpts1, image_for_detector.shape[:2]
                     )
                     num_inliers_current = int(num_inliers_current)

                     if ret and num_inliers_current > num_inliers_best:
                         denormalized_center = self.compute_center(dst)
                         sat_img_shape = satellite_image.image.shape
                         center_norm = self.normalize_center(denormalized_center, sat_img_shape[:2])

                         if not (0 <= center_norm[0] <= 1 and 0 <= center_norm[1] <= 1):
                              self.logger.warning(f"[SuperGlue] Center {center_norm} out of bounds for {satellite_image.name}. Discarding match.")
                              continue

                         num_inliers_best = num_inliers_current
                         num_potential_matches_best = num_potential_matches_current
                         best_dst = dst
                         matched_image = satellite_image
                         center = center_norm
                         matched_kpts0 = mkpts0
                         matched_kpts1 = mkpts1
                         matched_confidence = confidences_f

                except cv2.error as e:
                    self.logger.warning(f"[SuperGlue] cv2 error during RANSAC for {satellite_image.name}: {e}")
                    continue
                except Exception as e:
                    self.logger.error(f"[SuperGlue] Error during geometry/center processing for {satellite_image.name}: {e}", exc_info=True)
                    continue

        # --- Post-Loop Processing ---
        final_pred_coords = None
        final_distance_m = None

        if best_dst is not None and matched_image is not None:
            try:
                 # Ensure matched image has geo bounds before pose calculation
                 if not (matched_image.top_left and matched_image.bottom_right):
                      raise ValueError(f"Matched map image {matched_image.name} missing GPS bounds.")
                 final_pred_coords = self.compute_geo_pose(matched_image, center)
                 if gt_coordinates and final_pred_coords:
                     final_distance_m = haversine_distance(gt_coordinates, final_pred_coords) * 1000

                 self.logger.info(f"[SuperGlue] Match Found: {drone_image.name} -> {matched_image.name} ({num_inliers_best} inliers)")
                 self.logger.info(f"[SuperGlue] Predicted: {final_pred_coords}, GT: {gt_coordinates if gt_coordinates else 'N/A'}")
                 if final_distance_m is not None:
                     self.logger.info(f"[SuperGlue] Haversine distance (m): {final_distance_m:.2f}")
                 is_match = True
            except Exception as e:
                self.logger.error(f"[SuperGlue] Error computing final pose/distance for {drone_image.name}: {e}", exc_info=True)
                error_message = f"Pose/distance calculation failed: {e}"
                is_match = False # Eşleşme bulundu ama sonrası başarısız

        # --- Create Outputs (Visualization & TXT) ---
        if output_path:
            output_dir = Path(output_path)
            txt_path = output_dir / f"{drone_image.name}_superglue_summary.txt"
            viz_path = output_dir / f"{drone_image.name}_superglue_viz.png" # PNG kullanmaya devam edelim
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save Summary Text (always, whether match found or not)
            self._save_match_summary(
                drone_name=drone_image.name,
                matched_map_name=matched_image.name if is_match else None,
                num_potential_matches=num_potential_matches_best if is_match else total_potential_matches_tried,
                num_inliers=num_inliers_best if is_match else -1,
                predicted_coords=final_pred_coords if is_match else None,
                gt_coords=gt_coordinates, # Always include GT if available
                distance_m=final_distance_m if is_match else None,
                output_txt_path=txt_path
            )

            # Create and Save Visualization (only if match found)
            if is_match:
                try:
                    viz_drone_img = drone_image.image.copy()
                    viz_sat_img = matched_image.image.copy()
                    denorm_center_viz = self.compute_center(best_dst)
                    # Poligon/merkezi çizmeden ÖNCE renkli uydu görüntüsünü kopyala
                    viz_sat_img_clean = viz_sat_img.copy()
                    viz_sat_img_drawn = self.draw_transform_polygon_on_image(viz_sat_img_clean, best_dst)
                    viz_sat_img_drawn = self.draw_center(viz_sat_img_drawn, denorm_center_viz)

                    color_rgba = cm.jet(matched_confidence)
                    color_rgb_0_255 = (color_rgba[:, :3] * 255).astype(np.uint8)
                    color_bgr = color_rgb_0_255[:, ::-1]

                    # Use the image with drawings
                    out_img = make_matching_plot_fast(
                        image0=viz_drone_img, image1=viz_sat_img_drawn,
                        kpts0=drone_image.key_points.keypoints,
                        kpts1=matched_image.key_points.keypoints,
                        mkpts0=matched_kpts0, mkpts1=matched_kpts1, color=color_bgr,
                        text="", path=None, show_keypoints=False
                    )

                    # Add Matcher Name Text
                    matcher_name = "SuperGlue"
                    font = cv2.FONT_HERSHEY_SIMPLEX; font_scale = 1.5; font_thickness = 3; text_color = (255, 255, 0)
                    text_size, _ = cv2.getTextSize(matcher_name, font, font_scale, font_thickness)
                    text_x = out_img.shape[1] - text_size[0] - 20
                    text_y = text_size[1] + 20
                    cv2.putText(out_img, matcher_name, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

                    # Ensure uint8 and contiguous before saving
                    if out_img.dtype != np.uint8: out_img = out_img.astype(np.uint8)
                    if not out_img.flags['C_CONTIGUOUS']: out_img = np.ascontiguousarray(out_img)

                    self.save_viz(out_img, viz_path) # PNG olarak kaydet
                    self.logger.info(f"[SuperGlue] Saved visualization: {viz_path}")

                except Exception as e_viz:
                    self.logger.error(f"[SuperGlue] Error creating/saving visualization for {drone_image.name}: {e_viz}", exc_info=True)
                    # If viz fails, still report the match found in return dict
                    if error_message is None: # Don't overwrite previous errors
                        error_message = f"Visualization failed: {e_viz}"


        # --- Return Results ---
        return {
            "is_match": is_match,
            "gt_coordinate": gt_coordinates,
            "predicted_coordinate": final_pred_coords,
            "center": center,
            "best_dst": best_dst.tolist() if isinstance(best_dst, np.ndarray) else None,
            "num_inliers": num_inliers_best if num_inliers_best > -1 else 0,
            "matched_image": matched_image.name if matched_image else None,
            "distance": final_distance_m,
            "error": error_message # Include any error that occurred
        }

    def run(self, output_path: Union[str, Path] = None) -> List[Dict[str, Any]]:
        """Runs the SuperGlue pipeline on all drone images."""
        self.logger.info(f"[SuperGlue] Running the pipeline on {len(self.drone_streamer)} images")
        preds = []
        num_matches = 0

        viz_output_path = None
        if output_path:
             viz_output_path = Path(output_path)
             viz_output_path.mkdir(parents=True, exist_ok=True)

        # Ensure iterator reset if needed (though this is usually the first pipeline)
        if hasattr(self.drone_streamer, '__iter__'):
             self.drone_streamer.__iter__()

        for i, drone_image_orig in enumerate(self.drone_streamer):
            self.logger.info(f"[SuperGlue] Processing image {i+1}/{len(self.drone_streamer)}: {drone_image_orig.name}")
            processed_query = self.query_processor(drone_image_orig)

            try:
                pred = self.run_on_image(processed_query, viz_output_path)
            except Exception as e:
                self.logger.error(f"[SuperGlue] Unhandled error processing {drone_image_orig.name}: {e}", exc_info=True)
                pred = {"is_match": False, "distance": None, "error": f"Unhandled error: {e}", "num_inliers": 0}

            if isinstance(pred.get("best_dst"), np.ndarray):
                 pred["best_dst"] = pred["best_dst"].tolist()
            # Convert GpsCoordinate to dict for easier JSON serialization if needed later
            if isinstance(pred.get("gt_coordinate"), GpsCoordinate):
                 pred["gt_coordinate"] = {"lat": pred["gt_coordinate"].lat, "long": pred["gt_coordinate"].long}
            if isinstance(pred.get("predicted_coordinate"), GpsCoordinate):
                 pred["predicted_coordinate"] = {"lat": pred["predicted_coordinate"].lat, "long": pred["predicted_coordinate"].long}


            preds.append(pred)
            if pred.get("is_match", False):
                num_matches += 1
                if pred.get("distance") is not None and pred["distance"] > 50:
                    self.logger.warning(f"[SuperGlue] Large distance: {pred['distance']:.2f}m for {drone_image_orig.name}")

        self.logger.info(f"[SuperGlue] Pipeline finished. Found {num_matches} matches for {len(self.drone_streamer)} images")
        return preds

    # compute_geo_pose ve diğer yardımcı fonksiyonlar BasePipeline'dan miras alınır