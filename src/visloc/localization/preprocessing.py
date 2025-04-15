"""
Preprocessing module for the Visual Localization (VisLoc) project.

This module provides functionality for preprocessing drone images before 
localization, including resizing and warping operations to align with
satellite imagery.
"""

from typing import Dict, List, Optional, Tuple, Callable, Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from superglue.utils import process_resize
from visloc.tms.data_structures import CameraModel, DroneImage


def get_intrinsics(camera_model: CameraModel, scale: float = 1.0) -> np.ndarray:
    """
    Get the intrinsics matrix of a camera model.
    
    Args:
        camera_model: The camera model containing focal length and principal point.
        scale: Scale factor to apply to the focal length, default is 1.0.
        
    Returns:
        3x3 camera intrinsics matrix.
    """
    intrinsics = np.array(
        [
            [camera_model.focal_length_px / scale, 0, camera_model.principal_point_x],
            [0, camera_model.focal_length_px / scale, camera_model.principal_point_y],
            [0, 0, 1],
        ]
    )
    return intrinsics


def rotation_matrix_from_angles(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Compute the rotation matrix from the roll, pitch, and yaw angles.
    
    Args:
        roll: Roll angle in degrees.
        pitch: Pitch angle in degrees.
        yaw: Yaw angle in degrees.
        
    Returns:
        3x3 rotation matrix.
    """
    r = Rotation.from_euler("xyz", [roll, pitch, yaw], degrees=True).as_matrix()
    return r


class QueryProcessor:
    """
    Class to process query (drone) images for visual localization.
    
    This class applies various preprocessing steps to drone images to prepare
    them for matching against satellite imagery, such as resizing to match
    resolution or warping to account for perspective.
    
    Attributes:
        size: Target size for resized images.
        camera_model: Camera model for the drone images.
        satellite_resolution: Resolution of the satellite imagery in meters per pixel.
        processings: List of processing steps to apply.
        fcts: Dictionary mapping processing names to processing functions.
    """

    def __init__(
        self,
        processings: Optional[List[str]] = None,
        size: Optional[Tuple[int, ...]] = None,
        camera_model: Optional[CameraModel] = None,
        satellite_resolution: Optional[float] = None,
    ) -> None:
        """
        Initialize the QueryProcessor.
        
        Args:
            processings: List of processing steps to apply. Supported values: "resize", "warp".
            size: Target size for resized images.
            camera_model: Camera model for the drone images.
            satellite_resolution: Resolution of the satellite imagery in meters per pixel.
        """
        self.size = size
        self.camera_model = camera_model
        self.satellite_resolution = satellite_resolution
        self.processings = processings or []
        self.fcts: Dict[str, Callable[[DroneImage], DroneImage]] = {
            "resize": self.resize_image,
            "warp": self.warp_image,
        }

    def __call__(self, query: DroneImage) -> DroneImage:
        """
        Process the query image by applying all configured processing steps.
        
        Args:
            query: The drone image to process.
            
        Returns:
            Processed drone image.
        """
        if not self.processings:
            return query
            
        processed_query = query
        for processing in self.processings:
            if processing in self.fcts:
                processed_query = self.fcts[processing](processed_query)
                
        return processed_query

    def resize_image(self, query: DroneImage) -> DroneImage:
        """
        Resize the query image to match either a specific size or the satellite resolution.
        
        If size is provided, the image is resized to that size.
        If camera_model and satellite_resolution are provided, the image is 
        resized to match the meters-per-pixel resolution of the satellite imagery.
        
        Args:
            query: The drone image to resize.
            
        Returns:
            Drone image with resized image data.
        """
        image = query.image.copy()
        
        if self.size is not None:
            # Resize to specified size
            height, width = image.shape[:2]
            new_width, new_height = process_resize(width, height, self.size)
            resized_image = cv2.resize(
                image, (new_width, new_height), interpolation=cv2.INTER_AREA
            )
            query.image = resized_image

        elif self.camera_model is not None and self.satellite_resolution is not None:
            # Resize to match satellite resolution
            new_size = self.compute_resize_shape(
                self.camera_model, query.geo_point.altitude, self.satellite_resolution
            )
            resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
            query.image = resized_image

        return query

    def compute_resize_scale(
        self,
        camera_model: CameraModel,
        altitude: float,
        satellite_resolution: float,
    ) -> float:
        """
        Compute the scale factor to resize the query image to match satellite resolution.
        
        Args:
            camera_model: Camera model of the drone.
            altitude: Altitude of the drone in meters.
            satellite_resolution: Resolution of the satellite imagery in meters per pixel.
            
        Returns:
            Scale factor to apply to the image size.
        """
        # Calculate the horizontal field of view in meters
        hvof_m = 2 * altitude * np.tan(camera_model.hfov_rad / 2)
        
        # Calculate the drone image resolution in meters per pixel
        drone_resolution = hvof_m / camera_model.resolution_width
        
        # Compute the resize scale
        resize_scale = drone_resolution / satellite_resolution
        return resize_scale

    def compute_resize_shape(
        self,
        camera_model: CameraModel,
        altitude: float,
        satellite_resolution: float,
    ) -> Tuple[int, int]:
        """
        Compute the target shape for resizing the image to match satellite resolution.
        
        Args:
            camera_model: Camera model of the drone.
            altitude: Altitude of the drone in meters.
            satellite_resolution: Resolution of the satellite imagery in meters per pixel.
            
        Returns:
            Target width and height for the resized image.
        """
        resize_scale = self.compute_resize_scale(
            camera_model, altitude, satellite_resolution
        )
        
        # Compute the new size
        new_width = int(camera_model.resolution_width * resize_scale)
        new_height = int(camera_model.resolution_height * resize_scale)
        
        return (new_width, new_height)

    def warp_image(self, query: DroneImage) -> DroneImage:
        """
        Warp the query image to account for the drone's perspective.
        
        This applies a homography transformation based on the drone's orientation
        to create a nadir (top-down) view that better matches satellite imagery.
        
        Args:
            query: The drone image to warp.
            
        Returns:
            Drone image with warped image data.
        """
        if self.camera_model is None:
            return query
            
        # Copy the image to avoid modifying the original
        image = query.image.copy()
        height, width = image.shape[:2]
        
        # Get the rotation matrix from the drone's orientation
        rotation_matrix = rotation_matrix_from_angles(
            query.orientation.roll, query.orientation.pitch, query.orientation.yaw
        )
        
        # Get the intrinsics matrix
        scale = 1.0
        if self.size is not None:
            # Compute the scale factor for the intrinsics matrix
            new_width, new_height = process_resize(width, height, self.size)
            scale = new_width / width
            
        intrinsics = get_intrinsics(self.camera_model, scale)
        
        # Compute the homography matrix
        homography = intrinsics @ rotation_matrix @ np.linalg.inv(intrinsics)
        
        # Apply the homography
        if self.size is not None:
            # Resize and warp in one step
            new_width, new_height = process_resize(width, height, self.size)
            warped_image = cv2.warpPerspective(
                image, homography, (new_width, new_height)
            )
        else:
            # Just warp
            warped_image = cv2.warpPerspective(image, homography, (width, height))
            
        query.image = warped_image
        return query
