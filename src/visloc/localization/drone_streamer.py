import logging
from pathlib import Path
from typing import Union, Dict, Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from visloc.tms.data_structures import DroneImage
from visloc.tms.schemas import GeoPoint, Orientation


class DroneImageStreamer:
    """Drone image streamer that reads drone images from a folder.

    Parameters
    ----------
    image_folder : Union[str, Path]
        path to the folder containing the images to stream, if the images have ground
        truth metadata, the folder should also contain a CSV file with the metadata.
    has_gt : bool
        whether the images have ground truth metadata
    logger : logging.Logger
        logger to use for logging
    # CHANGE: Added read mode parameter, defaulting to COLOR
    cv2_read_mode : int
        OpenCV image reading mode (e.g., cv2.IMREAD_COLOR, cv2.IMREAD_GRAYSCALE)

    """

    IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    COLUMN_NAMES = [
        "Filename",
        "Latitude",
        "Longitude",
        "Altitude",
        "Gimball_Roll",
        "Gimball_Yaw",
        "Gimball_Pitch",
        "Flight_Roll",
        "Flight_Yaw",
        "Flight_Pitch",
    ]

    def __init__(
        self,
        image_folder: Union[str, Path],
        has_gt: bool = False,
        logger: logging.Logger = None,
        # CHANGE: Default to COLOR for OmniGlue compatibility
        cv2_read_mode: int = cv2.IMREAD_COLOR,
    ) -> None:
        self.image_folder = (
            image_folder if isinstance(image_folder, Path) else Path(image_folder)
        )
        if not self.image_folder.exists():
            raise FileNotFoundError(f"Image folder not found at {self.image_folder}")
        # Store the read mode
        self.cv2_read_mode = cv2_read_mode
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger
        self.has_gt = has_gt
        self._image_db: Dict[str, DroneImage] = {} # Initialize database
        self._num_images: int = 0 # Initialize count
        self._metadata: Optional[pd.DataFrame] = None # Initialize metadata
        self._initialize_db()

    def __len__(self) -> int:
        return self._num_images

    def __add__(self, other: "DroneImageStreamer") -> "DroneImageStreamer":
        new_streamer = DroneImageStreamer(
            image_folder=self.image_folder,
            has_gt=self.has_gt,
            logger=self.logger,
        )
        new_streamer._image_db = {**self._image_db, **other._image_db}
        new_streamer._num_images = len(new_streamer._image_db)
        return new_streamer

    @property
    def image_names(self) -> list[str]:
        """List of image names in the streamer."""
        return list(self._image_db.keys())

    def _initialize_db(self) -> None:
        """Initialize the image database."""
        # Reset before building
        self._image_db = dict()
        self._num_images = 0
        self._metadata = None
        if self.has_gt:
            self._build_image_db_with_gt()
        else:
            self._build_image_db()
        self.current_idx = 0

    def _build_image_db(self) -> None:
        """Build the image database without ground truth metadata."""
        self.logger.info(f"Building image database for {self.image_folder}")
        for image_path in tqdm(
            self.image_folder.glob("*"), total=len(list(self.image_folder.glob("*")))
        ):
            if image_path.suffix in self.IMAGE_EXTENSIONS:
                drone_image = DroneImage(image_path=image_path)
                self._image_db[image_path.name] = drone_image
                self._num_images += 1
        self.logger.info(
            f"Image database built successfully with {self._num_images} images"
        )

    def _build_image_db_with_gt(self) -> None:
        """Build the image database with ground truth metadata.

        The ground truth metadata should be in a CSV file with the following columns:
        - Filename
        - Latitude
        - Longitude
        - Altitude
        - Gimball_Roll
        - Gimball_Yaw
        - Gimball_Pitch
        - Flight_Roll
        - Flight_Yaw
        - Flight_Pitch
        """
        csv_files = list(self.image_folder.glob("*.csv"))
        if len(csv_files) == 0:
            raise FileNotFoundError(f"No CSV files found in {self.image_folder}")
        if len(csv_files) > 1:
            raise ValueError(f"Multiple CSV files found in {self.image_folder}")
        csv_file = csv_files[0]
        self.logger.info(f"Building image database with GT from {csv_file}")
        self._metadata = pd.read_csv(csv_file)
        for _, row in tqdm(self._metadata.iterrows(), total=len(self._metadata)):
            image_path = self.image_folder / row["Filename"]
            drone_image = DroneImage(
                image_path=image_path,
                geo_point=GeoPoint(
                    latitude=row["Latitude"],
                    longitude=row["Longitude"],
                    altitude=row["Altitude"],
                ),
                camera_orientation=Orientation(
                    pitch=row["Gimball_Pitch"],
                    roll=row["Gimball_Roll"],
                    yaw=row["Gimball_Yaw"],
                ),
                drone_orientation=Orientation(
                    pitch=row["Flight_Pitch"],
                    roll=row["Flight_Roll"],
                    yaw=row["Flight_Yaw"],
                ),
            )
            self._image_db[image_path.name] = drone_image
            self._num_images += 1
        self.logger.info(
            f"Image database built successfully with {self._num_images} images"
        )

    def read_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """Read an image from a path using the specified read mode.

        Parameters
        ----------
        image_path : Union[str, Path]
            path to the image

        Returns
        -------
        np.ndarray
            image as a numpy array, or None if read fails.
        """
        image_path = image_path if isinstance(image_path, Path) else Path(image_path)
        if not image_path.exists():
            # Raise error or log and return None? Let's log and return None.
            self.logger.error(f"Image not found at {image_path}")
            return None
        # Use the stored cv2_read_mode
        img = cv2.imread(str(image_path), self.cv2_read_mode)
        if img is None:
             self.logger.error(f"Failed to read image {image_path} with mode {self.cv2_read_mode}")
        return img

    def __next__(self) -> DroneImage:
        """Get the next image in the streamer."""
        if self.current_idx >= self._num_images:
             raise StopIteration

        image_name = self.image_names[self.current_idx]
        drone_image: DroneImage = self._image_db[image_name]

        # Load image only when requested by iterator
        if drone_image.image is None:
            drone_image.image = self.read_image(drone_image.image_path)
            # Handle case where image loading failed
            if drone_image.image is None:
                 self.logger.warning(f"Skipping image {drone_image.name} due to loading error.")
                 self.current_idx += 1
                 # Try fetching the next one recursively, or raise StopIteration if at end
                 if self.current_idx < self._num_images:
                      return self.__next__()
                 else:
                      raise StopIteration


        self.current_idx += 1
        return drone_image

    def __iter__(self) -> "DroneImageStreamer":
        """Return the streamer as an iterator."""
        self.current_idx = 0
        # Reset loaded images? No, keep them loaded if already accessed.
        return self
