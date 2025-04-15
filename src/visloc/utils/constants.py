import math
import torch # Import torch to check for CUDA availability

EARTH_RADIUS = 6371.0  # km
EARTH_CIRCUMFERENCE = 2 * math.pi * EARTH_RADIUS  # km
EQUTORIAL_CIRCUMFERENCE_METERS = 40075016.686  # meters
TILE_SIZE = 256  # pixels
MAX_ZOOM_LEVEL = 20  # maximum zoom level
MIN_ZOOM_LEVEL = 1  # minimum zoom level
MAX_LATITUDE = 90  # degrees
MIN_LATITUDE = -90  # degrees
MAX_LONGITUDE = 180  # degrees
MIN_LONGITUDE = -180  # degrees

"""
Constants used throughout the Visual Localization (VisLoc) project.

This module defines global constants to ensure consistency across the project.
"""

# Configuration constants
DEFAULT_CONFIG_PATH = "config.yaml"
DEFAULT_ENV_VAR_NAME = "VISLOC_CONFIG_PATH"

# Path constants
DEFAULT_OUTPUT_PATH = "data/output"
DEFAULT_MAP_PATH = "data/map/"
DEFAULT_QUERY_PATH = "data/query/"

# Pipeline constants
DEFAULT_RESIZE_SIZE = (800,)

# SuperPoint constants (for SuperGlue pipeline)
DEFAULT_NMS_RADIUS = 4
DEFAULT_KEYPOINT_THRESHOLD = 0.005
DEFAULT_MAX_KEYPOINTS = 1024

# SuperGlue constants (for SuperGlue pipeline)
DEFAULT_WEIGHTS = "indoor"
DEFAULT_SINKHORN_ITERATIONS = 20
DEFAULT_MATCH_THRESHOLD = 0.2

# OmniGlue constants
DEFAULT_OG_EXPORT_PATH = "models/og_export"
DEFAULT_SP_EXPORT_PATH = "models/sp_v6" # SuperPoint for OmniGlue
DEFAULT_DINO_EXPORT_PATH = "models/dinov2_vitb14_pretrain.pth"
DEFAULT_OG_MATCH_THRESHOLD = 0.02

# Camera constants
DEFAULT_FOCAL_LENGTH = 0.0045  # 4.5mm
DEFAULT_RESOLUTION_HEIGHT = 4056
DEFAULT_RESOLUTION_WIDTH = 3040
DEFAULT_HFOV_DEG = 82.9

# Logging constants
DEFAULT_LOGGING_LEVEL = "INFO"
DEFAULT_LOGGING_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_LOGGING_DATE_FORMAT = "%H:%M:%S"

# Device constants
CPU_DEVICE = "cpu"
DEFAULT_DEVICE = CPU_DEVICE # Default to CPU initially
if torch.cuda.is_available():
    DEFAULT_DEVICE = "cuda" # Use CUDA if available (SuperGlue/SuperPoint benefit)
# Note: OmniGlue (TF) might handle its device selection separately.
#       DINO (Torch) within OmniGlue needs device awareness.
#       The OG wrapper seems to handle device placement for DINO.