"""
Keypoint pipeline for the Visual Localization (VisLoc) project.

This package provides functionality for keypoint detection, description, and matching.
It includes implementations of the SuperPoint algorithm for keypoint detection and
description, and the SuperGlue algorithm for keypoint matching.
"""

from visloc.keypoint_pipeline.base import (
    BaseKeyPointDescriptor,
    BaseKeyPointDetector,
    CombinedKeyPointAlgorithm,
    KeyPointDescriptor,
    KeyPointDetector,
    KeyPointMatcher,
)
from visloc.keypoint_pipeline.detection_and_description import SuperPointAlgorithm
from visloc.keypoint_pipeline.matcher import SuperGlueMatcher
from visloc.keypoint_pipeline.typing import ImageKeyPoints, SuperGlueConfig, SuperPointConfig

__all__ = [
    "BaseKeyPointDescriptor",
    "BaseKeyPointDetector",
    "CombinedKeyPointAlgorithm",
    "ImageKeyPoints",
    "KeyPointDescriptor",
    "KeyPointDetector",
    "KeyPointMatcher",
    "SuperGlueConfig",
    "SuperGlueMatcher",
    "SuperPointAlgorithm",
    "SuperPointConfig",
] 