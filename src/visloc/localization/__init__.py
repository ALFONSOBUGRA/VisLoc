"""
Localization module for the Visual Localization (VisLoc) project.

This package provides functionality for satellite-based visual localization,
including map readers, drone image streamers, and the main localization pipeline.
"""

from visloc.localization.base import BasePipeline, PipelineConfig, BaseMapReader
from visloc.localization.drone_streamer import DroneImageStreamer
from visloc.localization.map_reader import SatelliteMapReader, TileSatelliteMapReader
from visloc.localization.superglue_pipeline import Pipeline
from visloc.localization.preprocessing import QueryProcessor
from visloc.localization.tile_pipeline import TilePipeline

__all__ = [
    "BasePipeline",
    "BaseMapReader", 
    "DroneImageStreamer",
    "Pipeline",
    "PipelineConfig",
    "QueryProcessor",
    "SatelliteMapReader",
    "TilePipeline",
    "TileSatelliteMapReader",
]
