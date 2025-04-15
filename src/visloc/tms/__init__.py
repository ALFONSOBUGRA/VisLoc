"""
Tile Mapping Service (TMS) module for the Visual Localization (VisLoc) project.

This package provides functionality for handling tile-based mapping services,
including coordinate conversions, data structures for geographical information,
and utilities for downloading map tiles.
"""

from visloc.tms.data_structures import (
    CameraModel,
    DroneImage,
    FlightZone,
    GeoSatelliteImage,
    Tile,
    TileImage,
)
from visloc.tms.download import FlightZoneDownloader, TileDownloader
from visloc.tms.geo import (
    get_tile_xy_from_lat_long,
    haversine_distance,
    get_lat_long_from_tile_xy,
)
from visloc.tms.schemas import GeoPoint, GpsCoordinate, Orientation, TileCoordinate

__all__ = [
    "CameraModel",
    "DroneImage",
    "FlightZone",
    "FlightZoneDownloader",
    "GeoPoint",
    "GeoSatelliteImage",
    "GpsCoordinate",
    "Orientation",
    "Tile",
    "TileCoordinate",
    "TileDownloader",
    "TileImage",
    "get_tile_xy_from_lat_long",
    "haversine_distance",
    "get_lat_long_from_tile_xy",
]
