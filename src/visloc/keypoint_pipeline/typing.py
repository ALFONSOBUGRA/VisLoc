"""
Type definitions for the keypoint pipeline in the Visual Localization system.

This module defines configuration classes and data containers used throughout
the keypoint pipeline, including detector and matcher configurations and the
ImageKeyPoints class for storing keypoint data.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, TypeVar, overload

import numpy as np
import torch

# Type variable for tensor-like objects (numpy arrays or torch tensors)
TensorLike = Union[np.ndarray, torch.Tensor]


@dataclass
class DetectorConfig(ABC):
    """
    Abstract base configuration for keypoint detectors.
    
    Attributes:
        name: Name of the detector algorithm.
    """
    name: str


@dataclass
class SuperPointConfig(DetectorConfig):
    """
    Configuration for the SuperPoint keypoint detector and descriptor.
    
    Attributes:
        name: Name of the detector algorithm, defaults to "SuperPoint".
        device: Computation device ("cuda" or "cpu"), defaults to "cpu".
        nms_radius: Non-maximum suppression radius, defaults to 4.
        keypoint_threshold: Minimum confidence for keypoints, defaults to 0.005.
        max_keypoints: Maximum number of keypoints to return (-1 for no limit), defaults to -1.
    """
    name: str = "SuperPoint"
    device: str = "cpu"
    nms_radius: int = 4
    keypoint_threshold: float = 0.005
    max_keypoints: int = -1


@dataclass
class MatcherConfig(ABC):
    """
    Abstract base configuration for keypoint matchers.
    
    Attributes:
        name: Name of the matcher algorithm.
    """
    name: str


@dataclass
class SuperGlueConfig(MatcherConfig):
    """
    Configuration for the SuperGlue keypoint matcher.
    
    Attributes:
        name: Name of the matcher algorithm, defaults to "SuperGlue".
        device: Computation device ("cuda" or "cpu"), defaults to "cpu".
        weights: Pre-trained weights to use ("indoor" or "outdoor"), defaults to "outdoor".
        descriptor_dim: Dimension of keypoint descriptors, defaults to 256.
        keypoint_encoder: Dimensions of keypoint encoder layers, defaults to [32, 64, 128, 256].
        GNN_layers: Graph neural network layer types, defaults to ["self", "cross"] * 9.
        sinkhorn_iterations: Number of Sinkhorn iterations for matching, defaults to 100.
        match_threshold: Minimum confidence for matches, defaults to 0.2.
    """
    name: str = "SuperGlue"
    device: str = "cpu"
    weights: str = "outdoor"
    descriptor_dim: int = 256
    keypoint_encoder: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    GNN_layers: List[str] = field(default_factory=lambda: ["self", "cross"] * 9)
    sinkhorn_iterations: int = 100
    match_threshold: float = 0.2


@dataclass
class ImageKeyPoints:
    """
    Class to store keypoints, descriptors, and scores for an image.
    
    This class provides a container for keypoint data with methods for
    converting between different formats and devices.
    
    Attributes:
        keypoints: Keypoint coordinates as tensor of shape (N, 2).
        descriptors: Keypoint descriptors as tensor of shape (N, D).
        scores: Optional keypoint confidence scores as tensor of shape (N,).
        image_size: Optional tuple containing the original image dimensions (H, W).
    """
    keypoints: TensorLike
    descriptors: TensorLike
    scores: Optional[TensorLike] = None
    image_size: Optional[Tuple[int, int]] = None

    def __post_init__(self) -> None:
        """
        Validate the object state after initialization.
        
        Ensures that scores, if provided, match the number of keypoints.
        Also determines whether the data is stored as torch tensors.
        """
        if self.scores is not None:
            assert len(self.keypoints) == len(self.scores), \
                "Number of scores must match number of keypoints"

        self._is_torch = isinstance(self.keypoints, torch.Tensor)

    def __len__(self) -> int:
        """
        Get the number of keypoints.
        
        Returns:
            Number of keypoints in the container.
        """
        return len(self.keypoints)

    def __getitem__(self, index: int) -> ImageKeyPoints:
        """
        Get a subset of keypoints by index.
        
        Args:
            index: Index or slice to extract.
            
        Returns:
            New ImageKeyPoints object containing only the selected keypoints.
        """
        return ImageKeyPoints(
            keypoints=self.keypoints[index],
            descriptors=self.descriptors[index],
            scores=self.scores[index] if self.scores is not None else None,
        )

    def attributes(self) -> List[str]:
        """
        Get the list of attribute names in this container.
        
        Returns:
            List of attribute names.
        """
        return ["keypoints", "descriptors", "scores", "image_size"]

    @property
    def device(self) -> str:
        """
        Get the device where the data is stored.
        
        Returns:
            Device name ('cpu' for numpy arrays, or the torch device).
        """
        return self.keypoints.device if self.is_torch else "cpu"

    @property
    def is_torch(self) -> bool:
        """
        Check if the data is stored as torch tensors.
        
        Returns:
            True if data is stored as torch tensors, False if numpy arrays.
        """
        return self._is_torch

    def to(self, device: str) -> ImageKeyPoints:
        """
        Move the data to the specified device (for torch tensors only).
        
        Args:
            device: Target device name.
            
        Returns:
            New ImageKeyPoints object with data on the specified device.
        """
        if not self.is_torch:
            return self
            
        return ImageKeyPoints(
            keypoints=self.keypoints.to(device),
            descriptors=self.descriptors.to(device),
            scores=self.scores.to(device) if self.scores is not None else None,
            image_size=self.image_size,
        )

    def squeeze(self) -> ImageKeyPoints:
        """
        Remove singleton dimensions from all tensors.
        
        Returns:
            New ImageKeyPoints object with squeezed tensors.
        """
        return ImageKeyPoints(
            keypoints=self.keypoints.squeeze(),
            descriptors=self.descriptors.squeeze(),
            scores=self.scores.squeeze() if self.scores is not None else None,
            image_size=self.image_size,
        )

    def add_batch_dimension(self) -> ImageKeyPoints:
        """
        Add a batch dimension to all tensors.
        
        Returns:
            New ImageKeyPoints object with an added batch dimension.
        """
        return ImageKeyPoints(
            keypoints=self.keypoints[None, ...],
            descriptors=self.descriptors[None, ...],
            scores=self.scores[None, ...] if self.scores is not None else None,
            image_size=self.image_size,
        )

    def numpy(self) -> ImageKeyPoints:
        """
        Convert all data to numpy arrays.
        
        Returns:
            New ImageKeyPoints object with data as numpy arrays.
        """
        if not self.is_torch:
            return self
            
        return ImageKeyPoints(
            keypoints=self.keypoints.detach().cpu().numpy(),
            descriptors=self.descriptors.detach().cpu().numpy(),
            scores=self.scores.detach().cpu().numpy() if self.scores is not None else None,
            image_size=self.image_size,
        )

    def to_dict(self, suffix_idx: Optional[int] = None) -> Dict[str, TensorLike]:
        """
        Convert the object to a dictionary.
        
        Args:
            suffix_idx: Optional index to append to field names.
            
        Returns:
            Dictionary containing the keypoints data.
        """
        if suffix_idx is not None:
            return {
                f"keypoints{suffix_idx}": self.keypoints,
                f"descriptors{suffix_idx}": self.descriptors,
                f"scores{suffix_idx}": self.scores,
                "image_size": self.image_size,
            }
        else:
            return {
                "keypoints": self.keypoints,
                "descriptors": self.descriptors,
                "scores": self.scores,
                "image_size": self.image_size,
            }

    def torch(self) -> ImageKeyPoints:
        """
        Convert all data to torch tensors.
        
        Returns:
            New ImageKeyPoints object with data as torch tensors.
        """
        if self.is_torch:
            return self
            
        return ImageKeyPoints(
            keypoints=torch.tensor(self.keypoints, dtype=torch.float32),
            descriptors=torch.tensor(self.descriptors, dtype=torch.float32),
            scores=torch.tensor(self.scores, dtype=torch.float32) if self.scores is not None else None,
            image_size=self.image_size,
        )

    def detach(self) -> ImageKeyPoints:
        """
        Detach all torch tensors from the computation graph.
        
        Returns:
            New ImageKeyPoints object with detached tensors.
        """
        if not self.is_torch:
            return self
            
        return ImageKeyPoints(
            keypoints=self.keypoints.detach(),
            descriptors=self.descriptors.detach(),
            scores=self.scores.detach() if self.scores is not None else None,
            image_size=self.image_size,
        )
