"""
Base classes for keypoint detection, description, and matching in the Visual Localization pipeline.

This module defines the abstract base classes for keypoint detection, description, 
and matching functionality. These classes provide the interfaces that specific 
implementations like SuperPoint and SuperGlue must adhere to.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, ClassVar, Type, TypeVar

import cv2
import numpy as np
import torch

from visloc.keypoint_pipeline.typing import ImageKeyPoints

T = TypeVar('T', bound='CombinedKeyPointAlgorithm')


class BaseKeyPointDescriptor(ABC):
    """
    Abstract base class for keypoint descriptors.
    
    This class defines the interface for algorithms that generate descriptors
    for keypoints in an image.
    """

    def __init__(self) -> None:
        """Initialize the keypoint descriptor."""
        super().__init__()

    @abstractmethod
    def describe_keypoints(self, image: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        """
        Describe keypoints in an image.
        
        Args:
            image: Image to describe keypoints in (HxWxC numpy array).
            keypoints: Keypoints to describe, typically as Nx2 array of (x, y) coordinates.
            
        Returns:
            Descriptors for the keypoints, typically as NxD array where D is the descriptor dimension.
        """
        pass


class KeyPointDescriptor(BaseKeyPointDescriptor):
    """
    Concrete base class for keypoint descriptors.
    
    This class provides a concrete implementation of BaseKeyPointDescriptor
    that can be extended by specific descriptor algorithms.
    """
    
    pass


class BaseKeyPointDetector(ABC):
    """
    Abstract base class for keypoint detectors.
    
    This class defines the interface for algorithms that detect keypoints
    in an image.
    """

    def __init__(self) -> None:
        """Initialize the keypoint detector."""
        super().__init__()

    @abstractmethod
    def detect_keypoints(self, image: np.ndarray) -> np.ndarray:
        """
        Detect keypoints in an image.
        
        Args:
            image: Image to detect keypoints in (HxWxC numpy array).
            
        Returns:
            Detected keypoints, typically as Nx2 array of (x, y) coordinates.
        """
        pass


class KeyPointDetector(BaseKeyPointDetector):
    """
    Concrete base class for keypoint detectors.
    
    This class provides a concrete implementation of BaseKeyPointDetector
    that can be extended by specific detector algorithms.
    """
    
    def _keypoints_to_array(self, keypoints: List[cv2.KeyPoint]) -> np.ndarray:
        """
        Convert OpenCV KeyPoint objects to a numpy array of coordinates.
        
        Args:
            keypoints: List of OpenCV KeyPoint objects.
            
        Returns:
            Array of keypoint coordinates, as Nx2 array of (x, y) coordinates.
        """
        return np.array([kp.pt for kp in keypoints], dtype=np.float32)


class CombinedKeyPointAlgorithm(ABC):
    """
    Abstract base class for combined keypoint detection and description.
    
    This class defines the interface for algorithms that both detect and describe
    keypoints in a single pass, such as SuperPoint.
    """
    
    _registry: ClassVar[Dict[str, Type[T]]] = {}
    
    def __init__(self) -> None:
        """Initialize the combined keypoint algorithm."""
        super().__init__()
    
    @classmethod
    def register(cls, algorithm_cls: Type[T]) -> Type[T]:
        """
        Register a combined keypoint algorithm implementation.
        
        Args:
            algorithm_cls: The algorithm class to register.
            
        Returns:
            The registered class (allows use as a decorator).
        """
        cls._registry[algorithm_cls.__name__] = algorithm_cls
        return algorithm_cls
    
    @classmethod
    def get_algorithm(cls, name: str) -> Type[T]:
        """
        Get a registered algorithm by name.
        
        Args:
            name: Name of the algorithm to retrieve.
            
        Returns:
            The algorithm class.
            
        Raises:
            KeyError: If the algorithm name is not registered.
        """
        if name not in cls._registry:
            raise KeyError(f"Algorithm {name} not registered")
        return cls._registry[name]

    @abstractmethod
    def detect_keypoints(self, image: np.ndarray) -> np.ndarray:
        """
        Detect keypoints in an image.
        
        Args:
            image: Image to detect keypoints in (HxWxC numpy array).
            
        Returns:
            Detected keypoints, typically as Nx2 array of (x, y) coordinates.
        """
        pass

    @abstractmethod
    def describe_keypoints(self, image: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        """
        Describe keypoints in an image.
        
        Args:
            image: Image to describe keypoints in (HxWxC numpy array).
            keypoints: Keypoints to describe, typically as Nx2 array of (x, y) coordinates.
            
        Returns:
            Descriptors for the keypoints, typically as NxD array where D is the descriptor dimension.
        """
        pass

    @abstractmethod
    def detect_and_describe_keypoints(self, image: np.ndarray) -> ImageKeyPoints:
        """
        Detect and describe keypoints in an image.
        
        Args:
            image: Image to detect and describe keypoints in (HxWxC numpy array).
            
        Returns:
            Object containing both keypoints and their descriptors.
        """
        pass


class KeyPointMatcher(ABC):
    """
    Abstract base class for keypoint matchers.
    
    This class defines the interface for algorithms that match keypoints
    between two images based on their descriptors.
    """

    def __init__(self) -> None:
        """Initialize the keypoint matcher."""
        super().__init__()

    @abstractmethod
    def match_keypoints(
        self, keypoints1: ImageKeyPoints, keypoints2: ImageKeyPoints
    ) -> np.ndarray:
        """
        Match keypoints between two sets of descriptors.
        
        Args:
            keypoints1: Keypoints and descriptors from the first image.
            keypoints2: Keypoints and descriptors from the second image.
            
        Returns:
            Array of matches, typically containing indices or scores indicating 
            the strength of each match.
        """
        pass
