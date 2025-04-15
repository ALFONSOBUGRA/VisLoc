"""
Coding standards and guidelines for the Visual Localization (VisLoc) project.

This module defines the standard practices, naming conventions, and code structure
that should be followed throughout the project to maintain consistency and readability.
"""

from typing import List


class CodingStandards:
    """
    Coding standards for the VisLoc project.
    
    This class provides a reference for the coding standards used in the project.
    It includes naming conventions, code organization guidelines, and documentation
    standards.
    """
    
    @staticmethod
    def get_naming_conventions() -> List[str]:
        """
        Get naming conventions for the project.
        
        Returns:
            List of naming convention rules.
        """
        return [
            "Class names: Use CamelCase (e.g., SatelliteMapReader)",
            "Method/function names: Use snake_case (e.g., compute_metrics)",
            "Variable names: Use snake_case (e.g., output_path)",
            "Constant names: Use UPPER_SNAKE_CASE (e.g., DEFAULT_CONFIG_PATH)",
            "Module names: Use snake_case (e.g., map_reader.py)",
            "Package names: Use snake_case (e.g., keypoint_pipeline)",
            "Private methods/attributes: Prefix with underscore (e.g., _init_pipeline)",
        ]
    
    @staticmethod
    def get_documentation_standards() -> List[str]:
        """
        Get documentation standards for the project.
        
        Returns:
            List of documentation standard rules.
        """
        return [
            "Use docstrings for all modules, classes, methods, and functions",
            "Use Google-style docstrings with type annotations",
            "Specify parameter types and return types in docstrings",
            "Include descriptions for all parameters",
            "Use triple quotes (''') for multi-line docstrings",
            "Keep docstrings concise but informative",
            "Use type hints in function signatures",
        ]
    
    @staticmethod
    def get_code_organization_standards() -> List[str]:
        """
        Get code organization standards for the project.
        
        Returns:
            List of code organization standard rules.
        """
        return [
            "Group imports in the following order: standard library, third-party, local",
            "Sort imports alphabetically within each group",
            "Use absolute imports for clarity",
            "Limit line length to 100 characters",
            "Use 4 spaces for indentation (no tabs)",
            "Follow the principle of one class per file when possible",
            "Organize methods in classes: constructor, public methods, private methods",
            "Separate logical sections with blank lines",
        ]
    
    @staticmethod
    def get_oop_principles() -> List[str]:
        """
        Get object-oriented programming principles for the project.
        
        Returns:
            List of OOP principles to follow.
        """
        return [
            "Single Responsibility Principle: Classes should have only one reason to change",
            "Open/Closed Principle: Classes should be open for extension but closed for modification",
            "Liskov Substitution Principle: Derived classes should be substitutable for base classes",
            "Interface Segregation: Clients should not depend on interfaces they don't use",
            "Dependency Inversion: Depend on abstractions, not concrete implementations",
            "Favor composition over inheritance",
            "Keep classes focused and cohesive",
            "Use abstract base classes for defining interfaces",
        ]
    
    @staticmethod
    def get_file_structure() -> str:
        """
        Get file structure guidelines for the project.
        
        Returns:
            String describing the file structure.
        """
        return """
        src/visloc/
        ├── __init__.py
        ├── config/
        │   ├── __init__.py
        │   └── config_handler.py
        ├── keypoint_pipeline/
        │   ├── __init__.py
        │   ├── base.py
        │   ├── detection_and_description.py
        │   ├── matcher.py
        │   └── typing.py
        ├── localization/
        │   ├── __init__.py
        │   ├── base.py
        │   ├── drone_streamer.py
        │   ├── map_reader.py
        │   ├── pipeline.py
        │   └── preprocessing.py
        ├── tms/
        │   ├── __init__.py
        │   ├── data_structures.py
        │   ├── geo.py
        │   └── schemas.py
        └── utils/
            ├── __init__.py
            ├── constants.py
            ├── helpers.py
            └── standards.py
        """


# Constants for the project
DEFAULT_CONFIG_PATH = "config.yaml"
DEFAULT_OUTPUT_PATH = "data/output"
LOGGING_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class FileHeader:
    """
    Standard file header template for the project.
    """
    
    @staticmethod
    def get_python_header() -> str:
        """
        Get the standard Python file header.
        
        Returns:
            String containing the standard file header.
        """
        return '''"""
Module description.

This module provides functionality for...
"""

from typing import Dict, List, Optional

# Standard library imports
import os
import logging
from pathlib import Path

# Third-party imports
import numpy as np
import torch

# Local imports
from visloc.utils.constants import DEFAULT_CONFIG_PATH
''' 