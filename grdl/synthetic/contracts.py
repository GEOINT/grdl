"""
Synthetic Generation Contracts - Interfaces for data generators and annotations.

Author
------
GRDX Team

License
-------
MIT License
Copyright (c) 2024 geoint.org
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
import numpy as np

class AnnotationWriterBase(ABC):
    """Abstract interface for synthetic data annotation writers.

    Plugins implementing this interface can format bounding boxes, masks,
    and metadata into specific output formats (e.g., GeoJSON, COCO, YOLO).
    """

    @property
    @abstractmethod
    def format_name(self) -> str:
        """The identifier for this annotation format (e.g., 'geojson')."""
        pass

    @abstractmethod
    def write_annotation(
        self,
        output_dir: str,
        base_filename: str,
        bounding_boxes: List[Tuple[float, float, float, float]],
        masks: List[np.ndarray],
        metadata: List[Dict[str, Any]]
    ) -> None:
        """Write the annotations to disk.

        Parameters
        ----------
        output_dir : str
            Directory to save the annotation file(s).
        base_filename : str
            Base filename matching the corresponding image chips.
        bounding_boxes : List[Tuple[float, float, float, float]]
            List of [xmin, ymin, xmax, ymax] boxes for generated objects.
        masks : List[np.ndarray]
            List of 2D boolean masks for each object.
        metadata : List[Dict[str, Any]]
            List of metadata dictionaries per object (e.g., state parameterization).
        """
        pass


class PhysicalObject:
    """Descriptor for a generic physical object before sensor projection.
    
    Represents the 3D geometry and material properties independent of any 
    specific sensor modality.
    
    Dictionaries for materials and metadata should use the standardized 
    hierarchical field names from `grdl.image_processing.detection.fields.Fields`
    where applicable (e.g., `Fields.trait.MATERIAL`, `Fields.sar.RCS`).
    """
    def __init__(self, geometry_type: str, dimensions: Dict[str, float], materials: Dict[str, Any], metadata: Dict[str, Any] = None):
        self.geometry_type = geometry_type
        self.dimensions = dimensions
        self.materials = materials
        self.metadata = metadata or {}


class ObjectGeneratorBase(ABC):
    """Abstract interface for synthetic object geometry/material generators."""

    @property
    @abstractmethod
    def generator_name(self) -> str:
        pass

    @abstractmethod
    def generate_object(self, params: Dict[str, Any]) -> PhysicalObject:
        pass


class ImageInjectorBase(ABC):
    """Abstract interface for modality-specific object injectors.
    
    Takes a generic physical object and renders it into a specific 
    sensor modality chip based on physics, orientation, and perspective.
    
    Implementations must declare their target modalities using the 
    `@processor_tags` decorator from `grdl.image_processing.versioning`,
    e.g., `@processor_tags(modalities=[ImageModality.SAR])`.
    """

    @property
    @abstractmethod
    def injector_name(self) -> str:
        """The identifier for this injector."""
        pass

    @abstractmethod
    def inject(
        self,
        base_chip: np.ndarray,
        obj: PhysicalObject,
        position_xy: Tuple[float, float],
        sensor_perspective: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Injects the object into the image chip.

        Parameters
        ----------
        base_chip : np.ndarray
            The original image chip for this modality.
        obj : PhysicalObject
            The generic object to inject.
        position_xy : Tuple[float, float]
            The pixel coordinates at which to center/place the object.
        sensor_perspective : Dict[str, Any]
            Sensor-specific parameters (look angle, resolution, etc.).

        Returns
        -------
        Tuple containing:
            - np.ndarray: Modified image chip
            - np.ndarray: 2D boolean mask of pixels affected by the object
        """
        pass
