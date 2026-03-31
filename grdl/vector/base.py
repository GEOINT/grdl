# -*- coding: utf-8 -*-
"""
Vector Processor Base Class - Abstract interface for vector processors.

Defines the ``VectorProcessor`` ABC for processors that operate on
``FeatureSet`` objects rather than raster arrays.  Inherits from
``ImageProcessor`` for version checking and parameter flow.

Author
------
Duane Smalley, PhD
duane.d.smalley@gmail.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-03-25
"""

# Standard library
import logging
from abc import abstractmethod
from typing import Any, TYPE_CHECKING

# Third-party
import numpy as np

# GRDL internal
from grdl.image_processing.base import ImageProcessor
from grdl.vector.models import FeatureSet

if TYPE_CHECKING:
    from grdl.IO.models.base import ImageMetadata

logger = logging.getLogger(__name__)


class VectorProcessor(ImageProcessor):
    """
    Abstract base class for vector feature processors.

    Unlike ``ImageTransform`` (ndarray in, ndarray out), a
    ``VectorProcessor`` takes a ``FeatureSet`` and produces a
    ``FeatureSet``.  Spatial operations (buffer, clip, dissolve, etc.)
    are implemented as ``VectorProcessor`` subclasses.

    Subclasses must implement:

    - ``process()`` -- operate on a FeatureSet

    Examples
    --------
    >>> processor = SomeVectorOp(distance=100.0)
    >>> result = processor.process(features)
    >>> len(result)
    42
    """

    def execute(
        self,
        metadata: 'ImageMetadata',
        source: Any,
        **kwargs: Any,
    ) -> tuple:
        """Execute the vector processor.

        Dispatches to ``process()`` if *source* is a ``FeatureSet``.
        Raises ``TypeError`` with a clear message if *source* is an
        ndarray (wrong processor type for raster data).

        Parameters
        ----------
        metadata : ImageMetadata
            Input metadata.
        source : FeatureSet
            Input feature set.

        Returns
        -------
        tuple[FeatureSet, ImageMetadata]
        """
        if isinstance(source, np.ndarray):
            raise TypeError(
                f"{type(self).__name__} operates on FeatureSet objects, "
                f"not numpy arrays. Use an ImageTransform for raster data."
            )
        if not isinstance(source, FeatureSet):
            raise TypeError(
                f"{type(self).__name__} expects a FeatureSet, "
                f"got {type(source).__name__}."
            )
        self._metadata = metadata
        logger.debug(
            "VectorProcessor start: %d features", len(source)
        )
        result = self.process(source, **kwargs)
        return result, metadata

    @abstractmethod
    def process(
        self, features: FeatureSet, **kwargs: Any
    ) -> FeatureSet:
        """
        Process a feature set.

        Parameters
        ----------
        features : FeatureSet
            Input feature set.

        Returns
        -------
        FeatureSet
            Processed feature set.
        """
        ...
