# -*- coding: utf-8 -*-
"""
Raster-Vector Conversion - Convert between raster arrays and FeatureSets.

Provides ``RasterToPoints`` for extracting point features from raster data
and ``Rasterize`` for burning feature geometries into raster arrays.

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
from typing import Annotated, Any

# Third-party
import numpy as np
from shapely.geometry import Point, box as shapely_box
from shapely.geometry.base import BaseGeometry

# GRDL internal
from grdl.image_processing.params import Desc, Range
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.vocabulary import ProcessorCategory
from grdl.vector.models import Feature, FeatureSet


@processor_version('1.0.0')
@processor_tags(
    category=ProcessorCategory.ANALYZE,
    description='Convert raster pixels above threshold to point features.',
)
class RasterToPoints:
    """
    Convert a raster array to a FeatureSet of point features.

    Extracts pixel locations where values exceed a threshold and creates
    a point feature for each.  Supports sub-sampling via ``sample_step``.

    Parameters
    ----------
    threshold : float
        Minimum pixel value to include.  Default 0.0.
    band : int
        Band index for multi-band arrays.  Default 0.
    sample_step : int
        Step size for spatial sub-sampling.  Default 1 (every pixel).
    """

    threshold: Annotated[float, Desc('Minimum pixel value threshold')] = 0.0
    band: Annotated[int, Range(min=0), Desc('Band index for multi-band arrays')] = 0
    sample_step: Annotated[int, Range(min=1), Desc('Spatial sub-sampling step')] = 1

    def __init__(
        self,
        threshold: float = 0.0,
        band: int = 0,
        sample_step: int = 1,
    ) -> None:
        self.threshold = threshold
        self.band = band
        self.sample_step = sample_step

    def convert(
        self,
        source: np.ndarray,
        crs: str = 'EPSG:4326',
    ) -> FeatureSet:
        """
        Convert raster data to point features.

        Parameters
        ----------
        source : np.ndarray
            Input raster array.  2D ``(rows, cols)`` or 3D
            ``(bands, rows, cols)``.
        crs : str
            CRS for the output FeatureSet.

        Returns
        -------
        FeatureSet
            Point features at pixel locations exceeding threshold.
            Each feature has ``'value'``, ``'row'``, and ``'col'``
            properties.
        """
        if source.ndim == 3:
            data = source[self.band]
        elif source.ndim == 2:
            data = source
        else:
            raise ValueError(
                f"RasterToPoints expects 2D or 3D array, got {source.ndim}D"
            )

        features = []
        step = self.sample_step
        idx = 0
        for r in range(0, data.shape[0], step):
            for c in range(0, data.shape[1], step):
                val = float(data[r, c])
                if val >= self.threshold:
                    features.append(Feature(
                        geometry=Point(c, r),
                        properties={
                            'value': val,
                            'row': r,
                            'col': c,
                        },
                        feature_id=str(idx),
                    ))
                    idx += 1

        return FeatureSet(features=features, crs=crs)


@processor_version('1.0.0')
@processor_tags(
    category=ProcessorCategory.ANALYZE,
    description='Rasterize feature geometries into a numpy array.',
)
class Rasterize:
    """
    Rasterize feature geometries into a numpy array.

    Burns feature geometries into a raster array.  Each pixel that falls
    within a feature geometry receives either the feature's property value
    (``value_field``) or a fixed ``burn_value``.

    Parameters
    ----------
    value_field : str, optional
        Property field to use for pixel values.  If None, uses
        ``burn_value``.
    fill_value : float
        Background fill value.  Default 0.0.
    burn_value : float
        Value to burn for each feature when ``value_field`` is None.
        Default 1.0.
    """

    def __init__(
        self,
        value_field: str = None,
        fill_value: float = 0.0,
        burn_value: float = 1.0,
    ) -> None:
        self.value_field = value_field
        self.fill_value = fill_value
        self.burn_value = burn_value

    def convert(
        self,
        features: FeatureSet,
        shape: tuple,
    ) -> np.ndarray:
        """
        Rasterize features into a numpy array.

        Uses a simple pixel-center-in-geometry test.  For production
        use with large arrays, consider using rasterio.features.

        Parameters
        ----------
        features : FeatureSet
            Input feature set.
        shape : tuple
            Output array shape as ``(rows, cols)``.

        Returns
        -------
        np.ndarray
            Rasterized array with shape ``(rows, cols)``.
        """
        rows, cols = shape
        result = np.full((rows, cols), self.fill_value, dtype=np.float64)

        from shapely import prepare

        for feat in features:
            if self.value_field is not None:
                val = feat.properties.get(self.value_field, self.burn_value)
            else:
                val = self.burn_value

            geom = feat.geometry
            prepare(geom)

            # Get bounds to limit iteration
            minx, miny, maxx, maxy = geom.bounds
            r_start = max(0, int(miny))
            r_end = min(rows, int(maxy) + 1)
            c_start = max(0, int(minx))
            c_end = min(cols, int(maxx) + 1)

            for r in range(r_start, r_end):
                for c in range(c_start, c_end):
                    pt = Point(c, r)
                    if geom.contains(pt):
                        result[r, c] = val

        return result
