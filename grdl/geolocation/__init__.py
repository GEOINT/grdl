# -*- coding: utf-8 -*-
"""
Geolocation Module - Coordinate transformations for geospatial imagery.

Provides interfaces and implementations for transforming between image pixel
coordinates and geographic coordinates (latitude/longitude).

Key Classes
-----------
- Geolocation: Abstract base class for coordinate transformations
- NoGeolocation: Fallback for imagery without geolocation
- ChipGeolocation: Offset wrapper for sub-region (chip) coordinates
- create_geolocation: Factory that auto-selects the right Geolocation subclass

Usage
-----
Create a geolocation object from an imagery reader. The ``image_to_latlon``
and ``latlon_to_image`` methods use the **(N, M) stacked ndarray** convention:
scalar inputs return a 1-D array compatible with tuple unpacking, and batch
inputs accept an (N, 2) array and return an (N, 3) or (N, 2) stacked result.

    >>> from grdl.IO import open_biomass
    >>> from grdl.geolocation.sar.gcp import GCPGeolocation
    >>> import numpy as np
    >>>
    >>> with open_biomass('path/to/biomass_product') as reader:
    >>>     geo = GCPGeolocation(
    ...         reader.metadata['gcps'],
    ...         (reader.metadata['rows'], reader.metadata['cols']),
    ...     )
    >>>
    >>>     # Scalar — tuple unpacking works
    >>>     lat, lon, height = geo.image_to_latlon(1000, 500)
    >>>
    >>>     # Batch — (N, 2) stacked input, (N, 3) stacked output
    >>>     pixels = np.array([[100, 400],
    ...                        [200, 500],
    ...                        [300, 600]])           # (3, 2) [row, col]
    >>>     geo_pts = geo.image_to_latlon(pixels)     # (3, 3) [lat, lon, h]

Modules
-------
- base: Abstract base classes
- sar: SAR-specific geolocation (slant range, SICD, GRD)
- eo: EO-specific geolocation (geocoded rasters)
- utils: Utility functions

Dependencies
------------
scipy

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
2026-01-30

Modified
--------
2026-03-27
"""

from grdl.geolocation.base import Geolocation, NoGeolocation
from grdl.geolocation.chip import ChipGeolocation
from grdl.geolocation.coordinates import (
    geodetic_to_ecef,
    ecef_to_geodetic,
    geodetic_to_enu,
    enu_to_geodetic,
    meters_per_degree,
)
from grdl.geolocation.sar.gcp import GCPGeolocation
from grdl.geolocation.sar.nisar import NISARGeolocation
from grdl.geolocation.sar.sicd import SICDGeolocation
from grdl.geolocation.sar.sidd import SIDDGeolocation
from grdl.geolocation.sar.sentinel1_slc import Sentinel1SLCGeolocation
from grdl.geolocation.eo.affine import AffineGeolocation
from grdl.geolocation.eo.rpc import RPCGeolocation
from grdl.geolocation.eo.rsm import RSMGeolocation
from grdl.geolocation.elevation.base import ElevationModel
from grdl.geolocation.elevation.constant import ConstantElevation
from grdl.geolocation.projection import (
    COAProjection,
    image_to_ground_hae,
    image_to_ground_dem,
    image_to_ground_plane,
    ground_to_image,
    wgs84_norm,
)

__all__ = [
    'Geolocation',
    'NoGeolocation',
    'ChipGeolocation',
    'create_geolocation',
    'geodetic_to_ecef',
    'ecef_to_geodetic',
    'geodetic_to_enu',
    'enu_to_geodetic',
    'meters_per_degree',
    'GCPGeolocation',
    'NISARGeolocation',
    'SICDGeolocation',
    'SIDDGeolocation',
    'Sentinel1SLCGeolocation',
    'AffineGeolocation',
    'RPCGeolocation',
    'RSMGeolocation',
    'ElevationModel',
    'ConstantElevation',
    'COAProjection',
    'image_to_ground_hae',
    'image_to_ground_dem',
    'image_to_ground_plane',
    'ground_to_image',
    'wgs84_norm',
]

__version__ = '0.2.0'


def create_geolocation(reader: object, **kwargs) -> Geolocation:
    """Create a Geolocation from any supported ImageReader.

    Dispatches on the reader's metadata type to select the correct
    geolocation class and calls its ``from_reader`` factory method.
    Extra keyword arguments are forwarded to the factory.

    Parameters
    ----------
    reader : ImageReader
        Any GRDL imagery reader with populated ``metadata``.
    **kwargs
        Forwarded to the geolocation constructor or ``from_reader()``.
        Common options:

        - ``backend`` : str — SICD projection backend
          (``'native'``, ``'sarpy'``, ``'sarkit'``).
        - ``refine`` : bool — SIDD R/Rdot refinement toggle.
        - ``dem_path`` : str — DEM path for elevation.
        - ``geoid_path`` : str — Geoid model path.
        - ``interpolation`` : int — DEM spline order (1/3/5).

    Returns
    -------
    Geolocation
        A configured geolocation object matching the reader's format.

    Raises
    ------
    TypeError
        If the reader's metadata type is not recognized.

    Examples
    --------
    >>> from grdl.IO.sar import SICDReader
    >>> from grdl.geolocation import create_geolocation
    >>>
    >>> reader = SICDReader('image.nitf')
    >>> geo = create_geolocation(reader)
    >>> lat, lon, h = geo.image_to_latlon(500, 1000)
    """
    meta = reader.metadata

    # Lazy imports to avoid circular dependencies and allow partial
    # installation (not all readers/metadata models may be available).
    try:
        from grdl.IO.models.sicd import SICDMetadata
        if isinstance(meta, SICDMetadata):
            return SICDGeolocation.from_reader(reader, **kwargs)
    except ImportError:
        pass

    try:
        from grdl.IO.models.sidd import SIDDMetadata
        if isinstance(meta, SIDDMetadata):
            return SIDDGeolocation.from_reader(reader, **kwargs)
    except ImportError:
        pass

    try:
        from grdl.IO.models.nisar import NISARMetadata
        if isinstance(meta, NISARMetadata):
            return NISARGeolocation.from_reader(reader, **kwargs)
    except ImportError:
        pass

    try:
        from grdl.IO.models.sentinel1_slc import Sentinel1SLCMetadata
        if isinstance(meta, Sentinel1SLCMetadata):
            return Sentinel1SLCGeolocation.from_reader(reader, **kwargs)
    except ImportError:
        pass

    try:
        from grdl.IO.models.eo_nitf import EONITFMetadata
        if isinstance(meta, EONITFMetadata):
            # Prefer RPC (more common), fall back to RSM
            if getattr(meta, 'rpc', None) is not None:
                return RPCGeolocation.from_reader(reader, **kwargs)
            if getattr(meta, 'rsm', None) is not None:
                return RSMGeolocation.from_reader(reader, **kwargs)
    except ImportError:
        pass

    try:
        from grdl.IO.models.biomass import BIOMASSMetadata
        if isinstance(meta, BIOMASSMetadata):
            gcps = getattr(meta, 'gcps', None)
            if gcps is not None:
                return GCPGeolocation(
                    gcps=gcps,
                    shape=(meta.rows, meta.cols),
                )
            # Geocoded BIOMASS — fall through to affine
    except ImportError:
        pass

    # Affine fallback: any geocoded raster with transform and CRS
    if getattr(meta, 'transform', None) is not None and getattr(meta, 'crs', None) is not None:
        return AffineGeolocation.from_reader(reader, **kwargs)

    # GCP fallback (e.g., dict-like metadata)
    gcps = getattr(meta, 'gcps', None)
    if gcps is not None:
        return GCPGeolocation(
            gcps=gcps,
            shape=(meta.rows, meta.cols),
        )

    raise TypeError(
        f"Cannot determine geolocation type for reader with metadata "
        f"type {type(meta).__name__}. Provide a geolocation object "
        f"directly instead."
    )