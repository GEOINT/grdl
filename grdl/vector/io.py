# -*- coding: utf-8 -*-
"""
Vector I/O - Read and write vector feature data.

Provides ``VectorReader`` and ``VectorWriter`` for reading/writing
vector feature data.  GeoJSON is supported natively without dependencies.
Other formats (Shapefile, GeoPackage, etc.) require optional geopandas.

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
import json
from pathlib import Path
from typing import Optional, Union

# GRDL internal
from grdl.vector.models import FeatureSet

# Supported native extensions (no geopandas required)
_NATIVE_EXTENSIONS = {'.geojson', '.json'}

# Extensions that require geopandas/fiona
_GEOPANDAS_EXTENSIONS = {
    '.shp', '.gpkg', '.kml', '.gml', '.fgb', '.parquet',
}


class VectorReader:
    """
    Reader for vector feature data files.

    GeoJSON files are read natively using the standard library.
    Other formats are read via optional geopandas.

    Examples
    --------
    >>> reader = VectorReader()
    >>> features = reader.read('data.geojson')
    >>> len(features)
    42
    """

    @staticmethod
    def can_read(path: Union[str, Path]) -> bool:
        """
        Check if a file can be read by this reader.

        Parameters
        ----------
        path : str or Path
            File path to check.

        Returns
        -------
        bool
            True if the file extension is recognized.
        """
        ext = Path(path).suffix.lower()
        return ext in _NATIVE_EXTENSIONS | _GEOPANDAS_EXTENSIONS

    @staticmethod
    def read(
        path: Union[str, Path],
        crs: Optional[str] = None,
    ) -> FeatureSet:
        """
        Read vector features from a file.

        Parameters
        ----------
        path : str or Path
            Path to the vector data file.
        crs : str, optional
            Override CRS for the resulting FeatureSet.  If None,
            uses the CRS from the file or defaults to ``'EPSG:4326'``.

        Returns
        -------
        FeatureSet

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file format is not supported.
        ImportError
            If a non-GeoJSON format is requested and geopandas is
            not installed.
        """
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        ext = filepath.suffix.lower()

        if ext in _NATIVE_EXTENSIONS:
            return _read_geojson(filepath, crs)
        elif ext in _GEOPANDAS_EXTENSIONS:
            return _read_via_geopandas(filepath, crs)
        else:
            raise ValueError(
                f"Unsupported vector file format: {ext!r}. "
                f"Supported: {sorted(_NATIVE_EXTENSIONS | _GEOPANDAS_EXTENSIONS)}"
            )


class VectorWriter:
    """
    Writer for vector feature data files.

    GeoJSON files are written natively using the standard library.
    Other formats are written via optional geopandas.

    Examples
    --------
    >>> writer = VectorWriter()
    >>> writer.write(features, 'output.geojson')
    """

    @staticmethod
    def write(
        features: FeatureSet,
        path: Union[str, Path],
        driver: Optional[str] = None,
    ) -> None:
        """
        Write features to a file.

        Parameters
        ----------
        features : FeatureSet
            Feature set to write.
        path : str or Path
            Output file path.
        driver : str, optional
            Output driver name (e.g., ``'GeoJSON'``, ``'GPKG'``).
            If None, inferred from file extension.

        Raises
        ------
        ValueError
            If the file format is not supported.
        ImportError
            If a non-GeoJSON format is requested and geopandas is
            not installed.
        """
        filepath = Path(path)
        ext = filepath.suffix.lower()

        if ext in _NATIVE_EXTENSIONS:
            _write_geojson(features, filepath)
        elif ext in _GEOPANDAS_EXTENSIONS:
            _write_via_geopandas(features, filepath, driver)
        else:
            raise ValueError(
                f"Unsupported vector file format: {ext!r}. "
                f"Supported: {sorted(_NATIVE_EXTENSIONS | _GEOPANDAS_EXTENSIONS)}"
            )


# -----------------------------------------------------------------
# Native GeoJSON I/O
# -----------------------------------------------------------------

def _read_geojson(
    filepath: Path,
    crs: Optional[str],
) -> FeatureSet:
    """Read a GeoJSON file natively."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    fs = FeatureSet.from_geojson(data)
    if crs is not None:
        fs.crs = crs
    return fs


def _write_geojson(features: FeatureSet, filepath: Path) -> None:
    """Write a GeoJSON file natively."""
    geojson = features.to_geojson()
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(geojson, f, indent=2)


# -----------------------------------------------------------------
# Geopandas-backed I/O
# -----------------------------------------------------------------

def _read_via_geopandas(
    filepath: Path,
    crs: Optional[str],
) -> FeatureSet:
    """Read a vector file via geopandas."""
    try:
        import geopandas as gpd
    except ImportError:
        raise ImportError(
            f"Reading {filepath.suffix} files requires geopandas. "
            f"Install with: pip install geopandas"
        )
    gdf = gpd.read_file(str(filepath))
    fs = FeatureSet.from_geodataframe(gdf)
    if crs is not None:
        fs.crs = crs
    return fs


def _write_via_geopandas(
    features: FeatureSet,
    filepath: Path,
    driver: Optional[str],
) -> None:
    """Write a vector file via geopandas."""
    try:
        import geopandas as gpd
    except ImportError:
        raise ImportError(
            f"Writing {filepath.suffix} files requires geopandas. "
            f"Install with: pip install geopandas"
        )
    gdf = features.to_geodataframe()
    kwargs = {}
    if driver is not None:
        kwargs['driver'] = driver
    filepath.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(str(filepath), **kwargs)
