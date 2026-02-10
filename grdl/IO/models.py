# -*- coding: utf-8 -*-
"""
IO Models - Typed metadata containers for imagery readers.

Provides ``ImageMetadata``, a dataclass that stores universal image
metadata (format, rows, cols, dtype) as typed attributes while
supporting dict-like access for backward compatibility. Designed for
subclassing by sensor-specific metadata classes (e.g., ``SICDMetadata``,
``CPHDMetadata``) that add typed fields for their full native metadata.

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
2026-02-10

Modified
--------
2026-02-10
"""

# Standard library
from dataclasses import dataclass, field, fields as dc_fields
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class ImageMetadata:
    """Typed metadata for imagery read by GRDL IO readers.

    Provides typed attributes for universal and common metadata fields,
    with an ``extras`` dict for format-specific or sensor-specific
    fields. Supports dict-like access (``metadata['rows']``,
    ``'crs' in metadata``, ``metadata.get('bands')``) for backward
    compatibility with code that treats metadata as ``Dict[str, Any]``.

    Designed for subclassing: sensor-specific subclasses (e.g.,
    ``SICDMetadata``) can add typed fields that are automatically
    discovered by dict-like access methods via ``dataclasses.fields()``.

    Parameters
    ----------
    format : str
        Format identifier (e.g., ``'GeoTIFF'``, ``'SICD'``, ``'HDF5'``).
    rows : int
        Number of image rows (lines).
    cols : int
        Number of image columns (samples).
    dtype : str
        NumPy dtype string (e.g., ``'float32'``, ``'complex64'``).
    bands : int, optional
        Number of spectral bands. None for channel-based formats.
    crs : str, optional
        Coordinate reference system string (e.g., ``'EPSG:4326'``).
    nodata : float, optional
        No-data sentinel value.
    extras : Dict[str, Any]
        Format-specific metadata. All keys are accessible via
        dict-like access on this object.

    Examples
    --------
    >>> meta = ImageMetadata(
    ...     format='GeoTIFF', rows=1024, cols=2048, dtype='float32',
    ...     bands=3, crs='EPSG:4326',
    ...     extras={'transform': affine_obj, 'resolution': (10.0, 10.0)},
    ... )
    >>> meta.rows
    1024
    >>> meta['transform']
    Affine(...)
    >>> 'crs' in meta
    True
    >>> meta.get('nodata', -9999)
    -9999
    """

    # Universal fields (required)
    format: str
    rows: int
    cols: int
    dtype: str

    # Common optional fields
    bands: Optional[int] = None
    crs: Optional[str] = None
    nodata: Optional[float] = None

    # Format/sensor-specific catch-all
    extras: Dict[str, Any] = field(default_factory=dict)

    # ----------------------------------------------------------------
    # Dict-like access for backward compatibility
    # ----------------------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        """Access metadata by key, checking typed fields then extras.

        Parameters
        ----------
        key : str
            Metadata key.

        Returns
        -------
        Any
            Value for the key.

        Raises
        ------
        KeyError
            If key is not found in typed fields or extras.
        """
        for f in dc_fields(self):
            if f.name == key and f.name != 'extras':
                return getattr(self, key)
        if key in self.extras:
            return self.extras[key]
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set metadata by key, updating typed fields or extras.

        Parameters
        ----------
        key : str
            Metadata key.
        value : Any
            Value to set.
        """
        field_names = {f.name for f in dc_fields(self) if f.name != 'extras'}
        if key in field_names:
            setattr(self, key, value)
        else:
            self.extras[key] = value

    def __contains__(self, key: str) -> bool:
        """Check if key exists and has a non-None value.

        Returns ``False`` for typed fields that are ``None``, matching
        the behavior of a dict where absent keys return ``False``.

        Parameters
        ----------
        key : str
            Metadata key.

        Returns
        -------
        bool
        """
        for f in dc_fields(self):
            if f.name == key and f.name != 'extras':
                return getattr(self, key) is not None
        return key in self.extras

    def get(self, key: str, default: Any = None) -> Any:
        """Get value by key with a default, like ``dict.get()``.

        Parameters
        ----------
        key : str
            Metadata key.
        default : Any
            Default value if key not found. Default is None.

        Returns
        -------
        Any
        """
        try:
            val = self[key]
            if val is None:
                return default
            return val
        except KeyError:
            return default

    def keys(self) -> List[str]:
        """Return all available metadata keys.

        Returns typed fields with non-None values plus all extras keys.

        Returns
        -------
        List[str]
        """
        result = [
            f.name for f in dc_fields(self)
            if f.name != 'extras' and getattr(self, f.name) is not None
        ]
        result.extend(self.extras.keys())
        return result

    def values(self) -> List[Any]:
        """Return all available metadata values.

        Returns values for typed fields with non-None values plus all
        extras values, in the same order as ``keys()``.

        Returns
        -------
        List[Any]
        """
        return [self[k] for k in self.keys()]

    def items(self) -> List[tuple]:
        """Return all available metadata key-value pairs.

        Returns ``(key, value)`` tuples for typed fields with non-None
        values plus all extras, in the same order as ``keys()``.

        Returns
        -------
        List[tuple]
        """
        return [(k, self[k]) for k in self.keys()]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a flat dictionary.

        Typed fields with None values are excluded. Extras are merged
        into the top level.

        Returns
        -------
        Dict[str, Any]
        """
        result: Dict[str, Any] = {}
        for f in dc_fields(self):
            if f.name == 'extras':
                continue
            val = getattr(self, f.name)
            if val is not None:
                result[f.name] = val
        result.update(self.extras)
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ImageMetadata':
        """Construct from a plain dictionary.

        Typed field keys are extracted into attributes; remaining keys
        go into ``extras``. Works with subclasses via ``cls``.

        Parameters
        ----------
        d : Dict[str, Any]
            Metadata dictionary.

        Returns
        -------
        ImageMetadata
            New instance (or subclass instance if called on subclass).

        Raises
        ------
        KeyError
            If required fields are missing.
        """
        field_names = {f.name for f in dc_fields(cls) if f.name != 'extras'}
        typed_args: Dict[str, Any] = {}
        extras: Dict[str, Any] = {}
        for key, val in d.items():
            if key in field_names:
                typed_args[key] = val
            else:
                extras[key] = val
        return cls(**typed_args, extras=extras)

    def __iter__(self) -> Iterator[str]:
        """Iterate over available keys for ``dict(metadata)`` compat."""
        return iter(self.keys())

    def __len__(self) -> int:
        """Number of available metadata keys."""
        return len(self.keys())

    def __repr__(self) -> str:
        field_strs = []
        for f in dc_fields(self):
            if f.name == 'extras':
                if self.extras:
                    field_strs.append(f"extras=[{', '.join(self.extras.keys())}]")
                continue
            val = getattr(self, f.name)
            if val is not None:
                field_strs.append(f"{f.name}={val!r}")
        return f"{type(self).__name__}({', '.join(field_strs)})"
