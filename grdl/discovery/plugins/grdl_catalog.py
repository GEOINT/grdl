# -*- coding: utf-8 -*-
"""
GRDL Catalog Plugin - Bridge GRDL CatalogInterface to DiscoveryPlugin.

Wraps any GRDL catalog (Sentinel1SLCCatalog, NISARCatalog, etc.) as a
``DiscoveryPlugin`` so that local discovery via ``discover_local()`` is
accessible through the plugin registry.

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
2026-03-29

Modified
--------
2026-03-29
"""

# Standard library
from pathlib import Path
from typing import Any, Dict, List, Type

from grdl.discovery.base import DiscoveryPlugin


class GRDLCatalogPlugin(DiscoveryPlugin):
    """Bridge a GRDL ``CatalogInterface`` to the discovery plugin system.

    Parameters
    ----------
    catalog_cls : type
        A GRDL catalog class (e.g., ``Sentinel1SLCCatalog``).
    search_path : str or Path
        Root directory for local file discovery.
    sensor_name : str
        Human-readable sensor name for display.
    """

    def __init__(
        self,
        catalog_cls: Type,
        search_path: str | Path,
        sensor_name: str,
    ) -> None:
        self._catalog_cls = catalog_cls
        self._search_path = Path(search_path)
        self._sensor_name = sensor_name
        self._catalog = None

    @property
    def name(self) -> str:
        """Human-readable plugin name."""
        return f"GRDL {self._sensor_name}"

    @property
    def description(self) -> str:
        """Short description."""
        return (
            f"Discover {self._sensor_name} products in "
            f"{self._search_path} via GRDL catalog"
        )

    def discover(self, **kwargs: Any) -> List[Path]:
        """Discover local files via the GRDL catalog.

        Parameters
        ----------
        **kwargs
            Forwarded to the catalog constructor or discover method.
            Common keys: ``search_path`` (override default path).

        Returns
        -------
        List[Path]
            Discovered file paths.
        """
        search_path = kwargs.pop('search_path', self._search_path)
        search_path = Path(search_path)

        if self._catalog is None:
            self._catalog = self._catalog_cls(str(search_path))

        paths = self._catalog.discover_local(update_db=False)

        return [Path(p) for p in paths]

    def get_config_schema(self) -> Dict[str, Any]:
        """Return configuration schema.

        Returns
        -------
        dict
            JSON schema for plugin configuration.
        """
        return {
            'search_path': {
                'type': 'string',
                'description': f'Root directory for {self._sensor_name} products',
                'default': str(self._search_path),
            },
        }
