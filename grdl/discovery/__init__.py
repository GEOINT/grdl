# -*- coding: utf-8 -*-
"""
Discovery Module - Metadata scanning, cataloging, and data discovery.

Provides fast metadata-only scanning of imagery files, an in-memory
catalog with filtering and GeoJSON export, a plugin system for
extensible data discovery, and synthetic test data generation.

Core Classes
------------
- ``MetadataScanner`` — Parallel file/directory scanning via ``open_any()``
- ``ScanResult`` — Complete scan record with typed metadata reference
- ``LocalCatalog`` — In-memory catalog with filtering, stats, GeoJSON
- ``DiscoveryPlugin`` — ABC for data discovery plugins
- ``PluginRegistry`` — Plugin lifecycle management
- ``DataSynthesizer`` — Synthetic test imagery generation
- ``GRDLCatalogPlugin`` — Bridge GRDL catalogs to the plugin system

Usage
-----
    >>> from grdl.discovery import MetadataScanner, LocalCatalog
    >>>
    >>> scanner = MetadataScanner()
    >>> results = scanner.scan_directory('/data/imagery')
    >>>
    >>> catalog = LocalCatalog()
    >>> catalog.add_batch(results)
    >>> sar_items = catalog.filter(modality='SAR')
    >>> geojson = catalog.to_geojson(sar_items)

Author
------
Duane Smalley, PhD
170194430+DDSmalls@users.noreply.github.com

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

from grdl.discovery.base import DiscoveryPlugin, PluginRegistry
from grdl.discovery.scanner import MetadataScanner, ScanResult, compute_beam_footprint
from grdl.discovery.catalog import LocalCatalog
from grdl.discovery.synthesizer import DataSynthesizer
from grdl.discovery.plugins import GRDLCatalogPlugin

__all__ = [
    'DiscoveryPlugin',
    'PluginRegistry',
    'MetadataScanner',
    'ScanResult',
    'compute_beam_footprint',
    'LocalCatalog',
    'DataSynthesizer',
    'GRDLCatalogPlugin',
]

__version__ = '0.1.0'
