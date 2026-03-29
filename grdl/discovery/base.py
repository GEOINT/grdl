# -*- coding: utf-8 -*-
"""
Discovery Base - Plugin ABC and registry for data discovery services.

Defines the ``DiscoveryPlugin`` abstract base class for extensible data
discovery and the ``PluginRegistry`` for managing registered plugins.

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
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List


class DiscoveryPlugin(ABC):
    """Abstract base class for data discovery services.

    Subclass this to create plugins that discover imagery from local
    directories, remote catalogs, STAC APIs, or custom archives.  The
    ``PluginRegistry`` manages instances and makes them accessible to
    the scanner and web UI.

    Attributes
    ----------
    name : str
        Human-readable plugin name (e.g., ``'Sentinel-1 SLC Catalog'``).
    description : str
        Short description of what this plugin discovers.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable plugin name."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Short description of what this plugin discovers."""
        ...

    @abstractmethod
    def discover(self, **kwargs: Any) -> List[Path]:
        """Discover files or products.

        Parameters
        ----------
        **kwargs
            Plugin-specific parameters (e.g., ``bbox``, ``date_range``,
            ``search_path``).

        Returns
        -------
        List[Path]
            Local file paths to discovered products.
        """
        ...

    @abstractmethod
    def get_config_schema(self) -> Dict[str, Any]:
        """Return a JSON-compatible schema for plugin configuration.

        The schema describes the keyword arguments accepted by
        :meth:`discover` so that the UI can render appropriate input
        forms.

        Returns
        -------
        dict
            JSON-compatible dict describing accepted parameters.
        """
        ...


class PluginRegistry:
    """Manages registered :class:`DiscoveryPlugin` instances.

    Parameters
    ----------
    None
    """

    def __init__(self) -> None:
        self._plugins: Dict[str, DiscoveryPlugin] = {}

    def register(self, plugin: DiscoveryPlugin) -> None:
        """Register a plugin.

        Parameters
        ----------
        plugin : DiscoveryPlugin
            Plugin instance to register.

        Raises
        ------
        TypeError
            If *plugin* is not a ``DiscoveryPlugin``.
        ValueError
            If a plugin with the same name is already registered.
        """
        if not isinstance(plugin, DiscoveryPlugin):
            raise TypeError(
                f"Expected DiscoveryPlugin, got {type(plugin).__name__}"
            )
        if plugin.name in self._plugins:
            raise ValueError(
                f"Plugin already registered: {plugin.name!r}"
            )
        self._plugins[plugin.name] = plugin

    def get(self, name: str) -> DiscoveryPlugin:
        """Retrieve a registered plugin by name.

        Parameters
        ----------
        name : str
            Plugin name.

        Returns
        -------
        DiscoveryPlugin

        Raises
        ------
        KeyError
            If no plugin with the given name is registered.
        """
        if name not in self._plugins:
            raise KeyError(
                f"No plugin registered with name {name!r}. "
                f"Available: {sorted(self._plugins.keys())}"
            )
        return self._plugins[name]

    def list_plugins(self) -> List[Dict[str, str]]:
        """List all registered plugins.

        Returns
        -------
        List[dict]
            Each dict contains ``'name'`` and ``'description'`` keys.
        """
        return [
            {'name': p.name, 'description': p.description}
            for p in self._plugins.values()
        ]

    def __len__(self) -> int:
        return len(self._plugins)

    def __contains__(self, name: str) -> bool:
        return name in self._plugins
