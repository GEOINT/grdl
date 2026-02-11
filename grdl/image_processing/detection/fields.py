# -*- coding: utf-8 -*-
"""
Detection Data Dictionary - Standardized field names for detection attributes.

Provides a hierarchical naming system for sparse detection properties.
Detectors are strongly encouraged to use these standard names so that
downstream consumers, visualizations, and analytics can interpret
detection attributes without per-detector configuration.

Field names use **dot notation**: ``domain.attribute`` (e.g.,
``'sar.change_magnitude'``, ``'identity.label'``).  The ``Fields``
accessor class provides IDE-friendly constants:
``Fields.sar.CHANGE_MAGNITUDE`` evaluates to ``'sar.change_magnitude'``.

Domains
-------
physical
    Measurable physical quantities (length, area, RCS, speed, heading).
sar
    SAR phenomenological attributes (backscatter, coherence, polarimetric).
spectral
    Spectral phenomenological attributes (NDVI, reflectance, radiance).
volume
    Volumetric measurements (height, biomass, canopy cover).
identity
    Classification and identification (label, class ID, confidence).
trait
    Observable characteristics (color, shape, material, orientation).
temporal
    Time-related attributes (first/last seen, persistence, change type).
context
    Spatial context attributes (density, cluster, elevation, land cover).

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
2026-02-11

Modified
--------
2026-02-11
"""

from typing import Dict, List, Optional


class FieldDefinition:
    """Definition of a single field in the GRDL data dictionary.

    Parameters
    ----------
    name : str
        Hierarchical dotted name (e.g., ``'sar.change_magnitude'``).
    dtype : str
        Data type: ``'float'``, ``'int'``, ``'str'``, or ``'bool'``.
    description : str
        Human-readable description of the field.
    units : str, optional
        Physical units (e.g., ``'dB'``, ``'m'``).  None for dimensionless
        or non-numeric fields.
    """

    __slots__ = ('name', 'dtype', 'description', 'units', 'domain')

    def __init__(
        self,
        name: str,
        dtype: str,
        description: str,
        units: Optional[str] = None,
    ) -> None:
        self.name = name
        self.dtype = dtype
        self.description = description
        self.units = units
        self.domain = name.split('.')[0] if '.' in name else name

    def __repr__(self) -> str:
        parts = f"FieldDefinition({self.name!r}, {self.dtype!r}"
        if self.units is not None:
            parts += f", units={self.units!r}"
        return parts + ")"


# ---------------------------------------------------------------------------
# Data dictionary registry
# ---------------------------------------------------------------------------

def _build_dictionary() -> Dict[str, FieldDefinition]:
    """Build the comprehensive data dictionary."""
    _f = FieldDefinition
    entries = [
        # ---------------------------------------------------------------
        # physical -- Measurable physical quantities
        # ---------------------------------------------------------------
        _f('physical.length', 'float', 'Physical length', 'm'),
        _f('physical.width', 'float', 'Physical width', 'm'),
        _f('physical.height', 'float', 'Physical height', 'm'),
        _f('physical.area', 'float', 'Physical area', 'm^2'),
        _f('physical.perimeter', 'float', 'Physical perimeter', 'm'),
        _f('physical.heading', 'float', 'Heading angle', 'deg'),
        _f('physical.speed', 'float', 'Ground speed', 'm/s'),
        _f('physical.velocity_radial', 'float', 'Radial velocity', 'm/s'),
        _f('physical.rcs', 'float', 'Radar cross section', 'm^2'),
        _f('physical.rcs_db', 'float', 'Radar cross section in dB', 'dBsm'),
        _f('physical.temperature', 'float', 'Temperature', 'K'),
        _f('physical.emissivity_thermal', 'float', 'Thermal emissivity'),

        # ---------------------------------------------------------------
        # sar -- SAR phenomenological attributes
        # ---------------------------------------------------------------
        _f('sar.sigma0', 'float', 'Sigma naught', 'dB'),
        _f('sar.gamma0', 'float', 'Gamma naught', 'dB'),
        _f('sar.beta0', 'float', 'Beta naught', 'dB'),
        _f('sar.coherence', 'float', 'Interferometric coherence'),
        _f('sar.coherence_loss', 'float', 'Coherence loss ratio'),
        _f('sar.change_magnitude', 'float', 'Change magnitude', 'dB'),
        _f('sar.sublook_ratio', 'float', 'Sublook intensity ratio'),
        _f('sar.sublook_variance', 'float', 'Sublook intensity variance'),
        _f('sar.pol_entropy', 'float', 'Polarimetric entropy'),
        _f('sar.pol_alpha', 'float', 'Polarimetric alpha angle', 'deg'),
        _f('sar.pol_anisotropy', 'float', 'Polarimetric anisotropy'),
        _f('sar.pol_span', 'float', 'Polarimetric total power (span)', 'dB'),
        _f('sar.pol_pedestal_height', 'float', 'Polarimetric pedestal height'),
        _f('sar.scattering_type', 'str', 'Dominant scattering mechanism'),
        _f('sar.phase_difference', 'float', 'Interferometric phase difference', 'rad'),

        # ---------------------------------------------------------------
        # spectral -- Spectral phenomenological attributes
        # ---------------------------------------------------------------
        _f('spectral.ndvi', 'float', 'Normalized difference vegetation index'),
        _f('spectral.ndwi', 'float', 'Normalized difference water index'),
        _f('spectral.ndbi', 'float', 'Normalized difference built-up index'),
        _f('spectral.ndsi', 'float', 'Normalized difference snow index'),
        _f('spectral.evi', 'float', 'Enhanced vegetation index'),
        _f('spectral.reflectance', 'float', 'Surface reflectance'),
        _f('spectral.radiance', 'float', 'At-sensor radiance', 'W/m^2/sr/um'),
        _f('spectral.emissivity', 'float', 'Surface emissivity'),
        _f('spectral.band_ratio', 'float', 'Band ratio value'),
        _f('spectral.spectral_angle', 'float', 'Spectral angle distance', 'rad'),
        _f('spectral.brightness_temp', 'float', 'Brightness temperature', 'K'),
        _f('spectral.chlorophyll_index', 'float', 'Chlorophyll index'),

        # ---------------------------------------------------------------
        # volume -- Volumetric measurements
        # ---------------------------------------------------------------
        _f('volume.height', 'float', 'Above-ground height', 'm'),
        _f('volume.canopy_height', 'float', 'Canopy height', 'm'),
        _f('volume.biomass_density', 'float', 'Biomass density', 'Mg/ha'),
        _f('volume.biomass_agb', 'float', 'Above-ground biomass', 'Mg/ha'),
        _f('volume.canopy_cover', 'float', 'Fractional canopy cover'),
        _f('volume.leaf_area_index', 'float', 'Leaf area index'),
        _f('volume.tree_count', 'int', 'Tree count within geometry'),
        _f('volume.volume', 'float', 'Volumetric estimate', 'm^3'),

        # ---------------------------------------------------------------
        # identity -- Classification and identification
        # ---------------------------------------------------------------
        _f('identity.label', 'str', 'Semantic class label'),
        _f('identity.label_id', 'int', 'Numeric class identifier'),
        _f('identity.label_confidence', 'float', 'Classification confidence'),
        _f('identity.is_target', 'bool', 'Binary target indicator'),
        _f('identity.threat_level', 'str', 'Assessed threat level'),
        _f('identity.category', 'str', 'Broad category label'),
        _f('identity.subcategory', 'str', 'Refined subcategory label'),
        _f('identity.model_name', 'str', 'Classifier model name'),
        _f('identity.model_version', 'str', 'Classifier model version'),
        _f('identity.association_id', 'str', 'Cross-detection association ID'),

        # ---------------------------------------------------------------
        # trait -- Observable characteristics
        # ---------------------------------------------------------------
        _f('trait.color', 'str', 'Observed color'),
        _f('trait.shape', 'str', 'Shape descriptor'),
        _f('trait.material', 'str', 'Material composition'),
        _f('trait.orientation', 'float', 'Orientation angle', 'deg'),
        _f('trait.texture', 'str', 'Texture descriptor'),
        _f('trait.size_class', 'str', 'Size classification'),
        _f('trait.condition', 'str', 'Observed condition'),
        _f('trait.activity', 'str', 'Observed activity'),
        _f('trait.posture', 'str', 'Posture or configuration'),
        _f('trait.camouflage_index', 'float', 'Camouflage effectiveness index'),

        # ---------------------------------------------------------------
        # temporal -- Time-related attributes
        # ---------------------------------------------------------------
        _f('temporal.first_seen', 'str', 'ISO 8601 first observation'),
        _f('temporal.last_seen', 'str', 'ISO 8601 last observation'),
        _f('temporal.persistence', 'float', 'Temporal persistence score'),
        _f('temporal.revisit_count', 'int', 'Number of revisit observations'),
        _f('temporal.change_type', 'str', 'Type of temporal change'),
        _f('temporal.change_rate', 'float', 'Rate of change', '1/day'),
        _f('temporal.dwell_time', 'float', 'Dwell time at location', 's'),
        _f('temporal.activity_pattern', 'str', 'Temporal activity pattern'),

        # ---------------------------------------------------------------
        # context -- Spatial context attributes
        # ---------------------------------------------------------------
        _f('context.density', 'float', 'Spatial density of detections', '1/km^2'),
        _f('context.cluster_id', 'int', 'Cluster assignment identifier'),
        _f('context.nearest_neighbor', 'float', 'Nearest neighbor distance', 'm'),
        _f('context.nearest_road', 'float', 'Distance to nearest road', 'm'),
        _f('context.elevation', 'float', 'Terrain elevation', 'm'),
        _f('context.slope', 'float', 'Terrain slope', 'deg'),
        _f('context.land_cover', 'str', 'Land cover class'),
        _f('context.land_use', 'str', 'Land use class'),
    ]
    return {e.name: e for e in entries}


DATA_DICTIONARY: Dict[str, FieldDefinition] = _build_dictionary()
"""Comprehensive registry of standardized detection field definitions.

Maps dotted field names (e.g., ``'sar.change_magnitude'``) to their
``FieldDefinition``.  Detectors are strongly encouraged to use names
from this dictionary; non-dictionary names trigger a warning.
"""


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def lookup_field(name: str) -> Optional[FieldDefinition]:
    """Look up a field definition by name.

    Parameters
    ----------
    name : str
        Dotted field name (e.g., ``'sar.coherence'``).

    Returns
    -------
    FieldDefinition or None
        The definition if found, otherwise None.
    """
    return DATA_DICTIONARY.get(name)


def is_dictionary_field(name: str) -> bool:
    """Check whether a field name is in the data dictionary.

    Parameters
    ----------
    name : str
        Dotted field name.

    Returns
    -------
    bool
        True if the name exists in ``DATA_DICTIONARY``.
    """
    return name in DATA_DICTIONARY


def list_fields(domain: Optional[str] = None) -> List[FieldDefinition]:
    """List field definitions, optionally filtered by domain.

    Parameters
    ----------
    domain : str, optional
        If given, only return fields in this domain
        (e.g., ``'sar'``, ``'identity'``).

    Returns
    -------
    List[FieldDefinition]
        Matching field definitions, sorted by name.
    """
    if domain is None:
        return sorted(DATA_DICTIONARY.values(), key=lambda f: f.name)
    return sorted(
        [f for f in DATA_DICTIONARY.values() if f.domain == domain],
        key=lambda f: f.name,
    )


# ---------------------------------------------------------------------------
# Fields accessor class for IDE autocomplete
# ---------------------------------------------------------------------------

class _Domain:
    """Namespace for field name constants within a domain."""
    pass


class Fields:
    """Accessor for standard field names with IDE autocomplete.

    Usage::

        >>> Fields.sar.CHANGE_MAGNITUDE
        'sar.change_magnitude'
        >>> Fields.identity.LABEL
        'identity.label'

    Each domain is an attribute containing uppercase constants that
    resolve to the full dotted field name string.
    """

    physical = _Domain()
    sar = _Domain()
    spectral = _Domain()
    volume = _Domain()
    identity = _Domain()
    trait = _Domain()
    temporal = _Domain()
    context = _Domain()


# Populate Fields accessor from DATA_DICTIONARY at module load time.
for _name, _field in DATA_DICTIONARY.items():
    _domain_name, _attr_name = _name.split('.', 1)
    _domain_obj = getattr(Fields, _domain_name, None)
    if _domain_obj is not None:
        setattr(_domain_obj, _attr_name.upper(), _name)
