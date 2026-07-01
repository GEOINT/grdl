# -*- coding: utf-8 -*-
"""
BANDSB / BANDSA multispectral band TRE parsers.

Two entry points per TRE:

* ``parse_<name>(node)`` -- takes an ``<tre>``
  :class:`xml.etree.ElementTree.Element` from GDAL's ``xml:TRE``
  metadata domain (see :mod:`grdl.IO.eo._tre_xml`).  This is the
  **authoritative** path: GDAL has already decoded the BANDSB
  binary fields (IEEE-754 scale factors and the 4-byte existence
  mask) into text.
* ``parse_<name>_cedata(s)`` -- takes the raw fixed-width CEDATA
  string.  Best-effort: the BANDSB CEDATA layout embeds binary
  bytes (``SCALE_FACTOR``, ``ADDITIVE_FACTOR``, ``EXISTENCE_MASK``)
  that may not survive transport as text; the existence mask is
  recovered via latin-1 byte mapping when possible, and the parser
  returns ``None`` whenever the mask is undecodable or selects
  per-band fields this parser does not model.  Both CEDATA parsers
  return ``None`` on any parse failure and never raise.

Wavelength fields are normalized to micrometers using the TRE's
``WAVE_LENGTH_UNIT``: ``U`` (micrometers) verbatim; ``W``
(wavenumber, cm^-1) via ``lambda_um = 1e4 / nu`` and, for widths,
the first-order ``d_lambda = 1e4 * d_nu / nu**2``.

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
2026-06-09

Modified
--------
2026-06-09
"""

# Standard library
from typing import List, Optional
from xml.etree import ElementTree as ET

# GRDL internal
from grdl.IO.eo._tre_xml import (
    _optional_float,
    _optional_str,
    _repeated_field_values,
    _required_int,
)
from grdl.IO.models.eo_band import (
    BANDSABand,
    BANDSAMetadata,
    BANDSBBand,
    BANDSBMetadata,
)


# ===================================================================
# Wavelength unit normalization
# ===================================================================


def _wavelength_to_um(
    value: Optional[float], unit: Optional[str],
) -> Optional[float]:
    """Convert a BANDSB wavelength value to micrometers.

    ``U`` (or absent unit) is treated as micrometers; ``W`` is
    wavenumber in cm^-1 and converts via ``lambda_um = 1e4 / nu``.
    """
    if value is None:
        return None
    if unit == 'W':
        return 1.0e4 / value if value > 0 else None
    return value


def _width_to_um(
    width: Optional[float], center_raw: Optional[float],
    unit: Optional[str],
) -> Optional[float]:
    """Convert a BANDSB spectral width (FWHM) to micrometers.

    For wavenumber units the first-order conversion
    ``d_lambda = 1e4 * d_nu / nu**2`` (centered at the band's raw
    center wavenumber) is used.
    """
    if width is None:
        return None
    if unit == 'W':
        if center_raw is None or center_raw <= 0:
            return None
        return 1.0e4 * width / (center_raw * center_raw)
    return width


def _value_at(values: List[str], index: int) -> Optional[str]:
    """Return the stripped value at ``index`` or None when absent."""
    if index >= len(values):
        return None
    v = values[index].strip()
    return v if v else None


def _float_or_none(raw: Optional[str]) -> Optional[float]:
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _int_or_none(raw: Optional[str]) -> Optional[int]:
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


# ===================================================================
# BANDSB -- xml:TRE path (authoritative)
# ===================================================================


def parse_bandsb(node: ET.Element) -> Optional[BANDSBMetadata]:
    """Parse a ``<tre name="BANDSB">`` element.

    Field names follow GDAL's NITF TRE spec (which mirrors
    STDI-0002): top-level ``COUNT``, ``RADIOMETRIC_QUANTITY``,
    ``RADIOMETRIC_QUANTITY_UNIT``, ``WAVE_LENGTH_UNIT``; per-band
    repeated-group fields ``BANDID``, ``BAD_BAND``, ``NIIRS``,
    ``FOCAL_LEN``, ``CWAVE``, ``FWHM``, ``NOM_WAVE``, ``LBOUND``,
    ``UBOUND``.  Per-band fields gated off by the existence mask
    are simply absent from the XML and surface as ``None``.

    Parameters
    ----------
    node : xml.etree.ElementTree.Element
        ``<tre>`` element from the ``xml:TRE`` domain.

    Returns
    -------
    BANDSBMetadata or None
        ``None`` when the node is not a BANDSB TRE or required
        fields are missing or malformed.
    """
    if node.get('name') != 'BANDSB':
        return None

    try:
        count = _required_int(node, 'COUNT')
        if count < 0:
            return None

        unit = _optional_str(node, 'WAVE_LENGTH_UNIT')

        band_ids = _repeated_field_values(node, 'BANDID')
        bad_bands = _repeated_field_values(node, 'BAD_BAND')
        niirs_vals = _repeated_field_values(node, 'NIIRS')
        focal_vals = _repeated_field_values(node, 'FOCAL_LEN')
        cwaves = _repeated_field_values(node, 'CWAVE')
        fwhms = _repeated_field_values(node, 'FWHM')
        nom_waves = _repeated_field_values(node, 'NOM_WAVE')
        lbounds = _repeated_field_values(node, 'LBOUND')
        ubounds = _repeated_field_values(node, 'UBOUND')

        bands: List[BANDSBBand] = []
        for i in range(count):
            cwave_raw = _float_or_none(_value_at(cwaves, i))
            fwhm_raw = _float_or_none(_value_at(fwhms, i))
            nom_raw = _float_or_none(_value_at(nom_waves, i))
            lb_raw = _float_or_none(_value_at(lbounds, i))
            ub_raw = _float_or_none(_value_at(ubounds, i))

            bands.append(BANDSBBand(
                band_id=_value_at(band_ids, i),
                bad_band=_int_or_none(_value_at(bad_bands, i)),
                niirs=_float_or_none(_value_at(niirs_vals, i)),
                focal_length=_int_or_none(_value_at(focal_vals, i)),
                center_wavelength_um=_wavelength_to_um(cwave_raw, unit),
                fwhm_um=_width_to_um(fwhm_raw, cwave_raw, unit),
                nominal_wavelength_um=_wavelength_to_um(nom_raw, unit),
                lower_bound_um=_wavelength_to_um(lb_raw, unit),
                upper_bound_um=_wavelength_to_um(ub_raw, unit),
            ))

        return BANDSBMetadata(
            band_count=count,
            radiometric_quantity=_optional_str(
                node, 'RADIOMETRIC_QUANTITY'),
            radiometric_quantity_unit=_optional_str(
                node, 'RADIOMETRIC_QUANTITY_UNIT'),
            wave_length_unit=unit,
            bands=bands,
        )
    except (ValueError, TypeError):
        return None


# ===================================================================
# BANDSB -- CEDATA path (best-effort)
# ===================================================================

# Existence-mask bits whose fields this parser knows how to cursor
# past or extract (GDAL / STDI-0002 bit numbering, bit 31 = MSB).
# Any set bit outside this collection selects per-band binary or
# auxiliary fields we do not model -- the parser bails with None.
_BANDSB_SUPPORTED_MASK = sum(
    1 << b for b in (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19)
)


def parse_bandsb_cedata(s: str) -> Optional[BANDSBMetadata]:
    """Parse a raw BANDSB CEDATA string (best-effort).

    The BANDSB fixed-width layout embeds binary bytes: two IEEE-754
    floats (``SCALE_FACTOR``, ``ADDITIVE_FACTOR``) and the 4-byte
    ``EXISTENCE_MASK``.  Binary content frequently does not survive
    GDAL's text metadata transport, so the ``xml:TRE`` path
    (:func:`parse_bandsb`) is authoritative; this parser handles the
    common case defensively.  The mask is recovered by latin-1 byte
    mapping; returning ``None`` on binary-mask files that were
    mangled in transit is expected behavior.

    Returns
    -------
    BANDSBMetadata or None
        ``None`` on any parse failure (truncation, undecodable
        existence mask, or mask bits selecting unsupported fields).
    """
    try:
        v = s
        pos = 0

        def read(n: int) -> str:
            nonlocal pos
            if pos + n > len(v):
                raise ValueError('truncated BANDSB CEDATA')
            out = v[pos:pos + n]
            pos += n
            return out

        def read_str(n: int) -> Optional[str]:
            raw = read(n).strip()
            return raw if raw else None

        def read_opt_float(n: int) -> Optional[float]:
            raw = read(n).strip()
            if not raw or raw == '-':
                return None
            return float(raw)

        def read_opt_int(n: int) -> Optional[int]:
            raw = read(n).strip()
            if not raw:
                return None
            return int(raw)

        count = int(read(5))                      # COUNT
        if count < 0:
            return None
        radiometric_quantity = read_str(24)       # RADIOMETRIC_QUANTITY
        radiometric_unit = read_str(1)            # RADIOMETRIC_QUANTITY_UNIT
        read(4)                                   # SCALE_FACTOR (binary)
        read(4)                                   # ADDITIVE_FACTOR (binary)
        read(8)                                   # ROW_GSD + unit
        read(8)                                   # COL_GSD + unit
        read(8)                                   # SPT_RESP_ROW + unit
        read(8)                                   # SPT_RESP_COL + unit
        read(48)                                  # DATA_FLD_1

        mask_bytes = read(4).encode('latin-1')    # EXISTENCE_MASK (binary)
        if len(mask_bytes) != 4:
            return None
        mask = int.from_bytes(mask_bytes, 'big')
        if mask & ~_BANDSB_SUPPORTED_MASK:
            return None

        def bit(b: int) -> bool:
            return bool(mask & (1 << b))

        if bit(31):
            read(24)                              # RADIOMETRIC_ADJ_SURFACE
            read(4)                               # ATMOSPHERIC_ADJ (binary)
        if bit(30):
            read(7)                               # DIAMETER
        if bit(29):
            read(32)                              # DATA_FLD_2

        unit = read_str(1) if bit(24) else None   # WAVE_LENGTH_UNIT

        bands: List[BANDSBBand] = []
        for _ in range(count):
            band_id = read_str(50) if bit(28) else None
            bad_band = read_opt_int(1) if bit(27) else None
            niirs = read_opt_float(3) if bit(26) else None
            focal = read_opt_int(5) if bit(25) else None
            cwave_raw = read_opt_float(7) if bit(24) else None
            fwhm_raw = read_opt_float(7) if bit(23) else None
            if bit(22):
                read(7)                           # FWHM_UNC (not stored)
            nom_raw = read_opt_float(7) if bit(21) else None
            if bit(20):
                read(7)                           # NOM_WAVE_UNC (not stored)
            lb_raw = ub_raw = None
            if bit(19):
                lb_raw = read_opt_float(7)        # LBOUND
                ub_raw = read_opt_float(7)        # UBOUND

            bands.append(BANDSBBand(
                band_id=band_id,
                bad_band=bad_band,
                niirs=niirs,
                focal_length=focal,
                center_wavelength_um=_wavelength_to_um(cwave_raw, unit),
                fwhm_um=_width_to_um(fwhm_raw, cwave_raw, unit),
                nominal_wavelength_um=_wavelength_to_um(nom_raw, unit),
                lower_bound_um=_wavelength_to_um(lb_raw, unit),
                upper_bound_um=_wavelength_to_um(ub_raw, unit),
            ))

        return BANDSBMetadata(
            band_count=count,
            radiometric_quantity=radiometric_quantity,
            radiometric_quantity_unit=radiometric_unit,
            wave_length_unit=unit,
            bands=bands,
        )
    except (ValueError, TypeError, IndexError):
        return None


# ===================================================================
# BANDSA -- xml:TRE path
# ===================================================================


def parse_bandsa(node: ET.Element) -> Optional[BANDSAMetadata]:
    """Parse a ``<tre name="BANDSA">`` element.

    Field names follow STDI-0002: top-level ``ROW_SPACING``,
    ``ROW_SPACING_UNITS``, ``COL_SPACING``, ``COL_SPACING_UNITS``,
    ``FOCAL_LENGTH``, ``BANDCOUNT``; per-band repeated-group fields
    ``BANDPEAK``, ``BANDLBOUND``, ``BANDUBOUND``, ``BANDWIDTH``,
    ``BANDCALDRK``, ``BANDCALINC``, ``BANDRESP``, ``BANDASD``,
    ``BANDGSD``.

    Returns
    -------
    BANDSAMetadata or None
        ``None`` when the node is not a BANDSA TRE or required
        fields are missing or malformed.
    """
    if node.get('name') != 'BANDSA':
        return None

    try:
        count = _required_int(node, 'BANDCOUNT')
        if count < 0:
            return None

        peaks = _repeated_field_values(node, 'BANDPEAK')
        lbounds = _repeated_field_values(node, 'BANDLBOUND')
        ubounds = _repeated_field_values(node, 'BANDUBOUND')
        widths = _repeated_field_values(node, 'BANDWIDTH')
        cal_darks = _repeated_field_values(node, 'BANDCALDRK')
        cal_incs = _repeated_field_values(node, 'BANDCALINC')
        resps = _repeated_field_values(node, 'BANDRESP')
        asds = _repeated_field_values(node, 'BANDASD')
        gsds = _repeated_field_values(node, 'BANDGSD')

        bands: List[BANDSABand] = []
        for i in range(count):
            bands.append(BANDSABand(
                peak_response=_float_or_none(_value_at(peaks, i)),
                lower_bound=_float_or_none(_value_at(lbounds, i)),
                upper_bound=_float_or_none(_value_at(ubounds, i)),
                bandwidth=_float_or_none(_value_at(widths, i)),
                cal_dark_value=_float_or_none(_value_at(cal_darks, i)),
                cal_increment=_float_or_none(_value_at(cal_incs, i)),
                response=_float_or_none(_value_at(resps, i)),
                asd=_float_or_none(_value_at(asds, i)),
                gsd=_float_or_none(_value_at(gsds, i)),
            ))

        return BANDSAMetadata(
            row_gsd=_optional_float(node, 'ROW_SPACING'),
            row_gsd_units=_optional_str(node, 'ROW_SPACING_UNITS'),
            col_gsd=_optional_float(node, 'COL_SPACING'),
            col_gsd_units=_optional_str(node, 'COL_SPACING_UNITS'),
            focal_length=_optional_float(node, 'FOCAL_LENGTH'),
            band_count=count,
            bands=bands,
        )
    except (ValueError, TypeError):
        return None


# ===================================================================
# BANDSA -- CEDATA path
# ===================================================================


def parse_bandsa_cedata(s: str) -> Optional[BANDSAMetadata]:
    """Parse a raw BANDSA CEDATA string.

    Fixed-width layout per STDI-0002::

        ROW_SPACING(7) ROW_SPACING_UNITS(1)
        COL_SPACING(7) COL_SPACING_UNITS(1)
        FOCAL_LENGTH(6) BANDCOUNT(4)
        [BANDPEAK(5) BANDLBOUND(5) BANDUBOUND(5) BANDWIDTH(5)
         BANDCALDRK(6) BANDCALINC(5) BANDRESP(5) BANDASD(5)
         BANDGSD(5)] * BANDCOUNT

    Blank (all-space) fields are stored as ``None``.

    Returns
    -------
    BANDSAMetadata or None
        ``None`` on any parse failure.
    """
    try:
        v = s
        pos = 0

        def read(n: int) -> str:
            nonlocal pos
            if pos + n > len(v):
                raise ValueError('truncated BANDSA CEDATA')
            out = v[pos:pos + n]
            pos += n
            return out

        def read_str(n: int) -> Optional[str]:
            raw = read(n).strip()
            return raw if raw else None

        def read_opt_float(n: int) -> Optional[float]:
            raw = read(n).strip()
            if not raw or raw == '-':
                return None
            return float(raw)

        row_gsd = read_opt_float(7)               # ROW_SPACING
        row_units = read_str(1)                   # ROW_SPACING_UNITS
        col_gsd = read_opt_float(7)               # COL_SPACING
        col_units = read_str(1)                   # COL_SPACING_UNITS
        focal = read_opt_float(6)                 # FOCAL_LENGTH
        count = int(read(4))                      # BANDCOUNT
        if count < 0:
            return None

        bands: List[BANDSABand] = []
        for _ in range(count):
            bands.append(BANDSABand(
                peak_response=read_opt_float(5),  # BANDPEAK
                lower_bound=read_opt_float(5),    # BANDLBOUND
                upper_bound=read_opt_float(5),    # BANDUBOUND
                bandwidth=read_opt_float(5),      # BANDWIDTH
                cal_dark_value=read_opt_float(6),  # BANDCALDRK
                cal_increment=read_opt_float(5),  # BANDCALINC
                response=read_opt_float(5),       # BANDRESP
                asd=read_opt_float(5),            # BANDASD
                gsd=read_opt_float(5),            # BANDGSD
            ))

        return BANDSAMetadata(
            row_gsd=row_gsd,
            row_gsd_units=row_units,
            col_gsd=col_gsd,
            col_gsd_units=col_units,
            focal_length=focal,
            band_count=count,
            bands=bands,
        )
    except (ValueError, TypeError, IndexError):
        return None
