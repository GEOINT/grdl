# -*- coding: utf-8 -*-
"""
Collection Validation - Verify multi-channel SAR inputs belong together.

Checks that the polarization channels of a collection come from the same
collect (matching collector, time, scene center, grid and geometry) and
provides a reciprocity (HV vs VH) test for quad-pol data.

Thresholds are configurable defaults, not standards; tune them to the
sensor being processed.

Author
------
Jason Fritz
43161141+stryder-vtx@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-10-07

Modified
--------
2026-10-07
"""

# Standard library
import math
import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, List, Optional, Sequence, Tuple

# Third-party
import numpy as np

# GRDL internal
from grdl.IO.sar.get_antenna_polarization import get_antenna_polarization

_LINEAR = frozenset({'HH', 'HV', 'VH', 'VV'})
_CIRCULAR = frozenset({'RH', 'RV', 'RR', 'RL', 'LH', 'LV', 'LR', 'LL'})
_EARTH_RADIUS_M = 6371008.8

# SICD Tx/Rcv polarization tokens mapped to the one-letter codes used in labels.
_TX_RCV_CODES = {
    'H': 'H', 'V': 'V', 'X': 'X', 'Y': 'Y', 'RHC': 'R', 'LHC': 'L',
}


class CollectionValidationError(ValueError):
    """Raised when a collection fails validation in strict use."""


class ReciprocityError(ValueError):
    """Raised when HV and VH are not reciprocal and ``strict`` is set."""


@dataclass
class CollectionIssue:
    """One finding from collection validation.

    Attributes
    ----------
    level : str
        ``'error'`` or ``'warning'``.
    code : str
        Short machine-readable identifier.
    message : str
        Human-readable description.
    """

    level: str
    code: str
    message: str


@dataclass
class CollectionReport:
    """Result of :func:`validate_collection`.

    Attributes
    ----------
    polarizations : list of str
        Channel polarization labels in channel order.
    mode : str
        ``'quad'``, ``'triple'``, ``'dual'``, ``'compact'``, ``'single'``
        or ``'other'``.
    issues : list of CollectionIssue
        Errors and warnings.
    """

    polarizations: List[str] = field(default_factory=list)
    mode: str = 'other'
    issues: List[CollectionIssue] = field(default_factory=list)

    @property
    def errors(self) -> List[CollectionIssue]:
        return [i for i in self.issues if i.level == 'error']

    @property
    def warnings(self) -> List[CollectionIssue]:
        return [i for i in self.issues if i.level == 'warning']

    @property
    def ok(self) -> bool:
        """True when there are no errors (warnings allowed)."""
        return not self.errors

    @property
    def is_quad_pol(self) -> bool:
        return self.mode == 'quad'

    def raise_if_invalid(self) -> None:
        """Raise ``CollectionValidationError`` listing every error."""
        if self.errors:
            raise CollectionValidationError(
                '; '.join(f'[{i.code}] {i.message}' for i in self.errors)
            )

    def _add(self, level: str, code: str, message: str) -> None:
        self.issues.append(CollectionIssue(level, code, message))


@dataclass
class ReciprocityResult:
    """Outcome of :func:`check_reciprocity`.

    Attributes
    ----------
    applicable : bool
        False when the data has no HV/VH pair; other fields are then None.
    passed : bool
        True when all thresholds are met (or not applicable).
    correlation : float
        Magnitude of the normalized complex correlation of HV and VH.
    phase_deg : float
        Phase of that correlation; ~0 for reciprocal data.
    amplitude_ratio_db : float
        ``10*log10(<|HV|^2> / <|VH|^2>)``.
    n_pixels : int
        Pixels used.
    reason : str
        Explanation of a failure or non-applicability.
    """

    applicable: bool = True
    passed: bool = True
    correlation: Optional[float] = None
    phase_deg: Optional[float] = None
    amplitude_ratio_db: Optional[float] = None
    n_pixels: int = 0
    reason: str = ''


# ---------------------------------------------------------------------------
# Polarization helpers
# ---------------------------------------------------------------------------

def classify_polarizations(polarizations: Sequence[str]) -> str:
    """Classify a polarization set.

    Parameters
    ----------
    polarizations : sequence of str
        Labels such as ``'HH'`` (case-insensitive).

    Returns
    -------
    str
        ``'quad'`` (HH, HV, VH, VV), ``'triple'`` (three linear),
        ``'dual'`` (two linear), ``'compact'`` (any circular-transmit
        label), ``'single'`` or ``'other'``.
    """
    pols = {p.upper() for p in polarizations}
    if pols & _CIRCULAR:
        return 'compact'
    if not pols or not pols <= _LINEAR:
        return 'other'
    return {4: 'quad', 3: 'triple', 2: 'dual', 1: 'single'}[len(pols)]


# ---------------------------------------------------------------------------
# SICD polarization from metadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SICDPolarization:
    """Transmit and receive polarization of one SICD image.

    Attributes
    ----------
    tx, rcv : str or None
        One-letter codes (``'H'``, ``'V'``, ``'R'``, ``'L'``, ``'X'``,
        ``'Y'``); None when the metadata value is not a recognized code.
    source : str
        Metadata field the value came from.
    """

    tx: Optional[str]
    rcv: Optional[str]
    source: str

    @property
    def label(self) -> Optional[str]:
        """Two-letter ``tx + rcv`` label, or None if either is unknown."""
        return self.tx + self.rcv if self.tx and self.rcv else None


def parse_tx_rcv(
    value: Optional[str],
) -> Optional[Tuple[Optional[str], Optional[str]]]:
    """Split a SICD ``'Tx:Rcv'`` string (e.g. ``'H:V'``) into codes.

    A bare two-letter value such as ``'HV'`` is accepted. Returns None for
    values with no recognized code (``'OTHER'``, ``'SEQUENCE'``, ...).
    """
    if not value:
        return None
    text = value.strip().upper()
    if ':' in text:
        tx, rcv = (p.strip() for p in text.split(':', 1))
    elif len(text) == 2:
        tx, rcv = text[0], text[1]
    else:
        return None
    tx_code, rcv_code = _TX_RCV_CODES.get(tx), _TX_RCV_CODES.get(rcv)
    if tx_code is None and rcv_code is None:
        return None
    return tx_code, rcv_code


def _get(obj: Any, *path: str) -> Any:
    for name in path:
        obj = getattr(obj, name, None)
        if obj is None:
            return None
    return obj


def _pol_from_proc(meta: Any) -> Optional[SICDPolarization]:
    """``ImageFormation/TxRcvPolarizationProc``: the processed image itself."""
    value = _get(meta, 'image_formation', 'tx_rcv_polarization_proc')
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            return None
        value = value[0]
    parsed = parse_tx_rcv(value)
    if parsed is None:
        return None
    return SICDPolarization(
        *parsed, source='image_formation.tx_rcv_polarization_proc',
    )


def _pol_from_rcv_channels(meta: Any) -> Optional[SICDPolarization]:
    """The ``RadarCollection/RcvChannels`` entry this image was formed from.

    ``ImageFormation/RcvChanProc/ChanIndex`` names the channel (1-based, as
    ``ChanParameters@index``). Without it, only a single-channel file is
    unambiguous.
    """
    channels = _get(meta, 'radar_collection', 'rcv_channels')
    if not channels:
        return None
    indices = _get(meta, 'image_formation', 'rcv_chan_proc', 'chan_indices')
    if isinstance(indices, int):
        indices = [indices]
    if indices:
        if len(indices) != 1:
            return None
        wanted = indices[0]
        chosen = next((c for c in channels if c.index == wanted), None)
        if chosen is None and 1 <= wanted <= len(channels):
            chosen = channels[wanted - 1]
        source = f'radar_collection.rcv_channels[{wanted}] (chan_indices)'
    elif len(channels) == 1:
        chosen, source = channels[0], 'radar_collection.rcv_channels[0]'
    else:
        return None
    if chosen is None:
        return None
    parsed = parse_tx_rcv(chosen.tx_rcv_polarization)
    return None if parsed is None else SICDPolarization(*parsed, source=source)


def sicd_polarization(meta: Any) -> Optional[SICDPolarization]:
    """Resolve the Tx/Rcv polarization of one SICD image from its metadata.

    Prefers ``image_formation.tx_rcv_polarization_proc`` (the polarization
    of the formed image), then the receive channel named by
    ``rcv_chan_proc.chan_indices``, then the only receive channel.

    Parameters
    ----------
    meta : SICDMetadata

    Returns
    -------
    SICDPolarization or None
        None when the metadata does not identify a single polarization,
        e.g. several receive channels and no ``chan_indices``.
    """
    return _pol_from_proc(meta) or _pol_from_rcv_channels(meta)


# ---------------------------------------------------------------------------
# Collection validation
# ---------------------------------------------------------------------------

def _parse_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    text = value.strip().rstrip('Zz')
    text = re.sub(r'(\.\d{6})\d+', r'\1', text)
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _scp_distance_m(a: Any, b: Any) -> Optional[float]:
    sa = getattr(getattr(a, 'geo_data', None), 'scp', None)
    sb = getattr(getattr(b, 'geo_data', None), 'scp', None)
    if sa is None or sb is None:
        return None
    if sa.ecf is not None and sb.ecf is not None:
        return math.dist(
            (sa.ecf.x, sa.ecf.y, sa.ecf.z), (sb.ecf.x, sb.ecf.y, sb.ecf.z),
        )
    if sa.llh is not None and sb.llh is not None:
        p1, p2 = math.radians(sa.llh.lat), math.radians(sb.llh.lat)
        dlon = math.radians(sb.llh.lon - sa.llh.lon)
        dlat = p2 - p1
        h = (math.sin(dlat / 2) ** 2
             + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2)
        return 2 * _EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(h)))
    return None


def _antenna_label(meta: Any) -> Optional[str]:
    """Two-letter label from ``get_antenna_polarization``, or None.

    None when the metadata lacks the fields it needs, the file holds
    several channels, or the transmit polarization is not a single code
    (e.g. ``SEQUENCE``).
    """
    try:
        pol, _ = get_antenna_polarization(meta)
    except (ValueError, IndexError, KeyError, TypeError, AttributeError):
        return None
    if isinstance(pol, list) and len(pol) == 1:
        pol = pol[0]
    if not isinstance(pol, str) or len(pol) != 2:
        return None
    return pol.upper()


def validate_sicd_collection(
    metadata: Sequence[Any],
    polarizations: Optional[Sequence[str]] = None,
    *,
    max_time_delta_s: float = 1.0,
    max_scp_distance_m: float = 25.0,
    max_spacing_rel_diff: float = 1e-3,
    max_angle_diff_deg: float = 0.5,
) -> CollectionReport:
    """Check that per-polarization SICD files describe one collect.

    Parameters
    ----------
    metadata : sequence of SICDMetadata
        One per channel.
    polarizations : sequence of str, optional
        Channel labels, same order. When given, each is checked against
        the polarization in the SICD metadata, taken both from
        :func:`sicd_polarization` and from
        :func:`~grdl.IO.sar.get_antenna_polarization.get_antenna_polarization`
        (which maps rotated ``X``/``Y`` receive polarizations onto
        ``H``/``V``); a label matching either is accepted. When omitted,
        labels are taken from the metadata and an unresolvable channel is
        labelled ``'UNKNOWN'``.
    max_time_delta_s : float
        Allowed spread of ``timeline.collect_start``.
    max_scp_distance_m : float
        Allowed distance between scene center points.
    max_spacing_rel_diff : float
        Allowed relative difference of row/column sample spacing.
    max_angle_diff_deg : float
        Allowed difference of graze and Doppler cone angles.

    Returns
    -------
    CollectionReport
        Dimension, collector, side-of-track, time, SCP, spacing, angle and
        polarization-label mismatches are errors; a differing ``core_name``
        and missing or ambiguous metadata are warnings.
    """
    resolved = [sicd_polarization(m) for m in metadata]
    antenna = [_antenna_label(m) for m in metadata]
    if polarizations is None:
        pols = [
            (p.label if p and p.label else a or 'UNKNOWN')
            for p, a in zip(resolved, antenna)
        ]
    else:
        pols = [p.upper() for p in polarizations]
    rep = CollectionReport(polarizations=pols,
                           mode=classify_polarizations(pols))
    if len(metadata) != len(pols):
        rep._add('error', 'count_mismatch',
                 f'{len(metadata)} metadata objects for {len(pols)} labels')
        return rep

    for i, (label, found, ant) in enumerate(zip(pols, resolved, antenna)):
        raw = found.label if found is not None else None
        known = {x for x in (raw, ant) if x}
        if not known:
            rep._add('warning', 'polarization_not_in_metadata',
                     f'channel {i}: SICD metadata does not identify a single '
                     f'Tx/Rcv polarization; label {label!r} is unverified')
        elif polarizations is not None and label not in ('UNKNOWN', 'OTHER') \
                and label not in known:
            sources = []
            if raw:
                sources.append(f'{found.source} says {raw}')
            if ant:
                sources.append(f'get_antenna_polarization says {ant}')
            rep._add('error', 'polarization_label_mismatch',
                     f'channel {i} is labelled {label} but '
                     + ' and '.join(sources))
        # Rotated platforms legitimately differ (X:X in the metadata is HH
        # after antenna mapping), so only compare plain H/V/R/L labels.
        if raw and ant and raw != ant and not ({'X', 'Y'} & set(raw)):
            rep._add('warning', 'antenna_polarization_disagrees',
                     f'channel {i}: {found.source} says {raw} but '
                     f'get_antenna_polarization says {ant}; check '
                     f'radar_collection.tx_polarization')
        channel = _pol_from_rcv_channels(metadata[i])
        if found is not None and channel is not None \
                and found.label and channel.label \
                and found.label != channel.label:
            rep._add('warning', 'polarization_fields_disagree',
                     f'channel {i}: {found.source} says {found.label} but '
                     f'{channel.source} says {channel.label}')

    dupes = sorted({p for p in pols if pols.count(p) > 1})
    if dupes:
        rep._add('error', 'duplicate_polarization',
                 f'polarization(s) {dupes} appear more than once')
    if 'UNKNOWN' in pols or 'OTHER' in pols:
        rep._add('warning', 'unknown_polarization',
                 f'polarization labels {pols} include an undetermined entry')
    if rep.mode == 'triple':
        rep._add('warning', 'partial_quad',
                 'three linear polarizations; reciprocity cannot be checked')

    ref = metadata[0]
    dims = {(m.rows, m.cols) for m in metadata}
    if len(dims) > 1:
        rep._add('error', 'dimension_mismatch',
                 f'image sizes differ: {sorted(dims)}')

    def _differs(path: str, values: list, code: str, level: str) -> None:
        present = [v for v in values if v is not None]
        if len(present) < len(values):
            rep._add('warning', f'missing_{path}',
                     f'{path} absent from some files; check skipped')
        if len(set(present)) > 1:
            rep._add(level, code, f'{path} differs: {sorted(set(present))}')

    _differs('collector_name',
             [_get(m, 'collection_info', 'collector_name') for m in metadata],
             'collector_mismatch', 'error')
    _differs('side_of_track',
             [_get(m, 'scpcoa', 'side_of_track') for m in metadata],
             'side_of_track_mismatch', 'error')
    cores = [_get(m, 'collection_info', 'core_name') for m in metadata]
    if len({c for c in cores if c is not None}) > 1:
        rep._add('warning', 'core_name_differs',
                 f'core_name differs: {sorted({c for c in cores if c})}')

    times = [_parse_time(_get(m, 'timeline', 'collect_start'))
             for m in metadata]
    if any(t is None for t in times):
        rep._add('warning', 'missing_collect_start',
                 'collect_start absent or unparseable; time check skipped')
    else:
        spread = (max(times) - min(times)).total_seconds()
        if spread > max_time_delta_s:
            rep._add('error', 'time_mismatch',
                     f'collect_start spread {spread:.3f} s exceeds '
                     f'{max_time_delta_s} s')

    for i, m in enumerate(metadata[1:], start=1):
        d = _scp_distance_m(ref, m)
        if d is None:
            rep._add('warning', 'missing_scp',
                     f'scene center point missing for channel {i}')
        elif d > max_scp_distance_m:
            rep._add('error', 'scp_mismatch',
                     f'channel {i} SCP is {d:.1f} m from channel 0 '
                     f'(limit {max_scp_distance_m} m)')

    for axis in ('row', 'col'):
        ss = [_get(m, 'grid', axis, 'ss') for m in metadata]
        if all(v is not None for v in ss):
            if (max(ss) - min(ss)) > max_spacing_rel_diff * max(ss):
                rep._add('error', f'{axis}_spacing_mismatch',
                         f'grid {axis} sample spacing differs: {ss}')
    for name in ('graze_ang', 'doppler_cone_ang'):
        vals = [_get(m, 'scpcoa', name) for m in metadata]
        if all(v is not None for v in vals) \
                and max(vals) - min(vals) > max_angle_diff_deg:
            rep._add('error', f'{name}_mismatch',
                     f'{name} differs by more than {max_angle_diff_deg} deg')
    return rep


def validate_collection(reader: Any, **kwargs: Any) -> CollectionReport:
    """Validate the channels of an opened multi-channel reader.

    Multi-file readers (those exposing ``per_file_metadata``, e.g.
    ``SICDCollectionReader``) get the full cross-file checks. Single-file
    products (NISAR, Sentinel-1, BIOMASS) share one metadata block, so only
    the channel structure is checked.

    Parameters
    ----------
    reader : ImageReader
        Reader with ``metadata.channel_metadata``.
    **kwargs
        Thresholds forwarded to :func:`validate_sicd_collection`.

    Returns
    -------
    CollectionReport
    """
    meta = reader.metadata
    channels = getattr(meta, 'channel_metadata', None) or []
    pols = [c.polarization or c.name for c in channels]

    per_file = getattr(meta, 'per_file_metadata', None)
    if per_file:
        return validate_sicd_collection(per_file, pols, **kwargs)

    rep = CollectionReport(polarizations=[p.upper() for p in pols],
                           mode=classify_polarizations(pols))
    if not channels:
        rep._add('error', 'no_channels', 'metadata has no channel_metadata')
        return rep
    if len(set(rep.polarizations)) != len(rep.polarizations):
        rep._add('error', 'duplicate_polarization',
                 f'duplicate polarizations in {rep.polarizations}')
    if len(channels) > 1 and getattr(meta, 'axis_order', None) != 'CYX':
        rep._add('error', 'axis_order',
                 f'{len(channels)} channels but axis_order is '
                 f'{getattr(meta, "axis_order", None)!r}, expected CYX')
    return rep


# ---------------------------------------------------------------------------
# Reciprocity
# ---------------------------------------------------------------------------

def check_reciprocity(
    hv: np.ndarray,
    vh: np.ndarray,
    *,
    min_correlation: float = 0.8,
    max_phase_deg: float = 30.0,
    max_amplitude_ratio_db: float = 1.0,
    power_percentile: float = 50.0,
    strict: bool = False,
) -> ReciprocityResult:
    """Test whether HV and VH satisfy monostatic reciprocity (HV = VH).

    Statistics use only pixels whose combined power exceeds the given
    percentile, so noise-dominated pixels do not mask a reciprocal pair.
    Jointly zero pixels are excluded; an entirely zero sample fails and
    follows the same warning/strict policy as a threshold failure.

    Parameters
    ----------
    hv, vh : np.ndarray
        Finite complex arrays of equal shape. NaN and Inf are invalid.
    min_correlation : float
        Minimum magnitude of the normalized complex correlation.
    max_phase_deg : float
        Maximum absolute correlation phase.
    max_amplitude_ratio_db : float
        Maximum absolute HV/VH power ratio in dB.
    power_percentile : float
        Percentile of combined power below which pixels are ignored.
    strict : bool
        Raise ``ReciprocityError`` on failure instead of warning.

    Returns
    -------
    ReciprocityResult

    Raises
    ------
    ValueError
        If shapes differ or either input contains NaN or Inf.
    ReciprocityError
        If the check fails and ``strict=True``.
    """
    hv = np.asarray(hv)
    vh = np.asarray(vh)
    if hv.shape != vh.shape:
        raise ValueError(f'hv {hv.shape} and vh {vh.shape} shapes differ')
    if not np.isfinite(hv).all() or not np.isfinite(vh).all():
        raise ValueError('HV and VH inputs must contain only finite values')

    power = (np.abs(hv.astype(np.complex128)) ** 2
             + np.abs(vh.astype(np.complex128)) ** 2)
    valid = power > 0
    if not valid.any():
        msg = 'HV/VH reciprocity check failed: no non-zero pixels'
        if strict:
            raise ReciprocityError(msg)
        warnings.warn(msg, stacklevel=2)
        return ReciprocityResult(
            passed=False, reason='no non-zero pixels')
    mask = valid & (power >= np.percentile(power[valid], power_percentile))
    a = hv[mask].astype(np.complex128)
    b = vh[mask].astype(np.complex128)

    pa, pb = np.mean(np.abs(a) ** 2), np.mean(np.abs(b) ** 2)
    cross = np.mean(a * np.conj(b))
    denom = math.sqrt(pa * pb)
    corr = float(abs(cross) / denom) if denom > 0 else 0.0
    phase = float(np.degrees(np.angle(cross)))
    ratio = float(10 * math.log10(pa / pb)) if pa > 0 and pb > 0 else math.inf

    failures = []
    if corr < min_correlation:
        failures.append(f'correlation {corr:.3f} < {min_correlation}')
    if abs(phase) > max_phase_deg:
        failures.append(f'phase {phase:.1f} deg exceeds {max_phase_deg}')
    if abs(ratio) > max_amplitude_ratio_db:
        failures.append(
            f'HV/VH power ratio {ratio:.2f} dB exceeds {max_amplitude_ratio_db}')

    result = ReciprocityResult(
        passed=not failures, correlation=corr, phase_deg=phase,
        amplitude_ratio_db=ratio, n_pixels=int(mask.sum()),
        reason='; '.join(failures),
    )
    if failures:
        msg = f'HV/VH reciprocity check failed: {result.reason}'
        if strict:
            raise ReciprocityError(msg)
        warnings.warn(msg, stacklevel=2)
    return result


def check_reciprocity_reader(
    reader: Any,
    *,
    chip_size: int = 512,
    n_chips: int = 3,
    **kwargs: Any,
) -> ReciprocityResult:
    """Run :func:`check_reciprocity` on chips sampled from a reader.

    Reads ``n_chips`` windows along the image diagonal instead of the full
    scene.

    Parameters
    ----------
    reader : ImageReader
        Multi-channel reader returning ``(C, rows, cols)`` chips with
        ``metadata.channel_metadata`` polarization labels.
    chip_size : int
        Chip edge length in pixels (clipped to the image).
    n_chips : int
        Number of chips.
    **kwargs
        Forwarded to :func:`check_reciprocity`.

    Returns
    -------
    ReciprocityResult
        ``applicable=False`` unless both HV and VH channels exist.
    """
    channels = getattr(reader.metadata, 'channel_metadata', None) or []
    labels = [(c.polarization or c.name or '').upper() for c in channels]
    if 'HV' not in labels or 'VH' not in labels:
        return ReciprocityResult(
            applicable=False, reason='no HV/VH channel pair')
    i_hv, i_vh = labels.index('HV'), labels.index('VH')

    rows, cols = reader.metadata.rows, reader.metadata.cols
    h, w = min(chip_size, rows), min(chip_size, cols)
    hv_parts, vh_parts = [], []
    for k in range(n_chips):
        f = (k + 1) / (n_chips + 1)
        r0 = int(f * (rows - h))
        c0 = int(f * (cols - w))
        cube = reader.read_chip(r0, r0 + h, c0, c0 + w)
        if cube.ndim != 3 or cube.shape[0] != len(labels):
            raise ValueError(
                f'reader returned shape {cube.shape}; expected '
                f'({len(labels)}, rows, cols)'
            )
        hv_parts.append(cube[i_hv].ravel())
        vh_parts.append(cube[i_vh].ravel())
    return check_reciprocity(
        np.concatenate(hv_parts), np.concatenate(vh_parts), **kwargs,
    )
