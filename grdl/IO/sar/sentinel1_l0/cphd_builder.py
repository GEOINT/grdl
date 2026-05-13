# -*- coding: utf-8 -*-
"""
Sentinel-1 CRSD → CPHD conversion with TOPS azimuth deramping.

Reads a Sentinel-1 CRSD file produced by :class:`Sentinel1L0ToCRSD`,
applies per-burst TOPS azimuth deramping, and writes a CPHD 1.0.1 file
with fully corrected PVP fields including:

* ``aFDOP`` ≈ 0 Hz (TOPS steering phase removed)
* ``SRPPos`` set to the burst-center nadir point (not the global IARP)
* ``SC0`` / ``SCSS`` consistent with the hardware ADC window (Fs)
* ``FX1`` / ``FX2`` from the per-pulse chirp start frequency (TXPSF)
* ``aFRR1`` / ``aFRR2`` from range chirp FM rate: ``2*f0/(c*LFMRate)`` /
  ``2/(c*LFMRate)`` where ``LFMRate = BWInst / TXmtMin``

Pipeline stage responsibilities
--------------------------------
Stage 1 (``crsd_writer.py``, SAFE → CRSD):
    ISP decode, FDBAQ decompression, range FFT.  The CRSDsar format
    specification (NGA.STND.0068-1) requires FX-domain signal storage,
    so the range FFT is applied before writing.

Stage 2 (this module, CRSD → CPHD):
    Orbit interpolation, burst boundary detection, per-burst TOPS
    azimuth deramping, zero-padding, and PVP formation.  The input
    signal is already FX-domain; no range FFT is applied here.

Stage 3 (image formation, CPHD → SICD):
    Azimuth compression via CSA (Chirp Scaling Algorithm) or RDA.

TOPS deramping
--------------
In TOPS/IW mode the antenna steers aft-to-fore across each burst at
rate ``k_psi`` (Hz/s in Doppler-frequency units).  The received signal
carries a quadratic azimuth phase:

    phi(eta) = pi * k_psi * (eta - eta_c)^2

where ``eta_c`` is the burst centre time.  Multiplying by
``exp(-j * pi * k_psi * (eta - eta_c)^2)`` removes this ramp so that
``aFDOP`` at the burst-centre SRPPos is ~ 0 Hz.

``k_psi`` is derived from a linear fit to the per-pulse geometric
Doppler centroid (computed from orbit position/velocity and the
burst-centre nadir ground point).  This is equivalent to the
"Doppler rate steering" approach in Azouz et al. 2023 and the ESA
TOPS processing handbook.

The deramp filter ``h(eta) = exp(-j*pi*k_psi*(eta-eta_c)^2)`` depends
only on slow-time (azimuth), not on range frequency.  It therefore
applies identically whether the signal is in time-domain or FX-domain
(the two operations commute).  Since the CRSD stores FX-domain data,
the deramp is a per-row scalar multiply with no range FFT required.

Dependencies
------------
sarkit, lxml

Author
------
Ava Courtney
courtney-ava@zai.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-05-13

Modified
--------
2026-05-13
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from lxml import etree

import sarkit.cphd

from grdl.geolocation.coordinates import ecef_to_geodetic, geodetic_to_ecef
import sarkit.wgs84

logger = logging.getLogger(__name__)

# Speed of light (m/s)
_C = 299_792_458.0

# CPHD namespace
_CPHD_NS = "http://api.nsgreg.nga.mil/schema/cphd/1.0.1"
_CPHD_NSMAP = {None: _CPHD_NS}


# ===================================================================
# XML helpers
# ===================================================================


def _sub(parent: etree._Element, tag: str, text: Optional[str] = None) -> etree._Element:
    """Append a namespaced child element."""
    child = etree.SubElement(parent, f"{{{_CPHD_NS}}}{tag}")
    if text is not None:
        child.text = text
    return child


def _sub_xyz(parent: etree._Element, tag: str, x: float, y: float, z: float) -> etree._Element:
    elem = _sub(parent, tag)
    _sub(elem, "X", f"{x:.12g}")
    _sub(elem, "Y", f"{y:.12g}")
    _sub(elem, "Z", f"{z:.12g}")
    return elem


def _sub_latlon(parent: etree._Element, tag: str, lat: float, lon: float) -> etree._Element:
    elem = _sub(parent, tag)
    _sub(elem, "Lat", f"{lat:.8g}")
    _sub(elem, "Lon", f"{lon:.8g}")
    return elem


def _sub_xy(parent: etree._Element, tag: str, x: float, y: float) -> etree._Element:
    elem = _sub(parent, tag)
    _sub(elem, "X", f"{x:.12g}")
    _sub(elem, "Y", f"{y:.12g}")
    return elem


# ===================================================================
# Orbit geometry helpers
# ===================================================================


def _nadir_ecef(pos: np.ndarray) -> np.ndarray:
    """Project platform ECEF position to the WGS-84 surface (h=0).

    Parameters
    ----------
    pos : ndarray, shape (3,)
        Platform ECEF position in metres.

    Returns
    -------
    ndarray, shape (3,)
        Ground point directly below the platform (h=0).
    """
    llh = ecef_to_geodetic(pos)    # (3,) → (3,) [lat, lon, hae]
    return geodetic_to_ecef(np.array([llh[0], llh[1], 0.0])).ravel()


def _doppler_centroid_array(
    positions: np.ndarray,
    velocities: np.ndarray,
    srp: np.ndarray,
    f_c: float,
) -> np.ndarray:
    """Compute geometric Doppler centroid for every pulse.

    Parameters
    ----------
    positions : ndarray, shape (N, 3)
        Platform ECEF positions.
    velocities : ndarray, shape (N, 3)
        Platform ECEF velocities.
    srp : ndarray, shape (3,)
        Scene reference point ECEF.
    f_c : float
        Carrier frequency in Hz.

    Returns
    -------
    ndarray, shape (N,)
        Per-pulse aFDOP = -2*f_c/c * dot(v, r_hat).
    """
    r_vecs = srp[np.newaxis, :] - positions                  # (N, 3)
    r_mags = np.linalg.norm(r_vecs, axis=1, keepdims=True)   # (N, 1)
    r_hats = r_vecs / r_mags                                  # (N, 3)
    return -(2.0 * f_c / _C) * np.einsum("ni,ni->n", velocities, r_hats)


def _fit_tops_steering_rate(
    eta_rel: np.ndarray,
    f_dc: np.ndarray,
) -> Tuple[float, float]:
    """Fit a linear model to the per-pulse Doppler centroid across a burst.

    Returns ``(k_psi, f_dc0)`` where:

    * ``k_psi`` [Hz/s] — TOPS azimuth steering rate (slope)
    * ``f_dc0`` [Hz]   — Doppler centroid at burst centre (intercept)

    A linear fit is valid for the typical IW burst duration (~0.82 s)
    where the non-linear term is < 1 Hz.

    Parameters
    ----------
    eta_rel : ndarray, shape (N,)
        Pulse times relative to burst centre (seconds).
    f_dc : ndarray, shape (N,)
        Geometric Doppler centroid per pulse (Hz).

    Returns
    -------
    (k_psi, f_dc0) : tuple of float
    """
    coeffs = np.polyfit(eta_rel, f_dc, 1)
    return float(coeffs[0]), float(coeffs[1])


# ===================================================================
# Burst boundary detection
# ===================================================================


def _detect_burst_boundaries(
    pvp: np.ndarray,
    ref_time_gps: float,
    pri_tolerance_factor: float = 2.0,
) -> List[Tuple[int, int]]:
    """Find burst start/end indices from PRI gaps in RcvStart.

    The CRSD PVP ``RcvStart`` field stores times as ``(integer, frac)``
    int/frac pairs.  Consecutive-pulse time differences that exceed
    ``pri_tolerance_factor × median_PRI`` indicate a burst gap.

    Parameters
    ----------
    pvp : ndarray
        Structured PVP array with ``RcvStart`` int/frac field.
    ref_time_gps : float
        Collection reference time in GPS seconds (for context only).
    pri_tolerance_factor : float
        Gaps larger than this multiple of the median PRI are treated as
        burst boundaries.

    Returns
    -------
    list of (start_idx, end_idx)
        Inclusive start, exclusive end, for each burst.
    """
    # Reconstruct absolute receive times from the IntFrac PVP field.
    # RcvStart is stored as I8 (integer) + F8 (fractional) at adjacent
    # byte offsets.  sarkit returns them as a two-element structured
    # array with fields [0] and [1].
    if pvp.dtype.names and "RcvStart" in pvp.dtype.names:
        # IntFrac: sarkit represents this as a nested dtype with fields
        # 'Int' (i8) and 'Frac' (f8).
        rcv_int = pvp["RcvStart"]["Int"].astype(np.float64)
        rcv_frac = pvp["RcvStart"]["Frac"].astype(np.float64)
        t = rcv_int + rcv_frac
    else:
        raise ValueError("PVP does not contain 'RcvStart' field")

    dt = np.diff(t)
    median_pri = float(np.median(dt[dt > 0])) if np.any(dt > 0) else 1e-3
    threshold = pri_tolerance_factor * median_pri

    gap_indices = np.where(dt > threshold)[0]
    # gap_indices[i] is the last index of burst i; burst i+1 starts at i+1
    boundaries = []
    start = 0
    for gap in gap_indices:
        boundaries.append((start, int(gap) + 1))
        start = int(gap) + 1
    boundaries.append((start, len(pvp)))
    return boundaries


# ===================================================================
# Main conversion class
# ===================================================================


class CRSDtoCPHD:
    """Convert a Sentinel-1 CRSD file to CPHD with TOPS deramping.

    Reads the CRSD written by :class:`~grdl.IO.sar.sentinel1_l0.crsd_writer.Sentinel1L0ToCRSD`,
    applies per-burst TOPS azimuth deramping (removing the steering
    phase ramp so that aFDOP ≈ 0), applies the range FFT to convert
    from time-domain to FX-domain signal, and writes a CPHD 1.0.1 file
    with corrected PVP fields.

    Parameters
    ----------
    crsd_path : str or Path
        Input CRSD file produced by ``Sentinel1L0ToCRSD``.
    output_path : str or Path
        Output CPHD file path.
    orbit_source : str, optional
        Orbit resolution strategy passed to
        :class:`~grdl.IO.sar.sentinel1_l0.orbit.OrbitResolver`.
        One of ``"auto"``, ``"download"``, ``"file"``, or
        ``"annotation"``.  Default ``"auto"``.
    orbit_file : str or Path, optional
        Local ``.EOF`` file.  Required when ``orbit_source='file'``.
    orbit_cache_dir : str or Path, optional
        Directory for caching downloaded orbit files.
    channel_id : str, optional
        CRSD channel identifier to convert (e.g. ``"IW1_VV"``).
        Defaults to the first channel in the CRSD.
    safe_path : str or Path, optional
        Path to the source ``.SAFE`` directory.  Required when
        ``orbit_source`` is ``"auto"`` or ``"download"`` (needed to
        download the matching orbit file).  If not given and the CRSD
        filename follows the GRDL naming convention
        ``<SAFE_STEM>.<CHANNEL>.crsd``, the SAFE path is inferred from
        the CRSD filename.

    Examples
    --------
    >>> builder = CRSDtoCPHD(
    ...     crsd_path='S1A_IW_RAW__0SDV.IW1_VV.crsd',
    ...     output_path='S1A_IW_RAW__0SDV.IW1_VV.cphd',
    ...     orbit_source='annotation',
    ... )
    >>> builder.convert()
    """

    def __init__(
        self,
        crsd_path: Union[str, Path],
        output_path: Union[str, Path],
        orbit_source: str = "auto",
        orbit_file: Optional[Union[str, Path]] = None,
        orbit_cache_dir: Optional[Union[str, Path]] = None,
        channel_id: Optional[str] = None,
        safe_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.crsd_path = Path(crsd_path)
        self.output_path = Path(output_path)
        self.orbit_source = orbit_source
        self.orbit_file = Path(orbit_file) if orbit_file else None
        self.orbit_cache_dir = Path(orbit_cache_dir) if orbit_cache_dir else None
        self.channel_id = channel_id
        # Infer SAFE path from CRSD filename if not provided
        if safe_path is None:
            safe_path = self._infer_safe_path()
        self.safe_path = Path(safe_path) if safe_path else None

    def convert(self) -> Path:
        """Run the full CRSD → CPHD conversion.

        Returns
        -------
        Path
            Path to the written CPHD file.
        """
        logger.info("CRSDtoCPHD: %s → %s", self.crsd_path, self.output_path)

        import sarkit.crsd

        # ── 1. Open CRSD ──────────────────────────────────────────────
        with open(str(self.crsd_path), "rb") as fh:
            crsd_reader = sarkit.crsd.Reader(fh)
            crsd_xml = crsd_reader.metadata.xmltree
            crsd_root = crsd_xml.getroot()
            ns = crsd_root.tag.split("}")[0].strip("{")
            t = lambda tag: f"{{{ns}}}{tag}"  # noqa: E731

            # Resolve channel
            ch_params = crsd_root.findall(f".//{t('Channel')}/{t('Parameters')}")
            ch_ids = [p.findtext(t("Identifier")) for p in ch_params]
            ch_id = self.channel_id or ch_ids[0]
            if ch_id not in ch_ids:
                raise ValueError(
                    f"Channel {ch_id!r} not found in CRSD. "
                    f"Available: {ch_ids}"
                )
            logger.info("Using CRSD channel: %s", ch_id)

            # Read CRSD parameters needed for CPHD XML construction
            crsd_params = self._read_crsd_params(crsd_root, ch_id, t)

            # Read raw time-domain signal and PVP
            logger.info("Reading CRSD signal (%s)…", ch_id)
            signal_fx_crsd = crsd_reader.read_signal(ch_id)   # (N, M) complex64, FX-domain
            pvp = crsd_reader.read_pvps(ch_id)                 # structured array

        n_pulses, n_samples = signal_fx_crsd.shape
        logger.info(
            "CRSD signal: %d pulses × %d samples, Fs=%.4f MHz",
            n_pulses, n_samples, crsd_params["fs"] / 1e6,
        )

        # ── 2. Resolve orbit ──────────────────────────────────────────
        orbit = self._resolve_orbit(crsd_params)

        # ── 3. Detect burst boundaries from PRI gaps in PVP ───────────
        ref_time_gps = crsd_params["ref_time_gps"]
        bursts = _detect_burst_boundaries(pvp, ref_time_gps)
        logger.info("Detected %d bursts in channel %s", len(bursts), ch_id)

        # ── 4. Allocate output arrays ─────────────────────────────────
        # CPHD NumSamples: round up to nearest multiple of 1024 for
        # efficient downstream processing (matches NGA convention).
        n_cphd = int(np.ceil(n_samples / 1024) * 1024)
        signal_fx = np.zeros((n_pulses, n_cphd), dtype=">c8")

        # Build the structured CPHD PVP array
        pvp_cphd = self._build_cphd_pvp_dtype(n_pulses, n_cphd,
                                               crsd_params, pvp, orbit,
                                               bursts, signal_fx_crsd, signal_fx)

        # ── 5. Write CPHD ─────────────────────────────────────────────
        cphd_xmltree = self._build_cphd_xml(
            ch_id, crsd_params, pvp_cphd, n_pulses, n_cphd,
        )
        cphd_meta = sarkit.cphd.Metadata(xmltree=cphd_xmltree)

        logger.info("Writing CPHD: %s", self.output_path)
        with open(str(self.output_path), "wb") as out_fh:
            writer = sarkit.cphd.Writer(out_fh, cphd_meta)
            writer.write_signal(ch_id, signal_fx)
            writer.write_pvp(ch_id, pvp_cphd)
            writer.done()

        logger.info("CPHD written: %s", self.output_path)
        return self.output_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _infer_safe_path(self) -> Optional[Path]:
        """Try to infer the SAFE path from the CRSD filename.

        GRDL names CRSD files as ``<SAFE_STEM>.<CHANNEL>.crsd``.
        If the parent directory contains a matching ``.SAFE``, return it.
        """
        stem = self.crsd_path.name
        # Remove e.g. ".IW1_VV.crsd"
        parts = stem.rsplit(".", 2)
        if len(parts) >= 2:
            safe_stem = parts[0]
            candidate = self.crsd_path.parent / f"{safe_stem}.SAFE"
            if candidate.exists():
                return candidate
        return None

    def _read_crsd_params(
        self,
        crsd_root: etree._Element,
        ch_id: str,
        t: Any,
    ) -> Dict[str, Any]:
        """Extract numeric parameters from the CRSD XML.

        Parameters
        ----------
        crsd_root : lxml Element
        ch_id : str
        t : callable
            Namespace tag wrapper: ``t("Tag")`` → ``"{ns}Tag"``.

        Returns
        -------
        dict with keys: f0, fs, bw, fx_min, fx_max, frcv_min, frcv_max,
            tx_pulse_len, chirp_rate, ref_time_gps, iarp_ecf, collection_ref_dt,
            collector_name, mode_id, pol_tx, pol_rx
        """
        def _ft(elem, path: str, default: float = 0.0) -> float:
            """Find + parse float, with default."""
            raw = elem.findtext(path) if elem is not None else None
            try:
                return float(raw)
            except (TypeError, ValueError):
                return default

        # Channel parameters
        ch_params_elem = None
        for p in crsd_root.findall(f".//{t('Channel')}/{t('Parameters')}"):
            if p.findtext(t("Identifier")) == ch_id:
                ch_params_elem = p
                break
        if ch_params_elem is None:
            raise ValueError(f"Channel {ch_id!r} not found in CRSD XML")

        f0 = _ft(ch_params_elem, t("F0Ref"))
        fs = _ft(ch_params_elem, t("Fs"), 64.345238e6)
        bw = _ft(ch_params_elem, t("BWInst"))
        frcv_min = _ft(ch_params_elem, t("FrcvMin"), f0 - fs / 2)
        frcv_max = _ft(ch_params_elem, t("FrcvMax"), f0 + fs / 2)

        # Global transmit bounds give FX range
        gl_tx = crsd_root.find(f".//{t('Global')}/{t('Transmit')}")
        fx_min = _ft(gl_tx, t("FxMin"), frcv_min) if gl_tx is not None else frcv_min
        fx_max = _ft(gl_tx, t("FxMax"), frcv_max) if gl_tx is not None else frcv_max

        # TxSequence for pulse length / chirp rate
        tx_params = crsd_root.find(f".//{t('TxSequence')}/{t('Parameters')}")
        tx_pulse_len = _ft(tx_params, t("TXmtMin"), 52e-6)
        # Chirp rate not directly in CRSD TxSequence; derive from BW / pulse_len
        chirp_rate = bw / tx_pulse_len if tx_pulse_len > 0 else 1e12

        # TxWaveform for pulse length (CPHD TxRcv section)
        txwf = crsd_root.find(f".//{t('TxSequence')}/{t('Parameters')}")
        # Prefer dedicated fields from TxWaveform if present
        lfw_rate = _ft(txwf, t("FxBW"), 0.0) if txwf is not None else 0.0
        if lfw_rate > 0 and tx_pulse_len > 0:
            chirp_rate = lfw_rate / tx_pulse_len

        # Collection reference time
        crt_text = crsd_root.findtext(f".//{t('Global')}/{t('CollectionRefTime')}")
        if crt_text:
            crt_text = crt_text.rstrip("Z").strip()
            try:
                collection_ref_dt = datetime.strptime(crt_text, "%Y-%m-%dT%H:%M:%S.%f")
            except ValueError:
                collection_ref_dt = datetime.strptime(crt_text, "%Y-%m-%dT%H:%M:%S")
        else:
            collection_ref_dt = datetime.utcnow()

        # Convert CollectionRefTime to GPS seconds
        _GPS_EPOCH = datetime(1980, 1, 6, 0, 0, 0)
        _GPS_LEAP = 18
        ref_time_gps = (
            (collection_ref_dt - _GPS_EPOCH).total_seconds() + _GPS_LEAP
        )

        # IARP
        iarp_ecf_elem = crsd_root.find(f".//{t('SceneCoordinates')}/{t('IARP')}/{t('ECF')}")
        if iarp_ecf_elem is not None:
            iarp_ecf = np.array([
                float(iarp_ecf_elem.findtext(t("X")) or 0),
                float(iarp_ecf_elem.findtext(t("Y")) or 0),
                float(iarp_ecf_elem.findtext(t("Z")) or 0),
            ])
        else:
            iarp_ecf = np.zeros(3)

        # Collection metadata
        collector_name = crsd_root.findtext(
            f".//{t('TransmitInfo')}/{t('SensorName')}", "S1A"
        )
        mode_id = crsd_root.findtext(
            f".//{t('SARInfo')}/{t('RadarMode')}/{t('ModeID')}", "IW"
        )

        # Polarization from channel
        pol_rx = ch_params_elem.findtext(
            f".//{t('RcvPolarization')}/{t('PolarizationID')}", "V"
        )
        # TX polarization from TxSequence
        pol_tx_elem = crsd_root.find(
            f".//{t('TxSequence')}/{t('Parameters')}/{t('TxPolarization')}/{t('PolarizationID')}"
        )
        pol_tx = pol_tx_elem.text if pol_tx_elem is not None else pol_rx

        # ImageAreaCornerPoints for SceneCoordinates
        iacps = []
        for cp in crsd_root.findall(f".//{t('SceneCoordinates')}/{t('ImageAreaCornerPoints')}/{t('IACP')}"):
            lat = float(cp.findtext(t("Lat")) or 0)
            lon = float(cp.findtext(t("Lon")) or 0)
            iacps.append((lat, lon))

        return {
            "f0": f0,
            "fs": fs,
            "bw": bw,
            "fx_min": fx_min,
            "fx_max": fx_max,
            "frcv_min": frcv_min,
            "frcv_max": frcv_max,
            "tx_pulse_len": tx_pulse_len,
            "chirp_rate": chirp_rate,
            "ref_time_gps": ref_time_gps,
            "collection_ref_dt": collection_ref_dt,
            "iarp_ecf": iarp_ecf,
            "collector_name": collector_name,
            "mode_id": mode_id,
            "pol_tx": pol_tx,
            "pol_rx": pol_rx,
            "iacps": iacps,
        }

    def _resolve_orbit(self, crsd_params: Dict[str, Any]) -> Any:
        """Build an OrbitInterpolator from the source SAFE or CRSD.

        Falls back gracefully if the SAFE path is not available.
        """
        try:
            from grdl.IO.sar.sentinel1_l0 import Sentinel1L0Reader
            from grdl.IO.sar.sentinel1_l0.orbit import OrbitResolver

            if self.safe_path is None or not self.safe_path.exists():
                raise FileNotFoundError(
                    f"SAFE path not found: {self.safe_path!r}. "
                    "Provide safe_path= to CRSDtoCPHD for orbit interpolation."
                )

            reader = Sentinel1L0Reader(self.safe_path)
            meta = reader.metadata

            sensing_start = getattr(meta, "start_time", None)
            if sensing_start is None:
                raise ValueError("Cannot determine sensing start from SAFE metadata")
            if sensing_start.tzinfo is None:
                from datetime import timezone
                sensing_start = sensing_start.replace(tzinfo=timezone.utc)

            mission_obj = getattr(meta, "mission", None)
            mission_str = (
                str(mission_obj.name if hasattr(mission_obj, "name") else mission_obj)
                if mission_obj else "S1A"
            )

            # Load ephemeris vectors — L0 products have no annotation XML,
            # vectors come from the sub-commutated ISP ephemeris
            from grdl.IO.models.sentinel1_l0 import S1L0OrbitStateVector
            from datetime import timedelta, timezone

            _GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
            _GPS_LEAP = 18

            annotation_vectors: list = list(meta.orbit_state_vectors or [])
            if not annotation_vectors:
                for br in reader._burst_readers.values():
                    dec = getattr(br, "_decoder", None)
                    if dec is None:
                        continue
                    try:
                        from sentinel1decoder import Level0File as _L0F
                        l0f = _L0F(str(dec.measurement_file))
                        eph = l0f.ephemeris
                    except Exception:
                        continue
                    if eph is None or len(eph) == 0:
                        continue
                    for _, row in eph.iterrows():
                        try:
                            gps_ts = float(row["POD Solution Data Timestamp"])
                            utc_t = _GPS_EPOCH + timedelta(
                                seconds=gps_ts - _GPS_LEAP
                            )
                            annotation_vectors.append(S1L0OrbitStateVector(
                                time=utc_t,
                                x=float(row["X-axis position ECEF"]),
                                y=float(row["Y-axis position ECEF"]),
                                z=float(row["Z-axis position ECEF"]),
                                vx=float(row["X-axis velocity ECEF"]),
                                vy=float(row["Y-axis velocity ECEF"]),
                                vz=float(row["Z-axis velocity ECEF"]),
                            ))
                        except (KeyError, ValueError, TypeError):
                            continue
                    if annotation_vectors:
                        break

            resolver = OrbitResolver(
                orbit_source=self.orbit_source,
                orbit_file=self.orbit_file,
                orbit_cache_dir=self.orbit_cache_dir,
            )
            orbit = resolver.resolve(
                annotation_vectors=annotation_vectors,
                sensing_start=sensing_start,
                reference_time=sensing_start,
                mission=mission_str,
            )
            logger.info("Orbit resolved via %s", self.orbit_source)
            return orbit

        except Exception as exc:
            raise RuntimeError(
                f"Cannot resolve orbit for TOPS deramping: {exc}. "
                "Provide safe_path= and ensure orbit source is available."
            ) from exc

    def _build_cphd_pvp_dtype(
        self,
        n_pulses: int,
        n_cphd: int,
        crsd_params: Dict[str, Any],
        crsd_pvp: np.ndarray,
        orbit: Any,
        bursts: List[Tuple[int, int]],
        signal_fx_crsd: np.ndarray,
        signal_fx: np.ndarray,
    ) -> np.ndarray:
        """Apply TOPS deramping and build the CPHD PVP structured array.

        For each burst:
        1. Compute burst centre time ``eta_c`` from PVP RcvStart.
        2. Derive per-burst SRPPos as platform nadir at ``eta_c``.
        3. Fit ``k_psi`` from geometric Doppler centroid across the burst.
        4. Multiply signal by ``exp(-j*pi*k_psi*(eta-eta_c)^2)`` (deramp).
        5. Copy deramped FX-domain rows into output array with zero-padding.
           Note: the range FFT was applied in Stage 1 (``crsd_writer.py``) to
           comply with the CRSDsar format requirement that signal arrays be
           stored in the FX (range-frequency) domain.  No second FFT is applied
           here.
        6. Set ``aFDOP = f_dc0`` (residual ≈ 0 Hz after deramp).

        Parameters
        ----------
        n_pulses : int
        n_cphd : int
            Output sample count (rounded up to multiple of 1024).
        crsd_params : dict
        crsd_pvp : ndarray
            Structured PVP from CRSD.
        orbit : OrbitInterpolator
        bursts : list of (start, end)
            Burst boundaries as inclusive-start, exclusive-end index pairs.
        signal_fx_crsd : ndarray, shape (N, M)
            FX-domain signal read from CRSD (range FFT already applied in
            Stage 1 by ``crsd_writer.py`` to satisfy CRSDsar format spec).
        signal_fx : ndarray, shape (N, n_cphd)
            Output FX-domain signal after TOPS deramping (written in-place).

        Returns
        -------
        ndarray
            Structured CPHD PVP array.
        """
        from grdl.IO.sar.sentinel1_l0.constants import GPS_LEAP_SECONDS

        f0 = crsd_params["f0"]
        fs = crsd_params["fs"]
        ref_time_gps = crsd_params["ref_time_gps"]

        # Build CPHD PVP dtype matching the standard 19-field layout
        cphd_pvp_dtype = np.dtype([
            ("TxTime",      np.float64),
            ("TxPos",       np.float64, (3,)),
            ("TxVel",       np.float64, (3,)),
            ("RcvTime",     np.float64),
            ("RcvPos",      np.float64, (3,)),
            ("RcvVel",      np.float64, (3,)),
            ("SRPPos",      np.float64, (3,)),
            ("aFDOP",       np.float64),
            ("aFRR1",       np.float64),
            ("aFRR2",       np.float64),
            ("FX1",         np.float64),
            ("FX2",         np.float64),
            ("TOA1",        np.float64),
            ("TOA2",        np.float64),
            ("TDTropoSRP",  np.float64),
            ("SC0",         np.float64),
            ("SCSS",        np.float64),
            ("SIGNAL",      np.int64),
            ("AmpSF",       np.float64),
        ])
        pvp_out = np.zeros(n_pulses, dtype=cphd_pvp_dtype)

        # Orbit reference time in UTC seconds since GPS epoch.
        # The orbit interpolator uses times relative to orbit.reference_time
        # expressed in UTC.  ISP Coarse+Fine gives GPS seconds; subtract
        # GPS_LEAP_SECONDS to convert to UTC before computing the relative
        # time: orbit_rel = (abs_gps - GPS_LEAP_SECONDS) - orbit_ref_utc
        # _GPS_EPOCH must also be UTC-aware so the subtraction is well-defined.
        from datetime import timezone as _tz
        _GPS_EPOCH = datetime(1980, 1, 6, tzinfo=_tz.utc)
        orbit_ref_utc = (orbit.reference_time - _GPS_EPOCH).total_seconds()
        # (= UTC seconds since GPS epoch, no leap-second correction)

        # Reconstruct absolute receive times from CRSD RcvStart IntFrac
        # sarkit uses nested dtype fields named 'Int' and 'Frac'.
        rcv_int  = crsd_pvp["RcvStart"]["Int"].astype(np.float64)
        rcv_frac = crsd_pvp["RcvStart"]["Frac"].astype(np.float64)
        rcv_rel  = rcv_int + rcv_frac          # relative to ref_time_gps
        rcv_abs  = rcv_rel + ref_time_gps      # GPS seconds

        # Platform positions/velocities for all pulses (orbit query in UTC)
        orbit_rel_all = (rcv_abs - GPS_LEAP_SECONDS) - orbit_ref_utc
        pos_all, vel_all = orbit.interpolate(orbit_rel_all)

        # Global SC0 / SCSS consistent with FRCV1/Fs (NGA convention)
        sc0  = f0 - fs / 2.0                   # = FrcvMin = ADC window start
        scss = fs / (n_cphd - 1)               # Hz/sample over n_cphd samples

        # FX1/FX2: chirp bandwidth window centred on F0Ref, per NGA convention.
        # NGA uses FX1 = f0 - BWInst/2, FX2 = f0 + BWInst/2 (not the full ADC
        # window FrcvMin/FrcvMax).  BWInst is in crsd_params["bw"].
        bw = crsd_params["bw"]
        fx1_chirp = f0 - bw / 2.0
        fx2_chirp = f0 + bw / 2.0

        # aFRR1/aFRR2: encode the range chirp FM rate per the CPHD 1.0 spec.
        # sarkit consistency check: aFRR1 = 2*fx_c / (c * LFMRate)
        #                           aFRR2 = 2       / (c * LFMRate)
        # where LFMRate = BWInst / TXmtMin (from CRSD TxSequence/Parameters).
        chirp_rate = crsd_params["chirp_rate"]   # Hz/s = BWInst / TXmtMin
        afrr1 = 2.0 * f0 / (_C * chirp_rate)
        afrr2 = 2.0       / (_C * chirp_rate)

        # TOA window: half the receive window duration
        toa_half = float(n_cphd / (2.0 * fs))

        # Process burst by burst
        for b_idx, (b_start, b_end) in enumerate(bursts):
            n_b = b_end - b_start

            # Burst centre time (absolute GPS seconds)
            eta_c = float(
                rcv_abs[b_start] + (rcv_abs[b_end - 1] - rcv_abs[b_start]) / 2.0
            )

            # Per-burst SRP: platform nadir projected to h=0 at burst centre
            orbit_rel_c = (eta_c - GPS_LEAP_SECONDS) - orbit_ref_utc
            pos_c, vel_c = orbit.interpolate(np.array([orbit_rel_c]))
            srp = _nadir_ecef(pos_c[0])

            # Per-pulse Doppler centroid across the burst
            b_pos = pos_all[b_start:b_end]
            b_vel = vel_all[b_start:b_end]
            f_dc  = _doppler_centroid_array(b_pos, b_vel, srp, f0)
            eta_rel = rcv_abs[b_start:b_end] - eta_c

            # Fit k_psi and f_dc0
            k_psi, f_dc0 = _fit_tops_steering_rate(eta_rel, f_dc)
            logger.debug(
                "Burst %d [%d:%d]: k_psi=%.2f Hz/s, f_dc0=%.4f Hz, "
                "SRP=[%.1f, %.1f, %.1f]",
                b_idx, b_start, b_end, k_psi, f_dc0,
                srp[0], srp[1], srp[2],
            )

            # TOPS deramp: exp(-j*pi*k_psi*(eta-eta_c)^2) per pulse.
            # The CRSD signal is in the FX (range-frequency) domain because
            # Stage 1 (crsd_writer.py) applied the range FFT before writing
            # to comply with the CRSDsar format specification.  The deramp
            # filter h(eta) is separable from the range axis, so it applies
            # identically to FX-domain rows as it would to time-domain rows.
            # No additional range FFT is needed here.
            phase = -np.pi * k_psi * eta_rel ** 2            # (n_b,)
            deramp = np.exp(1j * phase).astype(np.complex64)  # (n_b,)
            fx_burst = signal_fx_crsd[b_start:b_end, :].astype(np.complex64)
            fx_deramped = fx_burst * deramp[:, np.newaxis]    # (n_b, M)

            # Copy into output array with zero-padding at high-frequency end
            n_samp_raw = signal_fx_crsd.shape[1]
            signal_fx[b_start:b_end, :n_samp_raw] = fx_deramped.astype(">c8")

            # Fill CPHD PVP for this burst
            sl = slice(b_start, b_end)

            # TxTime = RcvStart - SWST (approx): use RcvStart directly as
            # monostatic transmit time (bistatic correction is negligible
            # at SAR-scale precision for a single scene)
            pvp_out["TxTime"][sl]  = rcv_rel[b_start:b_end]
            pvp_out["RcvTime"][sl] = rcv_rel[b_start:b_end] + float(n_cphd) / (2.0 * fs)
            pvp_out["TxPos"][sl]   = b_pos
            pvp_out["TxVel"][sl]   = b_vel
            pvp_out["RcvPos"][sl]  = b_pos    # monostatic: same as TxPos
            pvp_out["RcvVel"][sl]  = b_vel
            pvp_out["SRPPos"][sl]  = srp[np.newaxis, :]

            # aFDOP ≈ 0 after deramp (residual = f_dc0)
            pvp_out["aFDOP"][sl]   = f_dc0

            pvp_out["aFRR1"][sl]   = afrr1
            pvp_out["aFRR2"][sl]   = afrr2

            pvp_out["FX1"][sl]     = fx1_chirp
            pvp_out["FX2"][sl]     = fx2_chirp
            pvp_out["TOA1"][sl]    = -toa_half
            pvp_out["TOA2"][sl]    =  toa_half
            pvp_out["TDTropoSRP"][sl] = 0.0
            pvp_out["SC0"][sl]     = sc0
            pvp_out["SCSS"][sl]    = scss
            pvp_out["SIGNAL"][sl]  = 1
            pvp_out["AmpSF"][sl]   = 1.0      # no radiometric cal

        logger.info("TOPS deramping complete for all %d bursts", len(bursts))
        return pvp_out

    def _build_cphd_xml(
        self,
        ch_id: str,
        params: Dict[str, Any],
        pvp: np.ndarray,
        n_pulses: int,
        n_cphd: int,
    ) -> etree.ElementTree:
        """Construct the CPHD 1.0.1 XML tree.

        Parameters
        ----------
        ch_id : str
        params : dict
        pvp : ndarray
            Structured PVP array (used to derive dwell time polynomial).
        n_pulses : int
        n_cphd : int

        Returns
        -------
        lxml.etree.ElementTree
        """
        root = etree.Element(f"{{{_CPHD_NS}}}CPHD", nsmap=_CPHD_NSMAP)

        f0     = params["f0"]
        fs     = params["fs"]
        bw     = params["bw"]
        sc0    = f0 - fs / 2.0
        scss   = fs / (n_cphd - 1)
        toa_half = float(n_cphd / (2.0 * fs))

        # Derive waveform parameters from PVP
        tx_time1 = float(pvp["TxTime"].min())
        tx_time2 = float(pvp["TxTime"].max())
        collection_start_str = (
            params["collection_ref_dt"].strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
        )
        tx_pol = params["pol_tx"]
        rx_pol = params["pol_rx"]

        # FX band: use chirp support (FX1/FX2) extremes across all pulses
        fx_min = float(pvp["FX1"].min())
        fx_max = float(pvp["FX2"].max())

        # ── CollectionID ──────────────────────────────────────────────
        cid = _sub(root, "CollectionID")
        _sub(cid, "CollectorName", params["collector_name"])
        stem = self.crsd_path.name.rsplit(".", 2)[0]
        _sub(cid, "CoreName", f"{stem}.{ch_id}")
        _sub(cid, "CollectType", "MONOSTATIC")
        rm = _sub(cid, "RadarMode")
        _sub(rm, "ModeType", "DYNAMIC STRIPMAP")
        _sub(rm, "ModeID", params["mode_id"])
        _sub(cid, "Classification", "UNCLASSIFIED")
        _sub(cid, "ReleaseInfo", "PUBLIC RELEASE")

        # ── Global ────────────────────────────────────────────────────
        gl = _sub(root, "Global")
        _sub(gl, "DomainType", "FX")
        _sub(gl, "SGN", "-1")
        tl = _sub(gl, "Timeline")
        _sub(tl, "CollectionStart", collection_start_str)
        _sub(tl, "TxTime1", f"{tx_time1:.12g}")
        _sub(tl, "TxTime2", f"{tx_time2:.12g}")
        fxb = _sub(gl, "FxBand")
        _sub(fxb, "FxMin", f"{fx_min:.12g}")
        _sub(fxb, "FxMax", f"{fx_max:.12g}")
        toa_sw = _sub(gl, "TOASwath")
        _sub(toa_sw, "TOAMin", f"{-toa_half:.12g}")
        _sub(toa_sw, "TOAMax", f"{toa_half:.12g}")

        # ── SceneCoordinates ──────────────────────────────────────────
        iarp_ecf = params["iarp_ecf"]
        llh = ecef_to_geodetic(iarp_ecf)  # (3,) → (3,) [lat, lon, hae]

        sc_elem = _sub(root, "SceneCoordinates")
        _sub(sc_elem, "EarthModel", "WGS_84")
        iarp_e = _sub(sc_elem, "IARP")
        _sub_xyz(iarp_e, "ECF", iarp_ecf[0], iarp_ecf[1], iarp_ecf[2])
        llh_e = _sub(iarp_e, "LLH")
        _sub(llh_e, "Lat", f"{float(llh[0]):.8g}")
        _sub(llh_e, "Lon", f"{float(llh[1]):.8g}")
        _sub(llh_e, "HAE", "0")

        # ReferenceSurface: planar tangent frame at IARP
        ref_surf = _sub(sc_elem, "ReferenceSurface")
        planar = _sub(ref_surf, "Planar")
        iarp_llh = np.array([float(llh[0]), float(llh[1]), 0.0])
        u_east  = sarkit.wgs84.east(iarp_llh)
        u_north = sarkit.wgs84.north(iarp_llh)
        _sub_xyz(planar, "uIAX", u_east[0],  u_east[1],  u_east[2])
        _sub_xyz(planar, "uIAY", u_north[0], u_north[1], u_north[2])

        # ImageArea
        ia_e = _sub(sc_elem, "ImageArea")
        _sub_xy(ia_e, "X1Y1", -1.0, -1.0)
        _sub_xy(ia_e, "X2Y2",  1.0,  1.0)

        # ImageAreaCornerPoints
        iacps = params.get("iacps", [])
        if iacps:
            iacp_root = _sub(sc_elem, "ImageAreaCornerPoints")
            for idx, (lat, lon) in enumerate(iacps[:4], start=1):
                cp = _sub_latlon(iacp_root, "IACP", lat, lon)
                cp.set("index", str(idx))

        # ── Data ──────────────────────────────────────────────────────
        pvp_bytes = 29 * 8   # 29 eight-byte slots (scalars=1, XYZ=3 each)
        data_e = _sub(root, "Data")
        _sub(data_e, "SignalArrayFormat", "CF8")
        _sub(data_e, "NumBytesPVP", str(pvp_bytes))
        _sub(data_e, "NumCPHDChannels", "1")
        ch_data = _sub(data_e, "Channel")
        _sub(ch_data, "Identifier", ch_id)
        _sub(ch_data, "NumVectors",  str(n_pulses))
        _sub(ch_data, "NumSamples",  str(n_cphd))
        _sub(ch_data, "SignalArrayByteOffset", "0")
        _sub(ch_data, "PVPArrayByteOffset",    "0")
        _sub(data_e, "NumSupportArrays", "0")

        # ── Channel ───────────────────────────────────────────────────
        ch_root_e = _sub(root, "Channel")
        _sub(ch_root_e, "RefChId", ch_id)
        _sub(ch_root_e, "FXFixedCPHD",  "true")
        _sub(ch_root_e, "TOAFixedCPHD", "true")
        _sub(ch_root_e, "SRPFixedCPHD", "false")   # SRP varies per burst
        ch_params_e = _sub(ch_root_e, "Parameters")
        _sub(ch_params_e, "Identifier",     ch_id)
        _sub(ch_params_e, "RefVectorIndex", "0")
        _sub(ch_params_e, "FXFixed",  "true")
        _sub(ch_params_e, "TOAFixed", "true")
        _sub(ch_params_e, "SRPFixed", "false")
        pol_e = _sub(ch_params_e, "Polarization")
        _sub(pol_e, "TxPol",  tx_pol)
        _sub(pol_e, "RcvPol", rx_pol)
        _sub(ch_params_e, "FxC",      f"{f0:.12g}")
        _sub(ch_params_e, "FxBW",     f"{bw:.12g}")
        _sub(ch_params_e, "TOASaved", f"{toa_half * 2:.12g}")
        dwell_e = _sub(ch_params_e, "DwellTimes")
        _sub(dwell_e, "CODId",   "COD1")
        _sub(dwell_e, "DwellId", "DW1")
        txrcv_e = _sub(ch_params_e, "TxRcv")
        _sub(txrcv_e, "TxWFId", "TXW1")
        _sub(txrcv_e, "RcvId",  "RCV1")

        # ── PVP ───────────────────────────────────────────────────────
        pvp_e = _sub(root, "PVP")
        fields = [
            ("TxTime",      0,  1, "F8"),
            ("TxPos",       1,  3, "X=F8;Y=F8;Z=F8;"),
            ("TxVel",       4,  3, "X=F8;Y=F8;Z=F8;"),
            ("RcvTime",     7,  1, "F8"),
            ("RcvPos",      8,  3, "X=F8;Y=F8;Z=F8;"),
            ("RcvVel",     11,  3, "X=F8;Y=F8;Z=F8;"),
            ("SRPPos",     14,  3, "X=F8;Y=F8;Z=F8;"),
            ("aFDOP",      17,  1, "F8"),
            ("aFRR1",      18,  1, "F8"),
            ("aFRR2",      19,  1, "F8"),
            ("FX1",        20,  1, "F8"),
            ("FX2",        21,  1, "F8"),
            ("TOA1",       22,  1, "F8"),
            ("TOA2",       23,  1, "F8"),
            ("TDTropoSRP", 24,  1, "F8"),
            ("SC0",        25,  1, "F8"),
            ("SCSS",       26,  1, "F8"),
            ("SIGNAL",     27,  1, "I8"),
            ("AmpSF",      28,  1, "F8"),
        ]
        for fname, offset, size, fmt in fields:
            fe = _sub(pvp_e, fname)
            _sub(fe, "Offset", str(offset))
            _sub(fe, "Size",   str(size))
            _sub(fe, "Format", fmt)

        # ── Dwell ─────────────────────────────────────────────────────
        dwell_root = _sub(root, "Dwell")
        _sub(dwell_root, "NumCODTimes", "1")
        cod_e = _sub(dwell_root, "CODTime")
        _sub(cod_e, "Identifier", "COD1")
        cod_poly = _sub(cod_e, "CODTimePoly")
        cod_poly.set("order1", "0")
        cod_poly.set("order2", "0")
        coef_e = _sub(cod_poly, "Coef", "0.0")
        coef_e.set("exponent1", "0")
        coef_e.set("exponent2", "0")

        dwell_span = float(pvp["TxTime"].max() - pvp["TxTime"].min())
        _sub(dwell_root, "NumDwellTimes", "1")
        dw_e = _sub(dwell_root, "DwellTime")
        _sub(dw_e, "Identifier", "DW1")
        dw_poly = _sub(dw_e, "DwellTimePoly")
        dw_poly.set("order1", "0")
        dw_poly.set("order2", "0")
        dw_coef = _sub(dw_poly, "Coef", f"{dwell_span:.12g}")
        dw_coef.set("exponent1", "0")
        dw_coef.set("exponent2", "0")

        # ── ReferenceGeometry ─────────────────────────────────────────
        # Use the median SRP (burst centre of the middle burst)
        mid = n_pulses // 2
        srp_med = pvp["SRPPos"][mid]
        pos_med = pvp["TxPos"][mid]
        vel_med = pvp["TxVel"][mid]

        r_vec   = srp_med - pos_med
        slant_r = float(np.linalg.norm(r_vec))
        r_hat   = r_vec / slant_r

        # WGS-84 surface normal at SRP for graze angle
        srp_llh = ecef_to_geodetic(srp_med)  # (3,) → (3,) [lat, lon, hae]
        n_hat = sarkit.wgs84.up(np.array([float(srp_llh[0]), float(srp_llh[1]), 0.0]))
        graze_rad = float(np.arcsin(np.dot(-r_hat, n_hat)))
        graze_deg = float(np.degrees(graze_rad))
        inc_deg   = 90.0 - graze_deg
        look_side = "R"  # Sentinel-1 is always right-looking

        rg_e = _sub(root, "ReferenceGeometry")
        srp_rg = _sub(rg_e, "SRP")
        _sub_xyz(srp_rg, "ECF", float(srp_med[0]), float(srp_med[1]), float(srp_med[2]))
        _sub(rg_e, "ReferenceTime",   "0.0")
        _sub(rg_e, "SRPCODTime",      "0.0")
        _sub(rg_e, "SRPDwellTime",    f"{dwell_span:.12g}")
        mono = _sub(rg_e, "Monostatic")
        _sub_xyz(mono, "ARPPos", float(pos_med[0]), float(pos_med[1]), float(pos_med[2]))
        _sub_xyz(mono, "ARPVel", float(vel_med[0]), float(vel_med[1]), float(vel_med[2]))
        _sub(mono, "SideOfTrack",       look_side)
        _sub(mono, "SlantRange",        f"{slant_r:.6g}")
        ground_range = slant_r * float(np.cos(graze_rad))
        _sub(mono, "GroundRange",       f"{ground_range:.6g}")
        _sub(mono, "DopplerConeAngle",  f"{float(np.degrees(np.arccos(np.dot(vel_med / np.linalg.norm(vel_med), r_hat)))):.6g}")
        _sub(mono, "GrazeAngle",        f"{graze_deg:.6g}")
        _sub(mono, "IncidenceAngle",    f"{inc_deg:.6g}")
        _sub(mono, "AzimuthAngle",      "0.0")
        _sub(mono, "TwistAngle",        "0.0")
        _sub(mono, "SlopeAngle",        "0.0")
        _sub(mono, "LayoverAngle",      "0.0")

        # ── TxRcv ─────────────────────────────────────────────────────
        txrcv_root = _sub(root, "TxRcv")
        _sub(txrcv_root, "NumTxWFs", "1")
        txwf = _sub(txrcv_root, "TxWFParameters")
        _sub(txwf, "Identifier",  "TXW1")
        _sub(txwf, "PulseLength",  f"{params['tx_pulse_len']:.12g}")
        _sub(txwf, "RFBandwidth",  f"{bw:.12g}")
        _sub(txwf, "FreqCenter",   f"{f0:.12g}")
        _sub(txwf, "LFMRate",      f"{params['chirp_rate']:.12g}")
        _sub(txwf, "Polarization", tx_pol)
        _sub(txrcv_root, "NumRcvs", "1")
        rcv_p = _sub(txrcv_root, "RcvParameters")
        _sub(rcv_p, "Identifier",   "RCV1")
        _sub(rcv_p, "WindowLength", f"{float(n_cphd) / fs:.12g}")
        _sub(rcv_p, "SampleRate",   f"{fs:.12g}")
        _sub(rcv_p, "IFFilterBW",   f"{bw:.12g}")
        _sub(rcv_p, "FreqCenter",   f"{f0:.12g}")
        _sub(rcv_p, "LFMRate",      f"{params['chirp_rate']:.12g}")
        _sub(rcv_p, "Polarization", rx_pol)

        return etree.ElementTree(root)

# Convenience alias matching the GRDL naming convention
Sentinel1CRSDtoCPHD = CRSDtoCPHD
