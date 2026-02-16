# -*- coding: utf-8 -*-
"""
Stripmap PFA - Subaperture Polar Format Algorithm for stripmap SAR.

Partitions a stripmap collection into overlapping spotlight-like
sub-apertures, rephases each to a fixed centre SRP, forms each with
the existing ``PolarFormatAlgorithm``, and mosaics the results into
a continuous strip image.

Dependencies
------------
scipy
sarpy (for ``CollectionGeometry`` dependency)

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
2026-02-13

Modified
--------
2026-02-13
"""

# Standard library
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Third-party
import numpy as np
from numpy.linalg import norm

# GRDL internal
from grdl.IO.models.cphd import CPHDMetadata, create_subaperture_metadata
from grdl.image_processing.sar.image_formation.base import (
    ImageFormationAlgorithm,
)
from grdl.image_processing.sar.image_formation.geometry import (
    CollectionGeometry,
)
from grdl.image_processing.sar.image_formation.polar_grid import PolarGrid
from grdl.image_processing.sar.image_formation.pfa import (
    PolarFormatAlgorithm,
)
from grdl.image_processing.sar.image_formation.subaperture import (
    SubaperturePartitioner,
)


# Speed of light (m/s)
_C = 299792458.0


class StripmapPFA(ImageFormationAlgorithm):
    """Subaperture PFA for stripmap SAR image formation.

    Partitions the full stripmap aperture into overlapping
    spotlight-like sub-apertures, rephases each to a fixed centre
    SRP, forms each with ``PolarFormatAlgorithm``, and mosaics the
    sub-aperture images into a continuous strip.

    Parameters
    ----------
    metadata : CPHDMetadata
        Full CPHD metadata with populated PVP arrays.
    interpolator : callable, optional
        Interpolation function ``(x_old, y_old, x_new) -> y_new``.
    weighting : str or callable, optional
        Window function for PFA compression.
    grid_mode : str
        ``'inscribed'`` or ``'circumscribed'``.
    range_oversample : float
        Range k-space oversampling factor.
    azimuth_oversample : float
        Azimuth k-space oversampling factor.
    subaperture_pulses : int, optional
        Fixed sub-aperture length.  If None, auto-sized.
    overlap_fraction : float
        Overlap between adjacent sub-apertures (0.0–1.0).
    max_phase_error : float
        Phase error budget for auto-sizing (radians).
    slant : bool
        If True, use slant plane projection.
    pad_factor : float
        Zero-pad factor for compression.
    verbose : bool
        Print per-sub-aperture diagnostics.

    Examples
    --------
    >>> from grdl.IO.sar import CPHDReader
    >>> from grdl.image_processing.sar.image_formation import StripmapPFA
    >>> with CPHDReader('stripmap.cphd') as reader:
    ...     meta = reader.metadata
    ...     signal = reader.read_full()
    >>> ifp = StripmapPFA(meta)
    >>> image = ifp.form_image(signal, geometry=None)
    """

    def __init__(
        self,
        metadata: CPHDMetadata,
        interpolator: Optional[Callable] = None,
        weighting: Union[str, Callable, None] = None,
        grid_mode: str = 'inscribed',
        range_oversample: float = 1.0,
        azimuth_oversample: float = 1.0,
        subaperture_pulses: Optional[int] = None,
        overlap_fraction: float = 0.5,
        max_phase_error: float = np.pi / 4,
        slant: bool = True,
        pad_factor: float = 1.25,
        verbose: bool = True,
    ) -> None:
        if metadata.pvp is None:
            raise ValueError(
                "CPHDMetadata must have populated PVP arrays"
            )
        self._metadata = metadata
        self._interpolator = interpolator
        self._weighting = weighting
        self._grid_mode = grid_mode
        self._range_oversample = range_oversample
        self._azimuth_oversample = azimuth_oversample
        self._slant = slant
        self._pad_factor = pad_factor
        self._verbose = verbose

        # Partition the aperture
        self._partitioner = SubaperturePartitioner(
            metadata,
            subaperture_pulses=subaperture_pulses,
            max_phase_error=max_phase_error,
            overlap_fraction=overlap_fraction,
        )

        # Populated after form_image()
        self._output_grid: Optional[Dict[str, Any]] = None
        self._sub_grids: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Rephasing
    # ------------------------------------------------------------------

    @staticmethod
    def rephase_signal(
        signal: np.ndarray,
        pvp_sub,
        center_srp: np.ndarray,
        phase_sgn: int = -1,
    ) -> np.ndarray:
        """Rephase FX-domain signal from per-pulse SRP to a fixed SRP.

        In CPHD, each pulse is compensated to its own SRP.  For PFA
        the entire sub-aperture must be compensated to a single point.
        This applies a per-pulse, per-sample phase shift in the
        frequency domain.

        The CPHD signal model phase is ``SGN * fx * ΔTOA`` (in cycles).
        The rephase factor to move from ``R_old`` to ``R_new`` is:

            ``exp(+j * SGN * 4π * fx * (R_new - R_old) / c)``

        Parameters
        ----------
        signal : np.ndarray
            Phase history, shape ``(npulses, nsamples)``.
        pvp_sub : CPHDPVP
            PVP arrays for the sub-aperture (sliced).
        center_srp : np.ndarray
            Fixed SRP ECF position, shape ``(3,)``.
        phase_sgn : int
            CPHD phase sign convention (``+1`` or ``-1``).

        Returns
        -------
        np.ndarray
            Rephased signal, same shape and dtype.
        """
        npulses, nsamples = signal.shape
        result = signal.copy()

        # Monostatic ARP = midpoint of tx and rcv
        arp = 0.5 * (pvp_sub.tx_pos + pvp_sub.rcv_pos)

        # Range from ARP to original per-pulse SRP
        r_old = norm(arp - pvp_sub.srp_pos, axis=1)

        # Range from ARP to fixed centre SRP
        r_new = norm(arp - center_srp[np.newaxis, :], axis=1)

        # Range difference
        delta_r = r_new - r_old  # shape (npulses,)

        # Build frequency grid (npulses, nsamples) — vectorized
        sc0 = pvp_sub.sc0[:, np.newaxis]       # (npulses, 1)
        scss = pvp_sub.scss[:, np.newaxis]      # (npulses, 1)
        sample_idx = np.arange(nsamples)[np.newaxis, :]  # (1, nsamples)

        freq = sample_idx * scss + sc0  # (npulses, nsamples)
        phase = np.exp(
            1j * phase_sgn * 4.0 * np.pi * freq
            * delta_r[:, np.newaxis] / _C
        )
        result *= phase

        return result

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def form_image(
        self,
        signal: np.ndarray,
        geometry: Any,
    ) -> np.ndarray:
        """Run the stripmap subaperture PFA pipeline.

        Parameters
        ----------
        signal : np.ndarray
            Full stripmap phase history, shape
            ``(total_pulses, nsamples)``.
        geometry : Any
            Ignored (geometry is computed per sub-aperture from
            metadata).

        Returns
        -------
        np.ndarray
            Complex SAR image (mosaiced strip).
        """
        # Ensure signal is complex (sarkit may return structured int16)
        if signal.dtype.names:
            signal = (
                signal['real'].astype(np.float32)
                + 1j * signal['imag'].astype(np.float32)
            )

        partitions = self._partitioner.partitions
        n_subs = len(partitions)
        pvp = self._metadata.pvp

        # Phase sign convention from CPHD Global
        gp = self._metadata.global_params
        phase_sgn = gp.phase_sgn if gp is not None else -1

        if self._verbose:
            print(f"Stripmap PFA: {n_subs} sub-apertures, "
                  f"{self._partitioner.sub_length} pulses each, "
                  f"stride {self._partitioner.stride}")
            print(f"  PhaseSGN: {phase_sgn:+d}")

        sub_images: List[np.ndarray] = []
        sub_centers: List[np.ndarray] = []  # centre SRP per sub-aperture
        sub_grids: List[Dict[str, Any]] = []

        for i, (start, end) in enumerate(partitions):
            if self._verbose:
                print(f"\n  Sub-aperture {i + 1}/{n_subs}: "
                      f"pulses [{start}:{end}] "
                      f"({end - start} pulses)")

            # ── 1. Slice metadata and signal ──
            sub_meta = create_subaperture_metadata(
                self._metadata, start, end,
            )
            sub_signal = signal[start:end, :]

            # ── 2. Compute centre SRP ──
            center_idx = (end - start) // 2
            center_srp = pvp.srp_pos[start + center_idx].copy()
            sub_centers.append(center_srp)

            # ── 3. Rephase signal to centre SRP ──
            sub_signal = self.rephase_signal(
                sub_signal, sub_meta.pvp, center_srp,
                phase_sgn=phase_sgn,
            )

            # Fix the SRP to the centre point for PFA geometry
            sub_meta.pvp.srp_pos[:] = center_srp[np.newaxis, :]

            # ── 4. Compute geometry and grid ──
            sub_geo = CollectionGeometry(sub_meta, slant=self._slant)
            sub_grid = PolarGrid(
                sub_geo,
                grid_mode=self._grid_mode,
                range_oversample=self._range_oversample,
                azimuth_oversample=self._azimuth_oversample,
            )

            if self._verbose:
                print(f"    Grid: {sub_grid.rec_n_pulses} x "
                      f"{sub_grid.rec_n_samples}")
                print(f"    Range res: {sub_grid.range_resolution:.3f} m")
                print(f"    Azimuth res: "
                      f"{sub_grid.azimuth_resolution:.3f} m")

            # ── 5. Run PFA ──
            pfa = PolarFormatAlgorithm(
                grid=sub_grid,
                interpolator=self._interpolator,
                weighting=self._weighting,
                phase_sgn=phase_sgn,
            )
            sub_image = pfa.form_image(sub_signal, sub_geo)
            sub_images.append(sub_image)

            grid_info = pfa.get_output_grid()
            grid_info['center_srp'] = center_srp
            sub_grids.append(grid_info)

            if self._verbose:
                print(f"    Image shape: {sub_image.shape}")

        self._sub_grids = sub_grids

        # ── 6. Mosaic ──
        if self._verbose:
            print(f"\nMosaicing {n_subs} sub-apertures...")

        mosaic = self._mosaic_subapertures(
            sub_images, sub_centers, sub_grids,
        )

        if self._verbose:
            print(f"  Mosaic shape: {mosaic.shape}")

        return mosaic

    # ------------------------------------------------------------------
    # Mosaicing
    # ------------------------------------------------------------------

    def _mosaic_subapertures(
        self,
        images: List[np.ndarray],
        centers: List[np.ndarray],
        grids: List[Dict[str, Any]],
    ) -> np.ndarray:
        """Mosaic sub-aperture images into a continuous strip.

        Sub-apertures are placed on a common output grid using their
        SRP offsets.  Overlap regions are blended with a raised-cosine
        (Hann) taper for smooth transitions.

        The azimuth (row) axis corresponds to the along-track
        direction where sub-apertures tile; the range (column) axis
        is perpendicular.

        Parameters
        ----------
        images : List[np.ndarray]
            Per-sub-aperture complex images.
        centers : List[np.ndarray]
            Per-sub-aperture centre SRP in ECF.
        grids : List[Dict[str, Any]]
            Per-sub-aperture output grid dicts.

        Returns
        -------
        np.ndarray
            Mosaiced complex image.
        """
        n_subs = len(images)
        if n_subs == 1:
            return images[0]

        # Use the first sub-aperture as the reference grid
        ref_grid = grids[0]
        az_ss = ref_grid['az_ss']  # azimuth sample spacing (1/m → m)
        rg_ss = ref_grid['rg_ss']
        n_range = images[0].shape[1]

        # Compute along-track offsets between sub-aperture centers
        # Project SRP offsets onto azimuth direction
        # Use first sub-aperture's azimuth unit vector
        ref_center = centers[0]

        # Azimuth direction: along-track (approximate as SRP drift)
        if n_subs > 1:
            az_dir = centers[-1] - centers[0]
            az_dir = az_dir / norm(az_dir)
        else:
            az_dir = np.array([0.0, 1.0, 0.0])

        # Compute azimuth pixel offsets for each sub-aperture
        az_offsets_m = np.array([
            np.dot(c - ref_center, az_dir) for c in centers
        ])
        az_offsets_px = az_offsets_m / az_ss

        # Determine output image size
        sub_az_sizes = np.array([img.shape[0] for img in images])
        min_offset = az_offsets_px.min()
        max_offset = az_offsets_px.max()
        total_az = int(np.ceil(
            max_offset - min_offset + sub_az_sizes[-1],
        ))

        # Allocate output (weighted accumulation)
        output = np.zeros((total_az, n_range), dtype=images[0].dtype)
        weight_sum = np.zeros((total_az, n_range), dtype=np.float64)

        for i in range(n_subs):
            img = images[i]
            n_az_i = img.shape[0]
            n_rg_i = img.shape[1]

            # Azimuth placement
            az_start = int(np.round(az_offsets_px[i] - min_offset))
            az_end = az_start + n_az_i

            # Range placement (centre-align if sizes differ)
            rg_start = (n_range - n_rg_i) // 2
            rg_end = rg_start + n_rg_i

            # Clamp to output bounds
            out_az_start = max(0, az_start)
            out_az_end = min(total_az, az_end)
            out_rg_start = max(0, rg_start)
            out_rg_end = min(n_range, rg_end)

            # Corresponding source indices
            src_az_start = out_az_start - az_start
            src_az_end = src_az_start + (out_az_end - out_az_start)
            src_rg_start = out_rg_start - rg_start
            src_rg_end = src_rg_start + (out_rg_end - out_rg_start)

            # Hann taper in azimuth for blending
            taper = self._hann_taper(n_az_i)
            taper_slice = taper[src_az_start:src_az_end]
            weight_2d = taper_slice[:, np.newaxis] * np.ones(
                out_rg_end - out_rg_start,
            )

            output[
                out_az_start:out_az_end,
                out_rg_start:out_rg_end,
            ] += (
                img[src_az_start:src_az_end, src_rg_start:src_rg_end]
                * weight_2d
            )
            weight_sum[
                out_az_start:out_az_end,
                out_rg_start:out_rg_end,
            ] += weight_2d

        # Normalize by total weight (avoid division by zero)
        mask = weight_sum > 0
        output[mask] /= weight_sum[mask]

        return output

    @staticmethod
    def _hann_taper(n: int) -> np.ndarray:
        """Raised-cosine (Hann) taper for overlap blending.

        Parameters
        ----------
        n : int
            Length of the taper array.

        Returns
        -------
        np.ndarray
            Taper values in [0, 1], shape ``(n,)``.
        """
        if n <= 1:
            return np.ones(n)
        return 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n) / (n - 1)))

    # ------------------------------------------------------------------
    # Output grid
    # ------------------------------------------------------------------

    def get_output_grid(self) -> Dict[str, Any]:
        """Return output grid parameters from the first sub-aperture.

        Returns
        -------
        Dict[str, Any]
            Grid parameters from the first sub-aperture PFA.
        """
        if self._sub_grids:
            return self._sub_grids[0].copy()
        return {}

    @property
    def partitioner(self) -> SubaperturePartitioner:
        """Access the subaperture partitioner."""
        return self._partitioner
