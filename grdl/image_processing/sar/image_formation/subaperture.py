# -*- coding: utf-8 -*-
"""
Subaperture Partitioner - Divide stripmap apertures for PFA processing.

Computes overlapping sub-aperture boundaries from CPHD PVP arrays.
Each sub-aperture is short enough to be treated as a spotlight-like
collection for the Polar Format Algorithm, with a well-defined
phase error budget controlling the partition length.

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
from typing import List, Optional, Tuple

# Third-party
import numpy as np
from numpy.linalg import norm

# GRDL internal
from grdl.IO.models.cphd import CPHDMetadata


# Speed of light (m/s)
_C = 299792458.0


class SubaperturePartitioner:
    """Partition a stripmap aperture into overlapping sub-apertures.

    Each sub-aperture spans a contiguous block of pulses small enough
    for a single PFA pass.  Adjacent sub-apertures overlap by a
    configurable fraction to enable smooth mosaicing.

    Parameters
    ----------
    metadata : CPHDMetadata
        Full CPHD metadata with populated PVP arrays.
    subaperture_pulses : int, optional
        Fixed number of pulses per sub-aperture.  If None,
        automatically sized using ``max_phase_error``.
    max_phase_error : float
        Maximum tolerable quadratic phase error (radians) for
        automatic sub-aperture sizing.  Default ``pi/4``.
    overlap_fraction : float
        Fraction of sub-aperture length shared with the next
        sub-aperture.  0.5 = 50 % overlap.  Default 0.5.
    min_pulses : int
        Minimum pulses per sub-aperture.  Default 64.

    Attributes
    ----------
    n_pulses : int
        Total number of pulses in the collection.
    sub_length : int
        Computed sub-aperture length in pulses.
    stride : int
        Pulse stride between successive sub-aperture starts.
    partitions : List[Tuple[int, int]]
        ``(start, end)`` pulse index pairs for each sub-aperture.
    """

    def __init__(
        self,
        metadata: CPHDMetadata,
        subaperture_pulses: Optional[int] = None,
        max_phase_error: float = np.pi / 4,
        overlap_fraction: float = 0.5,
        min_pulses: int = 64,
    ) -> None:
        if metadata.pvp is None:
            raise ValueError(
                "CPHDMetadata must have populated PVP arrays"
            )
        self._metadata = metadata
        self._pvp = metadata.pvp
        self.n_pulses = self._pvp.num_vectors
        self.overlap_fraction = overlap_fraction
        self.min_pulses = min_pulses

        if subaperture_pulses is not None:
            self.sub_length = max(min_pulses, subaperture_pulses)
        else:
            self.sub_length = self._auto_size(max_phase_error)

        self.stride = max(
            1, int(self.sub_length * (1.0 - overlap_fraction)),
        )
        self.partitions = self._compute_partitions()

    # ------------------------------------------------------------------
    # Automatic sub-aperture sizing
    # ------------------------------------------------------------------

    def _auto_size(self, max_phase_error: float) -> int:
        """Compute sub-aperture length from quadratic phase error budget.

        For a stripmap collection the scene reference point drifts
        per pulse.  Within a sub-aperture of *N* pulses the SRP is
        frozen at the centre.  The quadratic phase error at the
        sub-aperture edges relative to this frozen SRP is:

            Δφ ≈ (4π / λ) · (v² · Δt² / (8R))

        where Δt is half the sub-aperture duration, v is platform
        speed, R is slant range, and λ is wavelength.  We solve for
        Δt such that Δφ ≤ ``max_phase_error``.

        Parameters
        ----------
        max_phase_error : float
            Maximum tolerable quadratic phase error (radians).

        Returns
        -------
        int
            Sub-aperture length in pulses.
        """
        pvp = self._pvp

        # Derive physical parameters from PVP
        mid_times = 0.5 * (pvp.tx_time + pvp.rcv_time)
        avg_dt = float(np.mean(np.diff(mid_times)))

        # Platform speed
        vel = np.mean([pvp.tx_vel, pvp.rcv_vel], axis=0)
        speed = float(np.mean(norm(vel, axis=1)))

        # Slant range
        arp = np.mean([pvp.tx_pos, pvp.rcv_pos], axis=0)
        slant_range = float(np.mean(norm(arp - pvp.srp_pos, axis=1)))

        # Wavelength from center frequency
        gp = self._metadata.global_params
        if gp is not None and gp.center_frequency is not None:
            wavelength = _C / gp.center_frequency
        else:
            fc = 0.5 * float(pvp.fx1[0] + pvp.fx2[0])
            wavelength = _C / fc

        # Solve: max_phase_error = (4π / λ) · (v² · Δt² / (8R))
        # Δt = sqrt(max_phase_error · 2 · λ · R / (π · v²))
        dt_half = np.sqrt(
            max_phase_error * 2.0 * wavelength * slant_range
            / (np.pi * speed ** 2)
        )
        # Full sub-aperture duration = 2 * dt_half
        n_pulses = int(2.0 * dt_half / avg_dt)

        return max(self.min_pulses, n_pulses)

    # ------------------------------------------------------------------
    # Partition computation
    # ------------------------------------------------------------------

    def _compute_partitions(self) -> List[Tuple[int, int]]:
        """Compute sub-aperture start/end pairs.

        Returns
        -------
        List[Tuple[int, int]]
            Each tuple is ``(start, end)`` pulse indices.  The last
            sub-aperture is extended or shifted to include the final
            pulses.
        """
        parts: List[Tuple[int, int]] = []
        start = 0

        while start < self.n_pulses:
            end = min(start + self.sub_length, self.n_pulses)

            # If the remaining tail is shorter than half a sub-aperture,
            # extend the last partition to cover it.
            remaining = self.n_pulses - end
            if 0 < remaining < self.stride:
                end = self.n_pulses

            parts.append((start, end))

            if end >= self.n_pulses:
                break
            start += self.stride

        return parts

    @property
    def num_subapertures(self) -> int:
        """Number of sub-apertures."""
        return len(self.partitions)
