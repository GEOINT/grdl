# -*- coding: utf-8 -*-
"""
CPHD-to-SICD end-to-end conversion pipeline.

Provides :class:`CPHDToSICDConverter`, a single-call converter that reads a
CPHD file, selects an image formation processor (IFP) from the collection
metadata (or an explicit override), forms the complex SAR image, and writes
a SICD NITF.

The converter follows the same ``__init__ / .convert() → Path`` contract as
every other multi-stage converter in this library
(:class:`~grdl.IO.sar.sentinel1_l0.crsd_writer.Sentinel1L0ToCRSD`,
:class:`~grdl.IO.sar.sentinel1_l0.cphd_builder.CRSDtoCPHD`).  All
mathematical and metadata work is delegated to the pure
:func:`~grdl.IO.sar.cphd_to_sicd.build_sicd_metadata` function, which
is the single source of truth for PFA polynomial fits, TimeCOAPoly,
Nyquist sample-spacing correction, and every other formula fix.

Algorithm selection
-------------------
When ``algorithm=None`` (default) the converter reads
``CollectionInfo.RadarMode`` from the CPHD and picks:

* ``SPOTLIGHT``         → ``'PFA'``
* ``STRIPMAP``          → ``'RDA'``
* ``DYNAMIC STRIPMAP``  → ``'RDA'``
* (anything else)       → ``'PFA'``  (with a warning)

Pass ``algorithm='PFA'``, ``'RDA'``, ``'FFBP'``, or ``'StripmapPFA'``
to override.

Dependencies
------------
sarkit or sarpy

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
2026-05-20

Modified
--------
2026-05-20
"""

from __future__ import annotations

# Standard library
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Third-party
import numpy as np

# GRDL internal — pure metadata builder (no I/O or IFP imports)
from grdl.IO.sar.cphd_to_sicd import build_sicd_metadata

logger = logging.getLogger(__name__)

# Maps internal algorithm name → SICD ImageFormation/ImageFormAlgo tag.
_ALGO_SICD_TAG: Dict[str, str] = {
    'PFA':         'PFA',
    'RDA':         'RGAZCOMP',
    'FFBP':        'OTHER',
    'StripmapPFA': 'PFA',
}

# Maps CPHD RadarMode → default algorithm name.
_MODE_DEFAULT_ALGO: Dict[str, str] = {
    'SPOTLIGHT':        'PFA',
    'STRIPMAP':         'RDA',
    'DYNAMIC STRIPMAP': 'RDA',
}


class CPHDToSICDConverter:
    """Convert a CPHD phase-history file to a focused SICD NITF.

    Reads the CPHD, selects an IFP from the collection metadata (or an
    explicit ``algorithm`` override), forms the complex SAR image, and
    writes a SICD NITF.  All metadata derivation — PFA polynomial fits,
    ``Grid.TimeCOAPoly``, Nyquist sample-spacing correction — is
    handled by :func:`~grdl.IO.sar.cphd_to_sicd.build_sicd_metadata`.

    Parameters
    ----------
    input_path : str or Path
        Path to the input CPHD file.
    output_path : str or Path
        Destination SICD NITF path.
    algorithm : str, optional
        IFP override — one of ``'PFA'``, ``'RDA'``, ``'FFBP'``, or
        ``'StripmapPFA'``.  When *None* (default) the algorithm is
        selected from ``CollectionInfo.RadarMode``.
    channel : str, optional
        CPHD channel identifier to process.  When *None* (default) the
        first channel is used.
    slant : bool
        If True (default) use slant-plane geometry; False for
        ground-plane.
    grid_mode : str
        PFA polar-grid mode — ``'inscribed'`` (default) or
        ``'circumscribed'``.  Only used when ``algorithm='PFA'``.
    range_oversample : float
        PFA range oversampling factor (default 1.0 = Nyquist).
    azimuth_oversample : float
        PFA azimuth oversampling factor (default 1.0 = Nyquist).
    weighting : str
        K-space taper applied before the PFA 2-D FFT.  One of
        ``'uniform'`` (default), ``'taylor'``, ``'hamming'``,
        ``'hanning'``.  Also accepted by RDA and StripmapPFA as
        ``range_weighting`` / ``azimuth_weighting``.
    phase_sgn : int
        CPHD ``PhaseSGN`` convention (+1 or -1, default -1).

    Examples
    --------
    >>> converter = CPHDToSICDConverter(
    ...     input_path='scene.cphd',
    ...     output_path='scene.nitf',
    ... )
    >>> out = converter.convert()

    Explicit RDA for a stripmap collect:

    >>> converter = CPHDToSICDConverter(
    ...     'stripmap.cphd',
    ...     'stripmap.nitf',
    ...     algorithm='RDA',
    ... )
    >>> out = converter.convert()
    """

    def __init__(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        *,
        algorithm: Optional[str] = None,
        channel: Optional[str] = None,
        slant: bool = True,
        grid_mode: str = 'inscribed',
        range_oversample: float = 1.0,
        azimuth_oversample: float = 1.0,
        weighting: str = 'uniform',
        phase_sgn: int = -1,
    ) -> None:
        if algorithm is not None and algorithm not in _ALGO_SICD_TAG:
            raise ValueError(
                f"algorithm must be one of {list(_ALGO_SICD_TAG)} or None, "
                f"got {algorithm!r}"
            )
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.algorithm = algorithm
        self.channel = channel
        self.slant = slant
        self.grid_mode = grid_mode
        self.range_oversample = range_oversample
        self.azimuth_oversample = azimuth_oversample
        self.weighting = weighting
        self.phase_sgn = phase_sgn

    # ------------------------------------------------------------------
    # Public entry-point
    # ------------------------------------------------------------------

    def convert(self) -> Path:
        """Run the full CPHD-to-SICD conversion pipeline.

        Returns
        -------
        Path
            Path to the written SICD NITF file.
        """
        from grdl.IO.sar.cphd import CPHDReader
        from grdl.IO.sar.sicd_writer import SICDWriter
        from grdl.image_processing.sar.image_formation.geometry import (
            CollectionGeometry,
        )

        logger.info("CPHDToSICDConverter: %s → %s",
                    self.input_path, self.output_path)

        # 1. Read CPHD
        logger.info("[1/4] Reading CPHD")
        with CPHDReader(self.input_path) as reader:
            meta   = reader.metadata
            signal = (
                reader.read_channel(self.channel)
                if self.channel is not None
                else reader.read_full()
            )
        logger.info("  Phase history: %d pulses × %d samples",
                    *signal.shape)

        # 2. Collection geometry
        logger.info("[2/4] Building CollectionGeometry (slant=%s)",
                    self.slant)
        geo = CollectionGeometry(meta, slant=self.slant)
        logger.info(
            "  image_plane=%s  side=%s  graze=%.2f°  azimuth=%.2f°",
            geo.image_plane, geo.side_of_track,
            np.degrees(geo.graz_ang_coa),
            np.degrees(geo.azim_ang_coa),
        )

        # 3. Form image
        algo = self._select_algorithm(meta)
        sicd_tag = _ALGO_SICD_TAG[algo]
        logger.info("[3/4] Image formation with %s (SICD tag: %s)",
                    algo, sicd_tag)
        ifp = self._build_ifp(algo, meta, geo)
        image = ifp.form_image(signal, geo)
        if image.dtype != np.complex64:
            image = image.astype(np.complex64)
        grid_params = ifp.get_output_grid()
        logger.info("  Formed image: %s  dtype=%s", image.shape, image.dtype)

        # 4. Build SICD metadata and write
        logger.info("[4/4] Building SICD metadata and writing NITF")
        sicd_meta = build_sicd_metadata(
            meta,
            geo,
            image.shape,
            image_form_algo=sicd_tag,
            grid_params=grid_params,
        )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = SICDWriter(self.output_path, metadata=sicd_meta)
        writer.write(image)
        logger.info("Written: %s", self.output_path)
        return self.output_path

    # ------------------------------------------------------------------
    # Algorithm selection
    # ------------------------------------------------------------------

    def _select_algorithm(self, meta: Any) -> str:
        """Return the algorithm name to use for ``meta``.

        Explicit constructor override takes precedence over the
        radar-mode look-up.
        """
        if self.algorithm is not None:
            logger.info("  Algorithm: %s (explicit)", self.algorithm)
            return self.algorithm

        radar_mode = ''
        if meta.collection_info is not None:
            radar_mode = (meta.collection_info.radar_mode or '').upper()

        algo = _MODE_DEFAULT_ALGO.get(radar_mode)
        if algo is None:
            logger.warning(
                "Unrecognised radar_mode %r — defaulting to PFA.", radar_mode
            )
            algo = 'PFA'
        else:
            logger.info("  Algorithm: %s (auto from radar_mode=%r)",
                        algo, radar_mode)
        return algo

    # ------------------------------------------------------------------
    # IFP construction
    # ------------------------------------------------------------------

    def _build_ifp(
        self,
        algo: str,
        meta: Any,
        geo: Any,
    ) -> Any:
        """Construct and return a configured IFP for ``algo``."""
        if algo == 'PFA':
            return self._build_pfa(meta, geo)
        if algo == 'RDA':
            return self._build_rda(meta, geo)
        if algo == 'FFBP':
            return self._build_ffbp(meta, geo)
        if algo == 'StripmapPFA':
            return self._build_stripmap_pfa(meta, geo)
        raise ValueError(f"Unknown algorithm {algo!r}")

    def _build_pfa(self, meta: Any, geo: Any) -> Any:
        from grdl.image_processing.sar import PolarGrid, PolarFormatAlgorithm

        gp = meta.global_params
        phase_sgn = (gp.phase_sgn if gp is not None else self.phase_sgn)

        grid = PolarGrid(
            geo,
            grid_mode=self.grid_mode.upper(),
            range_oversample=self.range_oversample,
            azimuth_oversample=self.azimuth_oversample,
        )
        logger.info(
            "  PolarGrid (%s): %d × %d  rng_res=%.3fm  az_res=%.3fm",
            self.grid_mode, grid.rec_n_pulses, grid.rec_n_samples,
            grid.range_resolution, grid.azimuth_resolution,
        )
        return PolarFormatAlgorithm(
            grid=grid,
            weighting=self.weighting,
            phase_sgn=phase_sgn,
        )

    def _build_rda(self, meta: Any, geo: Any) -> Any:
        from grdl.image_processing.sar.image_formation.rda import (
            RangeDopplerAlgorithm,
        )
        logger.info(
            "  RDA: rng_wt=%s  az_wt=%s",
            self.weighting, self.weighting,
        )
        return RangeDopplerAlgorithm(
            meta,
            range_weighting=self.weighting,
            azimuth_weighting=self.weighting,
        )

    def _build_ffbp(self, meta: Any, geo: Any) -> Any:
        from grdl.image_processing.sar.image_formation.ffbp import (
            FastBackProjection,
        )
        logger.info("  FFBP: defaults")
        return FastBackProjection()

    def _build_stripmap_pfa(self, meta: Any, geo: Any) -> Any:
        from grdl.image_processing.sar import StripmapPFA
        logger.info("  StripmapPFA: defaults")
        return StripmapPFA(meta)
