# -*- coding: utf-8 -*-
"""
CPHD-to-SICD Example - Convert CPHD phase history to a SICD image file.

Reads a CPHD file, selects an image formation processor (IFP) based on
the collection mode recorded in the metadata (SPOTLIGHT → PFA, STRIPMAP /
DYNAMIC STRIPMAP → RDA), forms the complex SAR image, writes it as a SICD
NITF, and optionally opens the result in grdk-viewer.

All parameters needed for image formation (waveform, PVP, frequency band,
geometry) are derived entirely from the CPHD metadata — no sensor-specific
knowledge is required beyond what the CPHD file itself contains.

Output path defaults to the input path with:
  - the file extension replaced by ``.nitf``
  - any occurrence of ``cphd`` in the filename stem replaced by ``sicd``

An optional YAML configuration file can override the auto-selected algorithm
and set algorithm-specific tuning parameters.  See the ``--config`` argument
and the example YAML block below.

Example YAML config (cphd_to_sicd.yaml)
-----------------------------------------
::

    # Algorithm override — omit to use metadata-driven auto-selection.
    # Choices: PFA | RDA | FFBP | StripmapPFA
    algorithm: PFA

    # slant: true uses slant plane geometry (default); false uses ground plane.
    slant: true

    pfa:
      grid_mode: inscribed        # inscribed | circumscribed
      range_oversample: 1.0       # 1.0 = Nyquist
      azimuth_oversample: 1.0
      weighting: uniform          # uniform | taylor | hamming | hanning
      interpolator:
        type: polyphase           # polyphase | kaiser_sinc (default)
        kernel_length: 8          # filter taps
        num_phases: 128           # polyphase branches
        prototype: kaiser         # kaiser | remez

    rda:
      range_weighting: taylor     # uniform | taylor | hamming | hanning
      azimuth_weighting: uniform
      block_size: null            # null = full aperture, 'auto', or integer
      overlap: 0.5                # subaperture overlap fraction

    ffbp:
      leaf_size: 8
      n_angular: 128
      apply_amp_sf: true

    stripmap_pfa:
      subaperture_pulses: 4096
      overlap: 0.3

Usage
-----
::

    python cphd_to_sicd_example.py <cphd_file>
    python cphd_to_sicd_example.py <cphd_file> --output formed.sicd
    python cphd_to_sicd_example.py <cphd_file> --config cphd_to_sicd.yaml
    python cphd_to_sicd_example.py <cphd_file> --no-viewer
    python cphd_to_sicd_example.py --help

Dependencies
------------
sarkit or sarpy
pyyaml (optional, required only when --config is used)

Author
------
Jason Fritz
43161141+stryder-vtx@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2026 geoint.org
See LICENSE file for full text.

Created
-------
2026-05-14

Modified
--------
2026-05-14
"""

# Standard library
import argparse
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Third-party
import numpy as np

# GRDL — resolve package root when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from grdl.IO.sar import CPHDReader, SICDWriter
from grdl.IO.sar.cphd_to_sicd import build_sicd_metadata
from grdl.image_processing.sar import CollectionGeometry
from grdl.image_processing.sar.image_formation.base import (
    ImageFormationAlgorithm,
)


# ── Constants ────────────────────────────────────────────────────────

# Maps internal algorithm names to the SICD ImageFormation/ImageFormAlgo tag.
# RDA outputs in range-Doppler (RGZERO) domain → RGAZCOMP.
# FFBP is not a named SICD algorithm → OTHER.
_ALGO_SICD_TAG: Dict[str, str] = {
    'PFA':          'PFA',
    'RDA':          'RGAZCOMP',
    'FFBP':         'OTHER',
    'StripmapPFA':  'PFA',
}

# Maps CPHD radar_mode to default algorithm name.
_MODE_DEFAULT_ALGO: Dict[str, str] = {
    'SPOTLIGHT':        'PFA',
    'STRIPMAP':         'RDA',
    'DYNAMIC STRIPMAP': 'RDA',
}

# Config sub-key for each algorithm's tuning section.
_ALGO_CFG_KEY: Dict[str, str] = {
    'PFA':          'pfa',
    'RDA':          'rda',
    'FFBP':         'ffbp',
    'StripmapPFA':  'stripmap_pfa',
}


# ── CLI ──────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Convert a CPHD file to SICD using an IFP selected from "
            "the collection metadata.  The output file is always written."
        ),
    )
    parser.add_argument(
        'input',
        type=Path,
        help='Path to the input CPHD file.',
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=None,
        help=(
            'Output SICD path.  Defaults to the input path with the '
            'extension replaced by .nitf and any "cphd" in the stem '
            'replaced by "sicd".'
        ),
    )
    parser.add_argument(
        '--config', '-c',
        type=Path,
        default=None,
        help=(
            'Optional YAML config file for algorithm selection and '
            'tuning parameters.  See module docstring for the format.'
        ),
    )
    parser.add_argument(
        '--no-viewer',
        action='store_true',
        default=False,
        help='Skip launching grdk-viewer after writing the SICD file.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        default=False,
        help=(
            'Print geometry and grid diagnostics then exit without '
            'running image formation or writing any output.'
        ),
    )
    return parser.parse_args()


# ── Output path ──────────────────────────────────────────────────────


def derive_output_path(input_path: Path) -> Path:
    """Derive the default SICD output path from the CPHD input path.

    Replaces the file extension with ``.sicd`` and substitutes any
    occurrence of ``cphd`` (case-insensitive) in the stem with ``sicd``.

    Parameters
    ----------
    input_path : Path
        Path to the input CPHD file.

    Returns
    -------
    Path
        Derived output path in the same directory as the input.
    """
    stem = re.sub(r'(?i)cphd', 'sicd', input_path.stem)
    return input_path.with_name(stem + '.nitf')


# ── Config loading ───────────────────────────────────────────────────


def load_config(config_path: Optional[Path]) -> Dict[str, Any]:
    """Load the optional YAML tuning config.

    Parameters
    ----------
    config_path : Path or None
        Path to the YAML file, or None to return an empty dict.

    Returns
    -------
    dict
        Parsed config dict (empty when no file is provided).
    """
    if config_path is None:
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "pyyaml is required to use --config: pip install pyyaml"
        ) from exc
    with open(config_path) as fh:
        return yaml.safe_load(fh) or {}


# ── Algorithm selection ───────────────────────────────────────────────


def select_algorithm(meta: Any, config: Dict[str, Any]) -> str:
    """Choose the IFP algorithm name.

    Precedence: explicit ``algorithm`` key in config > CPHD radar_mode.

    Parameters
    ----------
    meta : CPHDMetadata
        CPHD metadata from the reader.
    config : dict
        Loaded YAML config (may be empty).

    Returns
    -------
    str
        One of ``'PFA'``, ``'RDA'``, ``'FFBP'``, ``'StripmapPFA'``.
    """
    if 'algorithm' in config:
        algo = config['algorithm']
        if algo not in _ALGO_SICD_TAG:
            raise ValueError(
                f"Unknown algorithm {algo!r} in config.  "
                f"Valid choices: {list(_ALGO_SICD_TAG)}"
            )
        print(f"  Algorithm: {algo} (from config)")
        return algo

    radar_mode = ''
    if meta.collection_info is not None:
        radar_mode = (meta.collection_info.radar_mode or '').upper()

    algo = _MODE_DEFAULT_ALGO.get(radar_mode, 'PFA')
    if radar_mode not in _MODE_DEFAULT_ALGO:
        print(
            f"  Warning: unrecognised radar_mode {radar_mode!r}, "
            "defaulting to PFA."
        )
    else:
        print(f"  Algorithm: {algo} (auto from radar_mode={radar_mode!r})")
    return algo


# ── IFP builders ─────────────────────────────────────────────────────
#
# Each builder constructs and returns a fully configured IFP object that
# implements ImageFormationAlgorithm (form_image + get_output_grid).
# The convert() function calls form_image() and get_output_grid() itself
# so they are only invoked once.


def _build_pfa_interpolator(interp_cfg: Dict[str, Any]) -> Any:
    """Construct a PFA interpolator from a config dict.

    Parameters
    ----------
    interp_cfg : dict
        ``pfa.interpolator`` sub-dict.  Recognised keys:

        ``type`` : str
            ``'polyphase'`` or ``'kaiser_sinc'``.
            Default ``'kaiser_sinc'`` (PFA built-in default).
        ``kernel_length`` : int
            Filter kernel length in taps.
        ``num_phases`` : int
            Number of polyphase branches (``polyphase`` only).
        ``beta`` : float
            Kaiser window shape parameter.
        ``prototype`` : str
            ``'kaiser'`` or ``'remez'`` (``polyphase`` only).

    Returns
    -------
    callable or None
        Interpolator instance, or ``None`` to use the PFA default.
    """
    if not interp_cfg:
        return None

    interp_type = interp_cfg.get('type', 'kaiser_sinc')

    if interp_type == 'polyphase':
        from grdl.interpolation import PolyphaseInterpolator
        return PolyphaseInterpolator(
            kernel_length=int(interp_cfg.get('kernel_length', 8)),
            num_phases=int(interp_cfg.get('num_phases', 128)),
            beta=float(interp_cfg.get('beta', 5.0)),
            prototype=str(interp_cfg.get('prototype', 'kaiser')),
        )
    if interp_type == 'kaiser_sinc':
        from grdl.interpolation import KaiserSincInterpolator
        return KaiserSincInterpolator(
            kernel_length=int(interp_cfg.get('kernel_length', 8)),
            beta=float(interp_cfg.get('beta', 5.0)),
        )
    raise ValueError(
        f"Unknown interpolator type {interp_type!r}.  "
        "Valid choices: 'polyphase', 'kaiser_sinc'."
    )


def build_pfa(
    meta: Any,
    geo: CollectionGeometry,
    cfg: Dict[str, Any],
) -> ImageFormationAlgorithm:
    """Build a PolarFormatAlgorithm for spotlight collections.

    Parameters
    ----------
    meta : CPHDMetadata
        CPHD metadata.
    geo : CollectionGeometry
        Pre-computed collection geometry.
    cfg : dict
        ``pfa`` sub-dict from the YAML config (may be empty).

    Returns
    -------
    PolarFormatAlgorithm
        Configured IFP ready for ``form_image(signal, geo)``.
    """
    from grdl.image_processing.sar import PolarGrid, PolarFormatAlgorithm

    grid_mode        = cfg.get('grid_mode', 'inscribed')
    range_oversample = float(cfg.get('range_oversample', 1.0))
    az_oversample    = float(cfg.get('azimuth_oversample', 1.0))
    weighting        = cfg.get('weighting', 'uniform')
    interpolator     = _build_pfa_interpolator(cfg.get('interpolator') or {})

    grid = PolarGrid(
        geo,
        grid_mode=grid_mode,
        range_oversample=range_oversample,
        azimuth_oversample=az_oversample,
    )
    d = grid._diag
    print(f"  PolarGrid ({grid_mode}): {grid.rec_n_pulses} x {grid.rec_n_samples}  "
          f"rng_res={grid.range_resolution:.3f}m  az_res={grid.azimuth_resolution:.3f}m")
    print(f"  [diag] input: {d['npulses']} pulses x {d['nsamples']} samples")
    print(f"  [diag] proc_bw={d['proc_bw']:.3e} Hz  kv_span={d['kv_span']:.6e} 1/m  "
          f"ku_span={d['ku_span']:.6e} 1/m")
    print(f"  [diag] scene_range={d['scene_range']:.1f}m  scene_az={d['scene_az']:.1f}m  "
          f"graze={d['mean_graze_deg']:.2f}°")

    gp        = meta.global_params
    phase_sgn = gp.phase_sgn if gp is not None else -1

    kwargs: Dict[str, Any] = dict(grid=grid, weighting=weighting, phase_sgn=phase_sgn)
    if interpolator is not None:
        kwargs['interpolator'] = interpolator

    return PolarFormatAlgorithm(**kwargs)


def build_rda(
    meta: Any,
    geo: CollectionGeometry,
    cfg: Dict[str, Any],
) -> ImageFormationAlgorithm:
    """Build a RangeDopplerAlgorithm for stripmap collections.

    Parameters
    ----------
    meta : CPHDMetadata
        CPHD metadata (RDA reads PVP arrays directly from it).
    geo : CollectionGeometry
        Pre-computed collection geometry (passed through to form_image).
    cfg : dict
        ``rda`` sub-dict from the YAML config (may be empty).

    Returns
    -------
    RangeDopplerAlgorithm
        Configured IFP ready for ``form_image(signal, geo)``.
    """
    from grdl.image_processing.sar.image_formation.rda import (
        RangeDopplerAlgorithm,
    )

    range_weighting   = cfg.get('range_weighting', 'uniform')
    azimuth_weighting = cfg.get('azimuth_weighting', 'uniform')
    block_size        = cfg.get('block_size', None)
    overlap           = float(cfg.get('overlap', 0.5))

    print(f"  RDA: rng_wt={range_weighting}  az_wt={azimuth_weighting}  "
          f"block_size={block_size}  overlap={overlap}")
    return RangeDopplerAlgorithm(
        meta,
        range_weighting=range_weighting,
        azimuth_weighting=azimuth_weighting,
        block_size=block_size,
        overlap=overlap,
    )


def build_ffbp(
    meta: Any,
    geo: CollectionGeometry,
    cfg: Dict[str, Any],
) -> ImageFormationAlgorithm:
    """Build a FastBackProjection IFP.

    Parameters
    ----------
    meta : CPHDMetadata
        CPHD metadata.
    geo : CollectionGeometry
        Pre-computed collection geometry.
    cfg : dict
        ``ffbp`` sub-dict from the YAML config (may be empty).

    Returns
    -------
    FastBackProjection
        Configured IFP ready for ``form_image(signal, geo)``.
    """
    from grdl.image_processing.sar.image_formation.ffbp import (
        FastBackProjection,
    )

    leaf_size    = int(cfg.get('leaf_size', 8))
    n_angular    = int(cfg.get('n_angular', 128))
    apply_amp_sf = bool(cfg.get('apply_amp_sf', True))

    print(f"  FFBP: leaf_size={leaf_size}  n_angular={n_angular}  "
          f"apply_amp_sf={apply_amp_sf}")
    return FastBackProjection(
        leaf_size=leaf_size,
        n_angular=n_angular,
        apply_amp_sf=apply_amp_sf,
    )


def build_stripmap_pfa(
    meta: Any,
    geo: CollectionGeometry,
    cfg: Dict[str, Any],
) -> ImageFormationAlgorithm:
    """Build a StripmapPFA IFP (subaperture PFA mosaic).

    Parameters
    ----------
    meta : CPHDMetadata
        CPHD metadata.
    geo : CollectionGeometry
        Pre-computed collection geometry.
    cfg : dict
        ``stripmap_pfa`` sub-dict from the YAML config (may be empty).

    Returns
    -------
    StripmapPFA
        Configured IFP ready for ``form_image(signal, geo)``.
    """
    from grdl.image_processing.sar import StripmapPFA

    overlap = float(cfg.get('overlap', 0.3))
    kwargs: Dict[str, Any] = {'overlap': overlap}
    if 'subaperture_pulses' in cfg:
        kwargs['subaperture_pulses'] = int(cfg['subaperture_pulses'])

    print(f"  StripmapPFA: overlap={overlap}  "
          f"subaperture_pulses={cfg.get('subaperture_pulses', 'auto')}")
    return StripmapPFA(meta, **kwargs)


_BUILDERS = {
    'PFA':          build_pfa,
    'RDA':          build_rda,
    'FFBP':         build_ffbp,
    'StripmapPFA':  build_stripmap_pfa,
}


# ── Main pipeline ────────────────────────────────────────────────────


def convert(
    input_path: Path,
    output_path: Optional[Path] = None,
    config_path: Optional[Path] = None,
    launch_viewer: bool = True,
    dry_run: bool = False,
) -> Optional[Path]:
    """Full CPHD-to-SICD conversion pipeline.

    Parameters
    ----------
    input_path : Path
        Path to the input CPHD file.
    output_path : Path, optional
        Destination SICD path.  Derived from ``input_path`` when omitted.
    config_path : Path, optional
        Optional YAML tuning config.
    launch_viewer : bool
        If True (default), open the written SICD in grdk-viewer.
    dry_run : bool
        If True, run the conversion pipeline without writing the SICD file
        or launching the viewer.

    Returns
    -------
    Optional[Path]
        Path of the written SICD file, or ``None`` when ``dry_run=True``.
    """
    t0 = time.perf_counter()

    # ── Resolve output path ──────────────────────────────────────
    if output_path is None:
        output_path = derive_output_path(input_path)
    print(f"Input : {input_path}")
    print(f"Output: {output_path}")

    # ── Load config ──────────────────────────────────────────────
    config = load_config(config_path)
    if config_path is not None:
        print(f"Config: {config_path}")

    # ── Read CPHD ────────────────────────────────────────────────
    print("\n[1/5] Reading CPHD...")
    with CPHDReader(input_path) as reader:
        meta         = reader.metadata
        nrows, ncols = reader.get_shape()
        print(f"  Phase history: {nrows} pulses x {ncols} samples")
        if meta.collection_info is not None:
            ci = meta.collection_info
            print(f"  Collector : {ci.collector_name or 'unknown'}")
            print(f"  Radar mode: {ci.radar_mode or 'unknown'}")
        signal = reader.read_full()
    t_read = time.perf_counter()
    print(f"  Read time: {t_read - t0:.2f}s")

    # ── Select algorithm ─────────────────────────────────────────
    print("\n[2/5] Selecting algorithm...")
    algo     = select_algorithm(meta, config)
    algo_cfg = config.get(_ALGO_CFG_KEY[algo], {})
    sicd_tag = _ALGO_SICD_TAG[algo]

    # ── Collection geometry ──────────────────────────────────────
    print("\n[3/5] Computing collection geometry...")
    slant = bool(config.get('slant', True))
    geo   = CollectionGeometry(meta, slant=slant)
    print(f"  Image plane  : {geo.image_plane}")
    print(f"  Side of track: {geo.side_of_track}")
    print(f"  Graze (COA)  : {np.degrees(geo.graz_ang_coa):.2f}°")
    print(f"  Azimuth (COA): {np.degrees(geo.azim_ang_coa):.2f}°")
    t_geo = time.perf_counter()
    print(f"  Geometry time: {t_geo - t_read:.2f}s")

    # ── Build IFP (and optionally stop for dry-run) ──────────────
    print(f"\n[4/5] Forming image with {algo}...")
    ifp   = _BUILDERS[algo](meta, geo, algo_cfg)

    if dry_run:
        print("\n  --dry-run: stopping after grid diagnostics.")
        return None

    image = ifp.form_image(signal, geo)
    if image.dtype != np.complex64:
        image = image.astype(np.complex64)
    grid_params = ifp.get_output_grid()
    print(f"  Image shape: {image.shape}  dtype: {image.dtype}")
    t_ifp = time.perf_counter()
    print(f"  IFP time: {t_ifp - t_geo:.2f}s")

    # ── Build SICD metadata and write ────────────────────────────
    print(f"\n[5/5] Writing SICD ({sicd_tag})...")
    sicd_meta = build_sicd_metadata(
        meta,
        geo,
        image.shape,
        image_form_algo=sicd_tag,
        grid_params=grid_params,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = SICDWriter(output_path, metadata=sicd_meta)
    writer.write(image)
    t_write = time.perf_counter()
    print(f"  Write time: {t_write - t_ifp:.2f}s")

    print(f"\nTotal: {t_write - t0:.2f}s  →  {output_path}")

    # ── Launch viewer ────────────────────────────────────────────
    if launch_viewer:
        print("\nLaunching grdk-viewer...")
        subprocess.Popen(
            ['grdk-viewer', str(output_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    return output_path


# ── Entry point ──────────────────────────────────────────────────────


if __name__ == '__main__':
    args = parse_args()
    convert(
        input_path=args.input,
        output_path=args.output,
        config_path=args.config,
        launch_viewer=not args.no_viewer,
        dry_run=args.dry_run,
    )
