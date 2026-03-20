#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare two SIDD collects — chip and orthorectify at a common location.

Loads two SIDD products covering the same ground location (different look
angles), extracts ground-extent chips centered on a target point, and
orthorectifies both to a shared ENU meter grid at the coarsest sample
spacing of the two.  Produces two figures: raw chips and orthorectified.

Uses OrthoBuilder with explicit ENUGrid and with_reader for ortho — no
manual chip wrapper needed.

Author
------
Duane Smalley
170194430+DDSmalls@users.noreply.github.com

Created
-------
2026-03-19

Modified
--------
2026-03-19
"""

# ── Configuration (edit these) ───────────────────────────────────────

FILEPATH_A = '/Users/duanesmalley/SAR_DATA/SIDD/2025-06-11-08-42-52_UMBRA-10_SIDD.nitf'
FILEPATH_B = '/Users/duanesmalley/SAR_DATA/SIDD/2024-02-11-19-53-54_UMBRA-05_SIDD.nitf'

CENTER_LAT = -26.094455437019473
CENTER_LON = 29.47239658007181
EXTENT_M   = 500.0          # ground extent in meters
N_COMPONENTS = 48            # PCA components to keep for low-fi reconstruction
N_PATCHES    = 50           # total PCA basis vectors to compute
REVERSE_PCA  = True         # True = keep LAST N (fine detail); False = keep FIRST N (smooth)
INTERP       = 'bilinear'  # 'nearest', 'bilinear', or 'bicubic'
DTED_PATH  = '/Volumes/PRO-G40/terrain/FABDEM/'
GEOID_PATH = '/Volumes/PRO-G40/terrain/us_nga_egm08_25.tif'

# ─────────────────────────────────────────────────────────────────────

# Standard library
import sys
from pathlib import Path

# Third-party
import numpy as np

# GRDL
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from grdl.IO.sar import SIDDReader
from grdl.geolocation.sar.sidd import SIDDGeolocation
from grdl.geolocation.coordinates import enu_to_geodetic
from grdl.geolocation.elevation.constant import ConstantElevation
from grdl.image_processing.ortho import OrthoBuilder, ENUGrid


# ── Helpers ──────────────────────────────────────────────────────────


def _get_sample_spacing(meta):
    """Return (row_ss, col_ss) in meters from SIDD measurement metadata."""
    meas = meta.measurement
    if meas and meas.plane_projection and meas.plane_projection.sample_spacing:
        return (meas.plane_projection.sample_spacing.row,
                meas.plane_projection.sample_spacing.col)
    return 1.0, 1.0


def _load_and_chip(filepath, center_lat, center_lon, extent_m,
                   dted_path, geoid_path):
    """Load a SIDD, build geolocation, extract a ground-extent chip.

    Returns
    -------
    dict with keys: chip, img, meta, geo, region, row_ss, col_ss,
                    center_row, center_col, center_h, name, filepath
    """
    filepath = Path(filepath)
    half = extent_m / 2.0
    name = filepath.name

    print(f"\n{'='*60}")
    print(f"Loading: {name}")

    with SIDDReader(filepath) as reader:
        meta = reader.metadata
        rows, cols = meta.rows, meta.cols
        row_ss, col_ss = _get_sample_spacing(meta)
        print(f"  Image size: {rows} x {cols}")
        print(f"  Sample spacing: row={row_ss:.3f} m, col={col_ss:.3f} m")

        # Build geolocation once
        geo = SIDDGeolocation(meta, refine=True)

        # Load and attach DEM
        if dted_path is not None:
            from grdl.geolocation.elevation import open_elevation
            cl, co, _ = geo.image_to_latlon(
                float(rows // 2), float(cols // 2))
            try:
                elev = open_elevation(
                    dted_path, geoid_path=geoid_path,
                    location=(float(cl), float(co)),
                    fallback_height=geo._default_hae)
                if not isinstance(elev, ConstantElevation):
                    geo.elevation = elev
            except FileNotFoundError as e:
                print(f"  WARNING: {e}")

        print(f"  Projection: {geo.projection_type}")
        print(f"  R/Rdot: {geo.has_rdot}")
        print(f"  Default HAE: {geo._default_hae:.1f} m")
        print(f"  DEM: {type(geo.elevation).__name__ if geo.elevation else 'none (using scene height)'}")

        # Map target lat/lon to pixel
        cr, cc = geo.latlon_to_image(center_lat, center_lon)
        cr = float(np.asarray(cr).ravel()[0])
        cc = float(np.asarray(cc).ravel()[0])
        center_row = int(round(cr))
        center_col = int(round(cc))
        center_row = max(0, min(center_row, rows - 1))
        center_col = max(0, min(center_col, cols - 1))
        _, _, center_h = geo.image_to_latlon(
            float(center_row), float(center_col))
        center_h = float(center_h)
        print(f"  Target pixel: ({center_row}, {center_col}), "
              f"h={center_h:.1f} m")

        # Convert ground extent to pixel extent via ENU corners
        east_off = np.array([-half, half, -half, half])
        north_off = np.array([-half, -half, half, half])
        corner_lats, corner_lons, _ = enu_to_geodetic(
            east_off, north_off, np.zeros(4),
            ref_lat=center_lat, ref_lon=center_lon, ref_alt=center_h,
        )
        corner_rows, corner_cols = geo.latlon_to_image(
            corner_lats, corner_lons, center_h)

        row_min = max(0, int(np.floor(np.min(corner_rows))))
        row_max = min(rows, int(np.ceil(np.max(corner_rows))))
        col_min = max(0, int(np.floor(np.min(corner_cols))))
        col_max = min(cols, int(np.ceil(np.max(corner_cols))))
        chip_rows = row_max - row_min
        chip_cols = col_max - col_min

        if chip_rows <= 0 or chip_cols <= 0:
            raise ValueError(
                f"Empty chip bounds for {name} — center maps outside image")

        print(f"  Chip: {chip_rows} x {chip_cols} px "
              f"[{row_min}:{row_max}, {col_min}:{col_max}]")

        # Read chip for display
        chip = reader.read_chip(row_min, row_max, col_min, col_max)

    # Convert to float32 display image
    if chip.ndim == 3:
        if chip.shape[0] == 3:
            img = (0.299 * chip[0].astype(np.float32)
                   + 0.587 * chip[1].astype(np.float32)
                   + 0.114 * chip[2].astype(np.float32))
        else:
            img = chip[0].astype(np.float32)
    else:
        img = chip.astype(np.float32)

    return {
        'chip': chip, 'img': img,
        'meta': meta, 'geo': geo,
        'region': (row_min, row_max, col_min, col_max),
        'row_ss': row_ss, 'col_ss': col_ss,
        'center_row': center_row, 'center_col': center_col,
        'center_h': center_h, 'name': name,
        'filepath': str(filepath),
    }


def _orthorectify(data, grid, interp):
    """Orthorectify via OrthoBuilder with_reader + full-image geo.

    Parameters
    ----------
    data : dict
        Output from _load_and_chip.
    grid : ENUGrid
        Pre-built ENU grid (shared across both images).
    interp : str
        Interpolation method.

    Returns
    -------
    result : OrthoResult
    """
    geo = data['geo']
    ortho_elev = (geo.elevation if geo.elevation is not None
                  else ConstantElevation(height=data['center_h']))

    with SIDDReader(data['filepath']) as reader:
        result = (
            OrthoBuilder()
            .with_reader(reader)
            .with_geolocation(geo)
            .with_elevation(ortho_elev)
            .with_interpolation(interp)
            .with_nodata(np.nan)
            .with_output_grid(grid)
            .run()
        )
    return result


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    center_lat = float(CENTER_LAT)
    center_lon = float(CENTER_LON)
    extent_m = EXTENT_M

    # ── Load and chip both files ──────────────────────────────────
    data_a = _load_and_chip(
        FILEPATH_A, center_lat, center_lon, extent_m, DTED_PATH, GEOID_PATH)
    data_b = _load_and_chip(
        FILEPATH_B, center_lat, center_lon, extent_m, DTED_PATH, GEOID_PATH)

    # ── Determine coarsest sampling ───────────────────────────────
    coarsest_row = max(data_a['row_ss'], data_b['row_ss'])
    coarsest_col = max(data_a['col_ss'], data_b['col_ss'])
    pixel_size = max(coarsest_row, coarsest_col)
    print(f"\n{'='*60}")
    print(f"Coarsest sample spacing: row={coarsest_row:.3f} m, "
          f"col={coarsest_col:.3f} m")
    print(f"Ortho pixel size: {pixel_size:.3f} m")

    # Use consistent reference point for both orthos
    ref_alt = max(data_a['center_h'], data_b['center_h'])

    # ── Build a single shared ENU grid ────────────────────────────
    half = extent_m / 2.0
    shared_grid = ENUGrid(
        ref_lat=center_lat, ref_lon=center_lon, ref_alt=ref_alt,
        min_east=-half, max_east=half,
        min_north=-half, max_north=half,
        pixel_size_east=pixel_size, pixel_size_north=pixel_size,
    )
    print(f"Shared ENU grid: {shared_grid.rows} x {shared_grid.cols} px, "
          f"[{-half:.0f}, {half:.0f}] m")

    # ── Orthorectify both ─────────────────────────────────────────
    print(f"\nOrthorectifying A: {data_a['name']}...")
    result_a = _orthorectify(data_a, shared_grid, INTERP)
    print(f"  Output: {result_a.data.shape}")

    print(f"Orthorectifying B: {data_b['name']}...")
    result_b = _orthorectify(data_b, shared_grid, INTERP)
    print(f"  Output: {result_b.data.shape}")

    # ── Extract ortho arrays ─────────────────────────────────────
    ortho_a = result_a.data
    ortho_b = result_b.data

    # ── PCA decomposition and low-fidelity reconstruction ────────
    from sklearn.decomposition import PCA

    def _pca_reconstruct(img, n_components, n_keep, reverse=False):
        """Log-domain PCA on image rows; reconstruct with selected components."""
        nan_mask = np.isnan(img)
        work = np.nan_to_num(img, nan=0.0).astype(np.float64)

        floor = np.percentile(work[work > 0], 1) if np.any(work > 0) else 1.0
        work_db = 10.0 * np.log10(np.maximum(work, floor))

        pca = PCA(n_components=n_components)
        scores = pca.fit_transform(work_db)

        if reverse:
            start = n_components - n_keep
            scores[:, :start] = 0.0
            kept_str = f"{start}:{n_components}"
        else:
            scores[:, n_keep:] = 0.0
            kept_str = f"0:{n_keep}"

        recon_db = pca.inverse_transform(scores)
        recon = np.power(10.0, recon_db / 10.0).astype(np.float32)
        recon[nan_mask] = np.nan
        return recon, pca.explained_variance_ratio_, kept_str

    mode = "LAST" if REVERSE_PCA else "FIRST"
    print(f"\nPCA decomposition (log-domain): {N_PATCHES} total, "
          f"keeping {mode} {N_COMPONENTS}...")
    pca_a, var_a, kept_a = _pca_reconstruct(
        ortho_a, N_PATCHES, N_COMPONENTS, reverse=REVERSE_PCA)
    pca_b, var_b, kept_b = _pca_reconstruct(
        ortho_b, N_PATCHES, N_COMPONENTS, reverse=REVERSE_PCA)

    if REVERSE_PCA:
        kept_var_a = np.sum(var_a[N_PATCHES - N_COMPONENTS:])
        kept_var_b = np.sum(var_b[N_PATCHES - N_COMPONENTS:])
    else:
        kept_var_a = np.sum(var_a[:N_COMPONENTS])
        kept_var_b = np.sum(var_b[:N_COMPONENTS])
    print(f"  A: components [{kept_a}], {kept_var_a*100:.1f}% variance")
    print(f"  B: components [{kept_b}], {kept_var_b*100:.1f}% variance")

    # ── Normalized Cross-Correlation ──────────────────────────────
    def _ncc_2d(a, b):
        from scipy.signal import fftconvolve
        mask_a = ~np.isnan(a)
        mask_b = ~np.isnan(b)
        a0 = np.nan_to_num(a, nan=0.0).astype(np.float64)
        b0 = np.nan_to_num(b, nan=0.0).astype(np.float64)
        a0 = a0 - np.mean(a0[mask_a]) if np.any(mask_a) else a0
        b0 = b0 - np.mean(b0[mask_b]) if np.any(mask_b) else b0
        a0[~mask_a] = 0.0
        b0[~mask_b] = 0.0
        numer = fftconvolve(a0, b0[::-1, ::-1], mode='full')
        denom = np.sqrt(np.sum(a0 ** 2) * np.sum(b0 ** 2))
        ncc = numer / denom if denom > 0 else numer * 0
        return ncc

    print("\nComputing NCC between PCA-reconstructed chips...")
    ncc = _ncc_2d(pca_a, pca_b)
    peak_idx = np.unravel_index(np.argmax(ncc), ncc.shape)
    peak_val = ncc[peak_idx]
    center = np.array(ncc.shape) // 2
    shift_row = peak_idx[0] - center[0]
    shift_col = peak_idx[1] - center[1]
    shift_m_row = shift_row * pixel_size
    shift_m_col = shift_col * pixel_size
    print(f"  NCC peak: {peak_val:.4f}")
    print(f"  Peak shift: ({shift_row}, {shift_col}) px = "
          f"({shift_m_row:.1f}, {shift_m_col:.1f}) m")

    # ── Feature matching on PCA reconstructions ───────────────────
    from grdl.coregistration.feature_match import FeatureMatchCoRegistration

    print(f"\nFeature matching on PCA-reconstructed images...")
    matcher = FeatureMatchCoRegistration(
        method='orb', transform_type='affine', max_features=5000,
        ransac_threshold=5.0, match_ratio=0.75,
    )
    reg_result = matcher.estimate(pca_a, pca_b)
    print(f"  {reg_result}")

    aligned_b = matcher.apply(pca_b, reg_result)
    print(f"  Aligned shape: {aligned_b.shape}")

    # ── Plot ──────────────────────────────────────────────────────
    print("\nPlotting...")
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    def _pclim(img):
        pos = img[img > 0]
        if pos.size == 0:
            return 0, 1
        return float(np.percentile(pos, 2)), float(np.percentile(pos, 99))

    def _cross(fig, x, y, row, col):
        """Add a red cross marker to a subplot."""
        fig.add_trace(
            go.Scatter(x=[x], y=[y], mode='markers',
                       marker=dict(symbol='cross-thin', size=14,
                                   color='red', line=dict(width=2)),
                       showlegend=False),
            row=row, col=col,
        )

    # Region tuples: (row_min, row_max, col_min, col_max)
    reg_a = data_a['region']
    reg_b = data_b['region']
    sg = shared_grid

    # ── Figure 1: Raw chips ───────────────────────────────────────
    vmin_a, vmax_a = _pclim(data_a['img'])
    vmin_b, vmax_b = _pclim(data_b['img'])

    fig1 = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"A: {data_a['name']}<br>"
            f"{data_a['img'].shape[0]}x{data_a['img'].shape[1]} px, "
            f"ss={data_a['row_ss']:.2f}x{data_a['col_ss']:.2f} m",
            f"B: {data_b['name']}<br>"
            f"{data_b['img'].shape[0]}x{data_b['img'].shape[1]} px, "
            f"ss={data_b['row_ss']:.2f}x{data_b['col_ss']:.2f} m",
        ],
    )
    fig1.add_trace(
        go.Heatmap(z=data_a['img'], zmin=vmin_a, zmax=vmax_a,
                   colorscale='Gray', showscale=True,
                   colorbar=dict(x=0.45, len=0.9)),
        row=1, col=1,
    )
    _cross(fig1, data_a['center_col'] - reg_a[2],
           data_a['center_row'] - reg_a[0], 1, 1)
    fig1.add_trace(
        go.Heatmap(z=data_b['img'], zmin=vmin_b, zmax=vmax_b,
                   colorscale='Gray', showscale=True,
                   colorbar=dict(x=1.0, len=0.9)),
        row=1, col=2,
    )
    _cross(fig1, data_b['center_col'] - reg_b[2],
           data_b['center_row'] - reg_b[0], 1, 2)
    fig1.update_xaxes(title_text="Column", row=1, col=1)
    fig1.update_yaxes(title_text="Row", autorange='reversed', row=1, col=1)
    fig1.update_xaxes(title_text="Column", row=1, col=2)
    fig1.update_yaxes(title_text="Row", autorange='reversed', row=1, col=2)
    fig1.update_layout(
        title_text=(f"Raw Chips  |  {extent_m:.0f} m extent  |  "
                    f"({center_lat:.4f}, {center_lon:.4f})"),
        width=1400, height=650,
    )
    fig1.show()

    # ── Figure 2: Orthorectified ──────────────────────────────────
    vo_a = _pclim(np.nan_to_num(ortho_a))
    vo_b = _pclim(np.nan_to_num(ortho_b))

    fig2 = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"A: {data_a['name']}<br>"
            f"{ortho_a.shape[0]}x{ortho_a.shape[1]} px, {pixel_size:.2f} m",
            f"B: {data_b['name']}<br>"
            f"{ortho_b.shape[0]}x{ortho_b.shape[1]} px, {pixel_size:.2f} m",
        ],
    )
    fig2.add_trace(
        go.Heatmap(z=np.flipud(ortho_a), zmin=vo_a[0], zmax=vo_a[1],
                   x0=sg.min_east, dx=sg.pixel_size_east,
                   y0=sg.min_north, dy=sg.pixel_size_north,
                   colorscale='Gray', showscale=True,
                   colorbar=dict(x=0.45, len=0.9)),
        row=1, col=1,
    )
    _cross(fig2, 0.0, 0.0, 1, 1)
    fig2.add_trace(
        go.Heatmap(z=np.flipud(ortho_b), zmin=vo_b[0], zmax=vo_b[1],
                   x0=sg.min_east, dx=sg.pixel_size_east,
                   y0=sg.min_north, dy=sg.pixel_size_north,
                   colorscale='Gray', showscale=True,
                   colorbar=dict(x=1.0, len=0.9)),
        row=1, col=2,
    )
    _cross(fig2, 0.0, 0.0, 1, 2)
    fig2.update_xaxes(title_text="East (m)", row=1, col=1)
    fig2.update_yaxes(title_text="North (m)", row=1, col=1)
    fig2.update_xaxes(title_text="East (m)", row=1, col=2)
    fig2.update_yaxes(title_text="North (m)", row=1, col=2)
    fig2.update_layout(
        title_text=(f"Orthorectified  |  {pixel_size:.2f} m pixels, "
                    f"{INTERP}  |  ({center_lat:.4f}, {center_lon:.4f})"),
        width=1400, height=650,
    )
    fig2.show()

    # ── Shared helpers ───────────────────────────────────────────
    def _norm_pair(a, b):
        """Normalize two images to a shared [0, 1] range.

        Uses the pooled percentiles from both images so that equal
        scene brightness maps to the same intensity in both outputs.
        This prevents false color in the difference overlay caused by
        independent stretch of each image.

        Returns (a_norm, b_norm) both clipped to [0, 1].
        """
        va = np.nan_to_num(a, nan=0.0)
        vb = np.nan_to_num(b, nan=0.0)
        pooled = np.concatenate([va[va > 0], vb[vb > 0]])
        if pooled.size == 0:
            return np.zeros_like(va), np.zeros_like(vb)
        lo, hi = np.percentile(pooled, [2, 98])
        scale = hi - lo + 1e-12
        an = np.clip((va - lo) / scale, 0, 1)
        bn = np.clip((vb - lo) / scale, 0, 1)
        an[np.isnan(a)] = 0
        bn[np.isnan(b)] = 0
        return an, bn

    def _diff_map(a_raw, b_raw):
        """Compute a signed difference image on a jointly-normalized scale.

        Both images are jointly normalized to the same percentile
        range before differencing, so only true scene differences
        produce nonzero values.

        Returns
        -------
        np.ndarray
            Signed difference in [-1, 1].  Positive → *a* brighter
            (reference dominant), negative → *b* brighter (match
            dominant), zero → matched.
        """
        a, b = _norm_pair(a_raw, b_raw)
        return a - b

    # Diverging red–blue colorscale: red = positive (A dominant),
    # white/gray = zero (match), blue = negative (B dominant).
    _RB_DIVERGE = [
        [0.0, 'rgb(30,80,180)'],    # strong blue
        [0.25, 'rgb(120,150,220)'],  # light blue
        [0.5, 'rgb(200,200,200)'],   # neutral gray
        [0.75, 'rgb(220,130,110)'],  # light red
        [1.0, 'rgb(180,40,30)'],     # strong red
    ]

    # NCC-shifted B
    shifted_b = np.roll(ortho_b, shift_row, axis=0)
    shifted_b = np.roll(shifted_b, shift_col, axis=1)
    if shift_row > 0:
        shifted_b[:shift_row, :] = np.nan
    elif shift_row < 0:
        shifted_b[shift_row:, :] = np.nan
    if shift_col > 0:
        shifted_b[:, :shift_col] = np.nan
    elif shift_col < 0:
        shifted_b[:, shift_col:] = np.nan

    # ── Figure 3: PCA A vs PCA B ─────────────────────────────────
    vp_a = _pclim(np.nan_to_num(pca_a))
    vp_b = _pclim(np.nan_to_num(pca_b))

    fig3 = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"A PCA [{kept_a}] of {N_PATCHES}<br>"
            f"{kept_var_a*100:.1f}% variance",
            f"B PCA [{kept_b}] of {N_PATCHES}<br>"
            f"{kept_var_b*100:.1f}% variance",
        ],
    )
    fig3.add_trace(
        go.Heatmap(z=np.flipud(pca_a), zmin=vp_a[0], zmax=vp_a[1],
                   x0=sg.min_east, dx=sg.pixel_size_east,
                   y0=sg.min_north, dy=sg.pixel_size_north,
                   colorscale='Gray', showscale=True,
                   colorbar=dict(x=0.45, len=0.9)),
        row=1, col=1,
    )
    _cross(fig3, 0.0, 0.0, 1, 1)
    fig3.add_trace(
        go.Heatmap(z=np.flipud(pca_b), zmin=vp_b[0], zmax=vp_b[1],
                   x0=sg.min_east, dx=sg.pixel_size_east,
                   y0=sg.min_north, dy=sg.pixel_size_north,
                   colorscale='Gray', showscale=True,
                   colorbar=dict(x=1.0, len=0.9)),
        row=1, col=2,
    )
    _cross(fig3, 0.0, 0.0, 1, 2)
    for c in (1, 2):
        fig3.update_xaxes(title_text="East (m)", row=1, col=c)
        fig3.update_yaxes(title_text="North (m)", row=1, col=c)
    fig3.update_layout(
        title_text=(f"PCA Reconstructions  |  "
                    f"{N_COMPONENTS}/{N_PATCHES} components"),
        width=1400, height=650,
    )
    fig3.show()

    # ── Figure 4: NCC surface + overlay ───────────────────────────
    zoom = 50
    r0 = max(peak_idx[0] - zoom, 0)
    r1 = min(peak_idx[0] + zoom + 1, ncc.shape[0])
    c0 = max(peak_idx[1] - zoom, 0)
    c1 = min(peak_idx[1] + zoom + 1, ncc.shape[1])
    ncc_crop = ncc[r0:r1, c0:c1]
    row_shifts = np.arange(r0, r1) - center[0]
    col_shifts = np.arange(c0, c1) - center[1]

    diff_pca = _diff_map(pca_a, aligned_b)

    fig4 = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"NCC (peak={peak_val:.4f})<br>"
            f"shift=({shift_row},{shift_col}) px = "
            f"({shift_m_row:.1f},{shift_m_col:.1f}) m",
            f"Difference (red=A only, blue=B only, gray=match)<br>"
            f"RMS={reg_result.residual_rms:.2f} px, "
            f"{reg_result.num_matches} matches, "
            f"{reg_result.inlier_ratio*100:.0f}% inliers",
        ],
    )
    fig4.add_trace(
        go.Heatmap(z=np.flipud(ncc_crop),
                   x0=float(col_shifts[0]), dx=1.0,
                   y0=float(row_shifts[0]), dy=1.0,
                   colorscale='Hot', showscale=True,
                   colorbar=dict(x=0.45, len=0.9)),
        row=1, col=1,
    )
    _cross(fig4, shift_col, shift_row, 1, 1)
    fig4.add_trace(
        go.Heatmap(z=np.flipud(diff_pca), zmin=-1, zmax=1,
                   x0=sg.min_east, dx=sg.pixel_size_east,
                   y0=sg.min_north, dy=sg.pixel_size_north,
                   colorscale=_RB_DIVERGE, showscale=True,
                   colorbar=dict(x=1.0, len=0.9,
                                 title='A−B')),
        row=1, col=2,
    )
    _cross(fig4, 0.0, 0.0, 1, 2)
    fig4.update_xaxes(title_text="Col shift (px)", row=1, col=1)
    fig4.update_yaxes(title_text="Row shift (px)", row=1, col=1)
    fig4.update_xaxes(title_text="East (m)", row=1, col=2)
    fig4.update_yaxes(title_text="North (m)", row=1, col=2)
    fig4.update_layout(
        title_text=(f"NCC + Feature Alignment  |  "
                    f"ORB + Affine"),
        width=1400, height=650,
    )
    fig4.show()

    # ── Figure 5: NCC-shifted A vs B ──────────────────────────────
    vs_b = _pclim(np.nan_to_num(shifted_b))

    fig5 = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"A (reference)<br>{data_a['name']}",
            f"B (NCC-shifted: {shift_row},{shift_col} px)<br>"
            f"{data_b['name']}",
        ],
    )
    fig5.add_trace(
        go.Heatmap(z=np.flipud(ortho_a), zmin=vo_a[0], zmax=vo_a[1],
                   x0=sg.min_east, dx=sg.pixel_size_east,
                   y0=sg.min_north, dy=sg.pixel_size_north,
                   colorscale='Gray', showscale=True,
                   colorbar=dict(x=0.45, len=0.9)),
        row=1, col=1,
    )
    _cross(fig5, 0.0, 0.0, 1, 1)
    fig5.add_trace(
        go.Heatmap(z=np.flipud(shifted_b), zmin=vs_b[0], zmax=vs_b[1],
                   x0=sg.min_east, dx=sg.pixel_size_east,
                   y0=sg.min_north, dy=sg.pixel_size_north,
                   colorscale='Gray', showscale=True,
                   colorbar=dict(x=1.0, len=0.9)),
        row=1, col=2,
    )
    _cross(fig5, 0.0, 0.0, 1, 2)
    for c in (1, 2):
        fig5.update_xaxes(title_text="East (m)", row=1, col=c)
        fig5.update_yaxes(title_text="North (m)", row=1, col=c)
    fig5.update_layout(
        title_text=(f"NCC-Aligned Orthorectified  |  "
                    f"{pixel_size:.2f} m pixels  |  "
                    f"({center_lat:.4f}, {center_lon:.4f})"),
        width=1400, height=650,
    )
    fig5.show()

    # ── Figure 6: NCC-shifted difference ─────────────────────────
    diff_ncc = _diff_map(ortho_a, shifted_b)

    fig6 = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"A (reference)<br>{data_a['name']}",
            f"Difference (red=A only, blue=B only, gray=match)<br>"
            f"NCC peak={peak_val:.4f}, "
            f"shift=({shift_m_row:.1f},{shift_m_col:.1f}) m",
        ],
    )
    fig6.add_trace(
        go.Heatmap(z=np.flipud(ortho_a), zmin=vo_a[0], zmax=vo_a[1],
                   x0=sg.min_east, dx=sg.pixel_size_east,
                   y0=sg.min_north, dy=sg.pixel_size_north,
                   colorscale='Gray', showscale=True,
                   colorbar=dict(x=0.45, len=0.9)),
        row=1, col=1,
    )
    _cross(fig6, 0.0, 0.0, 1, 1)
    fig6.add_trace(
        go.Heatmap(z=np.flipud(diff_ncc), zmin=-1, zmax=1,
                   x0=sg.min_east, dx=sg.pixel_size_east,
                   y0=sg.min_north, dy=sg.pixel_size_north,
                   colorscale=_RB_DIVERGE, showscale=True,
                   colorbar=dict(x=1.0, len=0.9,
                                 title='A−B')),
        row=1, col=2,
    )
    _cross(fig6, 0.0, 0.0, 1, 2)
    for c in (1, 2):
        fig6.update_xaxes(title_text="East (m)", row=1, col=c)
        fig6.update_yaxes(title_text="North (m)", row=1, col=c)
    fig6.update_layout(
        title_text=(f"Coregistration Difference  |  "
                    f"{pixel_size:.2f} m pixels"),
        width=1400, height=650,
    )
    fig6.show()


if __name__ == "__main__":
    main()
