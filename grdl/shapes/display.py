# -*- coding: utf-8 -*-
"""
Display helpers: matplotlib overlay and RGB burn-in for geographic shapes.

Two complementary forms:

- :func:`overlay_shape` / :func:`overlay_shapes` -- attach a
  ``matplotlib.patches.Polygon`` to an existing ``Axes`` showing a SAR
  or EO image. Analytic notebook / interactive use.

- :func:`burn_shape` -- rasterise the shape outline (and optional fill)
  into an RGB ``uint8`` array. Headless / file-output use -- the
  returned array is ready for ``imageio.imwrite`` or embedding in a
  report.

matplotlib is imported lazily so users who only need masks or cueing do
not pay the import cost.

Dependencies
------------
numpy
scikit-image
matplotlib (lazy)

Author
------
Duane Smalley
170194430+DDSmalls@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-04-18

Modified
--------
2026-04-18
"""

# Standard library
from typing import TYPE_CHECKING, Optional, Sequence, Tuple

# Third-party
import numpy as np

# GRDL internal
from grdl.shapes.base import GeographicShape
from grdl.shapes.rasterize import rasterize_polygon


if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.patches
    from grdl.geolocation.base import Geolocation


def overlay_shape(
    ax: 'matplotlib.axes.Axes',
    shape: GeographicShape,
    geolocation: 'Geolocation',
    color: str = 'lime',
    linewidth: float = 1.5,
    fill: bool = False,
    alpha: float = 0.3,
    label: Optional[str] = None,
    n_initial: int = 128,
    pixel_tolerance: float = 0.5,
    height: Optional[float] = None,
) -> 'matplotlib.patches.Polygon':
    """Overlay a shape on an existing matplotlib Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes. Should already be showing an image via
        ``ax.imshow(...)``.
    shape : GeographicShape
        Shape to overlay.
    geolocation : Geolocation
        Geolocation for the image displayed on ``ax``.
    color : str
        Edge colour for the polygon.
    linewidth : float
        Edge width in points.
    fill : bool
        When True, fill the polygon interior with ``color`` at the
        given ``alpha``.
    alpha : float
        Fill opacity when ``fill`` is True. Edge is always drawn at
        full opacity.
    label : str, optional
        Legend label.
    n_initial, pixel_tolerance : see :meth:`GeographicShape.to_pixels`.
    height : float, optional
        Constant HAE (metres) at which to project every perimeter
        vertex. When ``None`` (default) the DEM attached to the
        geolocation is sampled per-vertex. Set a single value to render
        a ground shape cleanly on slant-range imagery over steep
        terrain, where per-vertex DEM variation causes layover-driven
        self-intersection.

    Returns
    -------
    matplotlib.patches.Polygon
        The patch added to the axes. Callers can mutate properties
        (``set_hatch``, ``set_zorder``, ...) as needed.
    """
    import matplotlib.patches as mpatches  # lazy

    pixels = shape.to_pixels(
        geolocation=geolocation,
        n_initial=n_initial,
        pixel_tolerance=pixel_tolerance,
        height=height,
    )
    # matplotlib Polygon expects (x, y) = (col, row)
    xy = np.column_stack([pixels[:, 1], pixels[:, 0]])
    patch = mpatches.Polygon(
        xy,
        closed=shape.is_closed,
        fill=fill and shape.is_closed,
        facecolor=color if fill and shape.is_closed else 'none',
        edgecolor=color,
        linewidth=linewidth,
        alpha=alpha if fill and shape.is_closed else 1.0,
        label=label,
    )
    if fill and shape.is_closed:
        # Edge stays fully opaque; patch-level alpha blends only the fill.
        patch.set_alpha(None)
        patch.set_facecolor((*_color_to_rgb(color), alpha))
        patch.set_edgecolor(color)
    ax.add_patch(patch)
    return patch


def overlay_shapes(
    ax: 'matplotlib.axes.Axes',
    shapes: Sequence[GeographicShape],
    geolocation: 'Geolocation',
    **kwargs,
) -> list:
    """Batch convenience around :func:`overlay_shape`. Returns patch list."""
    return [overlay_shape(ax, s, geolocation, **kwargs) for s in shapes]


def burn_shape(
    image: np.ndarray,
    shape: GeographicShape,
    geolocation: 'Geolocation',
    color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
    fill: bool = False,
    fill_alpha: float = 0.3,
    n_initial: int = 128,
    pixel_tolerance: float = 0.5,
) -> np.ndarray:
    """Rasterise the shape into an RGB copy of ``image``.

    Parameters
    ----------
    image : np.ndarray
        ``(H, W)`` single-band or ``(H, W, 3)`` RGB array. Single-band
        inputs are promoted to RGB via grayscale replication. The
        input is not mutated.
    shape : GeographicShape
        Shape to burn in.
    geolocation : Geolocation
        Geolocation for ``image``.
    color : Tuple[int, int, int]
        RGB colour for the outline (and fill, if enabled).
    thickness : int
        Outline thickness in pixels.
    fill : bool
        When True, fill the interior with ``color`` at ``fill_alpha``.
    fill_alpha : float
        Interior blend ratio in [0, 1].
    n_initial, pixel_tolerance : see :meth:`GeographicShape.to_pixels`.

    Returns
    -------
    np.ndarray
        ``(H, W, 3)`` uint8 RGB array.
    """
    if image.ndim == 2:
        rgb = _promote_gray_to_rgb(image)
    elif image.ndim == 3 and image.shape[2] == 3:
        rgb = _ensure_uint8(image).copy()
    else:
        raise ValueError(
            f"image must be (H, W) or (H, W, 3); got {image.shape}"
        )

    image_shape = rgb.shape[:2]
    pixels = shape.to_pixels(
        geolocation=geolocation,
        n_initial=n_initial,
        pixel_tolerance=pixel_tolerance,
    )

    if fill and shape.is_closed:
        fill_mask = rasterize_polygon(
            pixels=pixels,
            image_shape=image_shape,
            fill=True,
            outline=False,
            outline_thickness=1,
            closed=True,
        )
        rgb = _blend_color_into(rgb, fill_mask, color, fill_alpha)

    outline_mask = rasterize_polygon(
        pixels=pixels,
        image_shape=image_shape,
        fill=False,
        outline=True,
        outline_thickness=max(1, int(thickness)),
        closed=shape.is_closed,
    )
    rgb[outline_mask] = np.asarray(color, dtype=np.uint8)
    return rgb


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _promote_gray_to_rgb(image: np.ndarray) -> np.ndarray:
    gray = _ensure_uint8(image)
    return np.stack([gray, gray, gray], axis=-1)


def _ensure_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image.copy()
    data = np.asarray(image, dtype=np.float64)
    lo = np.nanmin(data)
    hi = np.nanmax(data)
    if hi <= lo:
        return np.zeros(image.shape, dtype=np.uint8)
    scaled = (data - lo) / (hi - lo) * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)


def _blend_color_into(
    rgb: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    a = float(np.clip(alpha, 0.0, 1.0))
    fg = np.asarray(color, dtype=np.float64)
    rgb_f = rgb.astype(np.float64)
    rgb_f[mask] = (1.0 - a) * rgb_f[mask] + a * fg
    return np.clip(rgb_f, 0, 255).astype(np.uint8)


def _color_to_rgb(name: str) -> Tuple[float, float, float]:
    import matplotlib.colors as mcolors  # lazy
    return mcolors.to_rgb(name)
