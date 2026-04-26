# -*- coding: utf-8 -*-
"""
Density-based SAR contrast remaps.

Direct ports of the Mangis-derived family from
``sarpy.visualization.remap``:

- :class:`MangisDensity` — ``Density`` (Kevin Mangis 1994).
- :class:`Brighter`, :class:`Darker`, :class:`HighContrast` — the three
  ``Density`` presets shipped by sarpy.
- :class:`GDM` — Generalized Density Mapping (data-driven cutoffs).
- :class:`PEDF` — Piecewise Extended Density Format (Density with top
  half compressed by a factor of 2).

All operate on complex or real amplitude and return ``float32`` in
``[0, 1]``.  Pre-computed ``data_mean`` (and ``data_median`` for GDM)
can be passed at ``apply()`` time for tile-consistent display.

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
2026-04-26

Modified
--------
2026-04-26
"""

# Standard library
from typing import Annotated, Any, Optional

# Third-party
import numpy as np

# GRDL internal
from grdl.image_processing.base import ImageTransform
from grdl.image_processing.params import Desc, Range
from grdl.image_processing.versioning import processor_tags, processor_version
from grdl.vocabulary import ImageModality as IM, ProcessorCategory as PC


# =====================================================================
# Mangis amplitude-to-density helper (shared by Density, GDM, PEDF)
# =====================================================================


def _amplitude_to_density(
    data: np.ndarray,
    dmin: float,
    mmult: float,
    data_mean: Optional[float],
    eps: float = 1e-5,
) -> np.ndarray:
    """Mangis 1994 amplitude → density (output range 0–255).

    Caller divides by 255 to land in ``[0, 1]``.  This factored helper
    lets ``GDM`` and ``PEDF`` reuse the same numerics without going
    through a class instance.
    """
    amplitude = np.abs(data)
    if np.all(amplitude == 0):
        return amplitude.astype(np.float64)

    if not data_mean:
        finite = amplitude[np.isfinite(amplitude)]
        data_mean = float(np.mean(finite)) if finite.size else 0.0

    c_l = 0.8 * data_mean
    c_h = mmult * c_l
    slope = (255.0 - dmin) / np.log10(c_h / c_l)
    constant = dmin - (slope * np.log10(c_l))
    return slope * np.log10(np.maximum(amplitude, eps)) + constant


# =====================================================================
# MangisDensity (sarpy.Density)
# =====================================================================


@processor_version('1.0.0')
@processor_tags(
    modalities=[IM.SAR],
    category=PC.ENHANCE,
    description='Mangis 1994 logarithmic density remap (sarpy Density).',
)
class MangisDensity(ImageTransform):
    """SAR magnitude → log-density via Kevin Mangis' 1994 algorithm.

    Slope and offset are derived so that ``0.8 * data_mean`` maps to
    ``dmin`` and ``mmult * 0.8 * data_mean`` maps to ``255``.  The
    output is divided by 255 so this class returns ``float32`` in
    ``[0, 1]``.

    Parameters
    ----------
    dmin : float, default=30.0
        Dynamic-range floor (0–254).  Lower widens range; higher narrows.
    mmult : float, default=40.0
        Contrast parameter.  Lower → higher contrast and quicker
        saturation; higher → lower contrast and slower saturation.
    eps : float, default=1e-5
        Floor inside ``log10`` to avoid ``-inf`` on zero pixels.
    """

    dmin: Annotated[
        float, Range(min=0.0, max=254.999),
        Desc('Dynamic range floor (0–255)'),
    ] = 30.0
    mmult: Annotated[
        float, Range(min=1.0, max=200.0),
        Desc('Contrast parameter'),
    ] = 40.0
    eps: Annotated[
        float, Range(min=1e-12, max=1.0),
        Desc('Floor offset for log of zero'),
    ] = 1e-5

    def __init__(
        self,
        dmin: float = 30.0,
        mmult: float = 40.0,
        eps: float = 1e-5,
    ) -> None:
        self.dmin = dmin
        self.mmult = mmult
        self.eps = eps

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply density remap.

        Parameters
        ----------
        source : np.ndarray
            Complex or real-valued amplitude data.
        **kwargs
            ``data_mean`` (float, optional) — pre-computed mean amplitude
            for cross-tile consistency.  When omitted the mean is taken
            on the input array.

        Returns
        -------
        np.ndarray
            ``float32`` in ``[0, 1]``.
        """
        params = self._resolve_params(kwargs)
        density = _amplitude_to_density(
            source,
            dmin=params['dmin'],
            mmult=params['mmult'],
            data_mean=kwargs.get('data_mean'),
            eps=params['eps'],
        )
        return np.clip(density / 255.0, 0.0, 1.0).astype(np.float32)


# =====================================================================
# Density presets — Brighter / Darker / HighContrast
# =====================================================================


@processor_version('1.0.0')
@processor_tags(
    modalities=[IM.SAR],
    category=PC.ENHANCE,
    description='MangisDensity preset: brighter (dmin=60, mmult=40).',
)
class Brighter(MangisDensity):
    """Density preset for brighter results — ``dmin=60``, ``mmult=40``."""

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__(dmin=60.0, mmult=40.0, eps=eps)


@processor_version('1.0.0')
@processor_tags(
    modalities=[IM.SAR],
    category=PC.ENHANCE,
    description='MangisDensity preset: darker (dmin=0, mmult=40).',
)
class Darker(MangisDensity):
    """Density preset for darker results — ``dmin=0``, ``mmult=40``."""

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__(dmin=0.0, mmult=40.0, eps=eps)


@processor_version('1.0.0')
@processor_tags(
    modalities=[IM.SAR],
    category=PC.ENHANCE,
    description='MangisDensity preset: high contrast (dmin=30, mmult=4).',
)
class HighContrast(MangisDensity):
    """Density preset for high contrast — ``dmin=30``, ``mmult=4``."""

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__(dmin=30.0, mmult=4.0, eps=eps)


# =====================================================================
# GDM — Generalized Density Mapping (graze/slope-aware)
# =====================================================================


@processor_version('1.0.0')
@processor_tags(
    modalities=[IM.SAR],
    category=PC.ENHANCE,
    description='Generalized Density Mapping with graze/slope-aware cutoffs.',
)
class GDM(ImageTransform):
    """Density remap with image-specific cutoffs (sarpy ``GDM``).

    Computes per-image low/high cutoffs from ``data_mean``,
    ``data_median``, ``graze_deg`` and ``slope_deg``, then applies the
    same Mangis density mapping with those cutoffs (``dmin=30``,
    ``mmult = c_h / c_l``).  Result divided by 255 → ``[0, 1]``.

    Parameters
    ----------
    graze_deg : float
        Required graze angle in degrees.
    slope_deg : float
        Required slope angle in degrees.
    weighting : {'uniform', 'taylor'}, default='uniform'
        Spectral taper weighting.  Picks coefficients in the cutoff
        formula.
    """

    graze_deg: Annotated[
        float, Range(min=0.0, max=90.0),
        Desc('Graze angle in degrees'),
    ] = 30.0
    slope_deg: Annotated[
        float, Range(min=-90.0, max=90.0),
        Desc('Slope angle in degrees'),
    ] = 0.0
    weighting: Annotated[
        str, Desc('Taper weighting (uniform|taylor)'),
    ] = 'uniform'

    def __init__(
        self,
        graze_deg: float,
        slope_deg: float,
        weighting: str = 'uniform',
    ) -> None:
        self.graze_deg = float(graze_deg)
        self.slope_deg = float(slope_deg)
        self.weighting = weighting.lower()

    def _cutoff_values(
        self, data_mean: float, data_median: float,
    ) -> tuple:
        """Replicates the sarpy GDM cutoff math.

        Coefficients copied verbatim from
        ``sarpy.visualization.remap.GDM._cutoff_values``.
        """
        c3 = {'taylor': 1.33, 'uniform': 1.23}.get(self.weighting, 1.33)
        w1 = {'taylor': 0.77, 'uniform': 0.77}.get(self.weighting, 0.77)
        w2 = -0.0422
        w5 = -1.95
        graze_rad = np.radians(self.graze_deg)
        slope_rad = np.radians(self.slope_deg)

        a1 = data_median
        a2 = np.sin(graze_rad) / np.cos(slope_rad)
        a5 = np.log10(data_median / data_mean)
        cl_init = a1 * w1
        ch_init = a1 * 10.0 ** (c3 + w2 * a2 + w5 * a5)
        r_min = 24.0
        r_max = 40.0
        r = 20.0 * np.log10(ch_init / cl_init)
        if r < r_min:
            beta = 10.0 ** ((r - r_min) / 40.0)
        elif r > r_max:
            beta = 10.0 ** ((r - r_max) / 40.0)
        else:
            beta = 1.0
        cl = cl_init * beta
        ch = ch_init / beta
        return cl, ch

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply GDM remap.

        Parameters
        ----------
        source : np.ndarray
            Complex or real-valued data.
        **kwargs
            ``data_mean``, ``data_median`` — pre-computed stats; either
            falls back to the input's own mean/median when not given.
        """
        amplitude = np.abs(source) if np.iscomplexobj(source) else source
        amplitude = np.asarray(amplitude)
        finite = amplitude[np.isfinite(amplitude)]
        if finite.size == 0:
            return np.zeros(amplitude.shape, dtype=np.float32)

        data_mean = kwargs.get('data_mean')
        if data_mean is None:
            data_mean = float(np.mean(finite))
        data_median = kwargs.get('data_median')
        if data_median is None:
            data_median = float(np.median(finite))

        c_l, c_h = self._cutoff_values(data_mean, data_median)
        density = _amplitude_to_density(
            source,
            dmin=30.0,
            mmult=c_h / c_l,
            data_mean=c_l / 0.8,
        )
        return np.clip(density / 255.0, 0.0, 1.0).astype(np.float32)


# =====================================================================
# PEDF — Piecewise Extended Density Format
# =====================================================================


@processor_version('1.0.0')
@processor_tags(
    modalities=[IM.SAR],
    category=PC.ENHANCE,
    description='Piecewise Extended Density Format (sarpy PEDF).',
)
class PEDF(ImageTransform):
    """Density with the top half compressed by a factor of two.

    For ``0 ≤ y ≤ 0.5`` (in normalized [0, 1] scale): identity to Density.
    For ``y > 0.5``: ``y' = 0.5 * (y + 0.5)`` — compresses highlights so
    bright targets don't saturate.

    Parameters mirror :class:`MangisDensity`.
    """

    dmin: Annotated[
        float, Range(min=0.0, max=254.999),
        Desc('Dynamic range floor (0–255)'),
    ] = 30.0
    mmult: Annotated[
        float, Range(min=1.0, max=200.0),
        Desc('Contrast parameter'),
    ] = 40.0
    eps: Annotated[
        float, Range(min=1e-12, max=1.0),
        Desc('Floor offset for log of zero'),
    ] = 1e-5

    def __init__(
        self,
        dmin: float = 30.0,
        mmult: float = 40.0,
        eps: float = 1e-5,
    ) -> None:
        self.dmin = dmin
        self.mmult = mmult
        self.eps = eps

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        params = self._resolve_params(kwargs)
        density = _amplitude_to_density(
            source,
            dmin=params['dmin'],
            mmult=params['mmult'],
            data_mean=kwargs.get('data_mean'),
            eps=params['eps'],
        )
        # Sarpy operates in 0-255 scale: top_mask is out > 127.5.
        # The 0.5 * (out + 127.5) compression in sarpy units becomes
        # 0.5 * (norm + 0.5) in [0, 1] units.
        norm = density / 255.0
        top_mask = norm > 0.5
        norm[top_mask] = 0.5 * (norm[top_mask] + 0.5)
        return np.clip(norm, 0.0, 1.0).astype(np.float32)
