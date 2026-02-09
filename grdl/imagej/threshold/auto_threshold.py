# -*- coding: utf-8 -*-
"""
Auto Threshold - Port of Fiji's Auto Threshold plugin (global methods).

Implements 16 global (whole-image) automatic thresholding methods.
Each method analyzes the image histogram and returns an optimal
threshold value, plus optionally a binary mask. This complements the
already-ported ``AutoLocalThreshold`` which operates on local
neighborhoods.

Particularly useful for:
- Binary segmentation of SAR backscatter for water/land classification
- Cloud masking in PAN/EO imagery (Otsu, Triangle)
- Ship/target detection thresholding in SAR amplitude images
- Vegetation mask extraction from NDVI (MSI)
- Hot-spot detection in thermal imagery (MaxEntropy, Moments)
- Foreground/background separation in any single-band image

Attribution
-----------
Algorithm references (per method):

- Default (IsoData): Ridler & Calvard, "Picture Thresholding Using an
  Iterative Selection Method", IEEE Trans. SMC 8, 1978, pp. 630-632.
- Huang: Huang & Wang, "A Thresholding Method Based on a Fuzzy
  Compactness Measure", Pattern Recognition Letters 16, 1995.
- IsoData: See Default.
- Li: Li & Lee, "Minimum Cross Entropy Thresholding", Pattern
  Recognition 26(4), 1993, pp. 617-625.
- MaxEntropy: Kapur, Sahoo & Wong, "A New Method for Gray-Level Picture
  Thresholding Using the Entropy of the Histogram", CVGIP 29, 1985.
- Mean: Glasbey, "An Analysis of Histogram-Based Thresholding
  Algorithms", CVGIP: Graphical Models and Image Processing 55, 1993.
- MinError: Kittler & Illingworth, "Minimum Error Thresholding",
  Pattern Recognition 19, 1986, pp. 41-47.
- Minimum: Prewitt & Mendelsohn, "The Analysis of Cell Images",
  Annals of the NY Academy of Sciences 128, 1966, pp. 1035-1053.
- Moments: Tsai, "Moment-Preserving Thresholding: A New Approach",
  CVGIP 29, 1985, pp. 377-393.
- Otsu: Otsu, "A Threshold Selection Method from Gray-Level
  Histograms", IEEE Trans. SMC 9(1), 1979, pp. 62-66.
- Percentile: Doyle, "Operations Useful for Similarity-Invariant
  Pattern Recognition", JACM 9, 1962, pp. 259-267.
- RenyiEntropy: Kapur, Sahoo & Wong, 1985 (see MaxEntropy).
- Shanbhag: Shanbhag, "Utilization of Information Measure as a Means
  of Image Thresholding", CVGIP: Graphical Models and Image
  Processing 56(5), 1994, pp. 414-419.
- Triangle: Zack, Rogers & Latt, "Automatic Measurement of Sister
  Chromatid Exchange Frequency", J. Histochemistry & Cytochemistry
  25(7), 1977, pp. 741-753.
- Yen: Yen, Chang & Chang, "A New Criterion for Automatic Multilevel
  Thresholding", IEEE Trans. Image Processing 4(3), 1995, pp. 370-378.
- Intermodes: Prewitt & Mendelsohn, 1966 (variant of Minimum).

Fiji implementation: Gabriel Landini (University of Birmingham).
Source: ``fiji/threshold/Auto_Threshold.java`` (Fiji, GPL-2).
This is an independent NumPy reimplementation following the published
algorithms, not a derivative of the GPL source.

Author
------
Jason Fritz

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-02-09

Modified
--------
2026-02-09
"""

# Standard library
from typing import Any, Optional, Tuple

# Third-party
import numpy as np

# GRDL internal
from grdl.image_processing.base import ImageTransform
from grdl.image_processing.versioning import processor_version, processor_tags


THRESHOLD_METHODS = (
    'default', 'huang', 'intermodes', 'isodata', 'li', 'maxentropy',
    'mean', 'minerror', 'minimum', 'moments', 'otsu', 'percentile',
    'renyientropy', 'shanbhag', 'triangle', 'yen',
)


def _smooth_histogram(hist: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Smooth histogram by iterative 3-point averaging."""
    h = hist.astype(np.float64).copy()
    for _ in range(iterations):
        h_new = h.copy()
        for i in range(1, len(h) - 1):
            h_new[i] = (h[i - 1] + h[i] + h[i + 1]) / 3.0
        h = h_new
    return h


def _threshold_default(hist: np.ndarray) -> int:
    """IsoData / iterative intermeans method (Ridler & Calvard 1978)."""
    total = hist.sum()
    if total == 0:
        return 0
    n_bins = len(hist)
    # Initial threshold at mean
    s = np.sum(np.arange(n_bins) * hist)
    threshold = int(s / total)

    for _ in range(1000):
        # Mean of pixels below and above threshold
        w_lo = hist[:threshold + 1].sum()
        w_hi = hist[threshold + 1:].sum()
        if w_lo == 0 or w_hi == 0:
            break
        mean_lo = np.sum(np.arange(threshold + 1) * hist[:threshold + 1]) / w_lo
        mean_hi = np.sum(np.arange(threshold + 1, n_bins) * hist[threshold + 1:]) / w_hi
        new_threshold = int((mean_lo + mean_hi) / 2.0)
        if new_threshold == threshold:
            break
        threshold = new_threshold

    return threshold


def _threshold_huang(hist: np.ndarray) -> int:
    """Huang's fuzzy thresholding (Huang & Wang 1995)."""
    n_bins = len(hist)
    total = hist.sum()
    if total == 0:
        return 0

    # First and last non-zero bins
    nz = np.nonzero(hist)[0]
    if len(nz) < 2:
        return nz[0] if len(nz) == 1 else 0
    first = nz[0]
    last = nz[-1]

    # Compute cumulative sums
    s = np.cumsum(hist)
    sm = np.cumsum(hist * np.arange(n_bins))

    # Entropy measure for each candidate threshold
    best_threshold = first
    best_ent = np.inf

    for t in range(first, last + 1):
        w_lo = s[t]
        w_hi = total - w_lo
        if w_lo == 0 or w_hi == 0:
            continue
        mu_lo = sm[t] / w_lo
        mu_hi = (sm[-1] - sm[t]) / w_hi

        # Fuzzy entropy
        ent = 0.0
        for i in range(first, last + 1):
            if hist[i] == 0:
                continue
            mu = mu_lo if i <= t else mu_hi
            dist = abs(i - mu) / (last - first)
            mu_x = 1.0 / (1.0 + dist)
            if 0 < mu_x < 1:
                ent -= hist[i] * (mu_x * np.log(mu_x) +
                                   (1.0 - mu_x) * np.log(1.0 - mu_x))
        if ent < best_ent:
            best_ent = ent
            best_threshold = t

    return best_threshold


def _threshold_intermodes(hist: np.ndarray) -> int:
    """Intermodes method (midpoint between two histogram modes)."""
    h = _smooth_histogram(hist, iterations=50)

    # Find the two largest peaks
    peaks = []
    for i in range(1, len(h) - 1):
        if h[i] > h[i - 1] and h[i] >= h[i + 1]:
            peaks.append((h[i], i))

    if len(peaks) < 2:
        return _threshold_default(hist)

    peaks.sort(reverse=True)
    p1 = peaks[0][1]
    p2 = peaks[1][1]
    return (min(p1, p2) + max(p1, p2)) // 2


def _threshold_li(hist: np.ndarray) -> int:
    """Li's Minimum Cross Entropy method (Li & Lee 1993)."""
    n_bins = len(hist)
    total = hist.sum()
    if total == 0:
        return 0

    sm = np.sum(np.arange(n_bins) * hist)
    mean_all = sm / total

    # Initial threshold
    threshold = int(mean_all)
    new_threshold = threshold

    for _ in range(1000):
        w_lo = hist[:threshold + 1].sum()
        w_hi = hist[threshold + 1:].sum()
        if w_lo == 0 or w_hi == 0:
            break
        mean_lo = np.sum(np.arange(threshold + 1) * hist[:threshold + 1]) / w_lo
        mean_hi = np.sum(np.arange(threshold + 1, n_bins) * hist[threshold + 1:]) / w_hi

        if mean_lo <= 0 or mean_hi <= 0:
            break
        new_threshold = int((mean_lo - mean_hi) /
                           (np.log(mean_lo) - np.log(mean_hi)) + 0.5)
        new_threshold = max(0, min(n_bins - 1, new_threshold))
        if new_threshold == threshold:
            break
        threshold = new_threshold

    return threshold


def _threshold_maxentropy(hist: np.ndarray) -> int:
    """Maximum Entropy method (Kapur, Sahoo & Wong 1985)."""
    n_bins = len(hist)
    total = hist.sum()
    if total == 0:
        return 0

    p = hist.astype(np.float64) / total
    best_threshold = 0
    best_ent = -np.inf

    for t in range(n_bins - 1):
        # Cumulative probability up to t
        p_lo = p[:t + 1].sum()
        p_hi = 1.0 - p_lo
        if p_lo < 1e-15 or p_hi < 1e-15:
            continue

        # Entropy of foreground and background
        h_lo = 0.0
        for i in range(t + 1):
            if p[i] > 0:
                pi = p[i] / p_lo
                h_lo -= pi * np.log(pi)

        h_hi = 0.0
        for i in range(t + 1, n_bins):
            if p[i] > 0:
                pi = p[i] / p_hi
                h_hi -= pi * np.log(pi)

        ent = h_lo + h_hi
        if ent > best_ent:
            best_ent = ent
            best_threshold = t

    return best_threshold


def _threshold_mean(hist: np.ndarray) -> int:
    """Mean threshold (Glasbey 1993): threshold = mean gray level."""
    total = hist.sum()
    if total == 0:
        return 0
    return int(np.sum(np.arange(len(hist)) * hist) / total + 0.5)


def _threshold_minerror(hist: np.ndarray) -> int:
    """Minimum Error method (Kittler & Illingworth 1986)."""
    n_bins = len(hist)
    total = hist.sum()
    if total == 0:
        return 0

    best_threshold = 0
    best_j = np.inf

    for t in range(1, n_bins - 1):
        w_lo = hist[:t + 1].sum()
        w_hi = hist[t + 1:].sum()
        if w_lo == 0 or w_hi == 0:
            continue

        idx_lo = np.arange(t + 1)
        idx_hi = np.arange(t + 1, n_bins)
        mean_lo = np.sum(idx_lo * hist[:t + 1]) / w_lo
        mean_hi = np.sum(idx_hi * hist[t + 1:]) / w_hi
        var_lo = np.sum(hist[:t + 1] * (idx_lo - mean_lo) ** 2) / w_lo
        var_hi = np.sum(hist[t + 1:] * (idx_hi - mean_hi) ** 2) / w_hi

        if var_lo <= 0 or var_hi <= 0:
            continue

        p_lo = w_lo / total
        p_hi = w_hi / total
        j = (p_lo * np.log(var_lo) + p_hi * np.log(var_hi)) / 2.0 - \
            (p_lo * np.log(p_lo) + p_hi * np.log(p_hi))

        if j < best_j:
            best_j = j
            best_threshold = t

    return best_threshold


def _threshold_minimum(hist: np.ndarray) -> int:
    """Minimum method (Prewitt & Mendelsohn 1966): valley between peaks."""
    h = _smooth_histogram(hist, iterations=50)

    # Find valleys (local minima)
    for i in range(1, len(h) - 1):
        if h[i] < h[i - 1] and h[i] <= h[i + 1]:
            return i

    return _threshold_default(hist)


def _threshold_moments(hist: np.ndarray) -> int:
    """Moments method (Tsai 1985): moment-preserving thresholding."""
    n_bins = len(hist)
    total = hist.sum()
    if total == 0:
        return 0

    p = hist.astype(np.float64) / total
    idx = np.arange(n_bins, dtype=np.float64)

    # Normalized moments
    m0 = 1.0
    m1 = np.sum(idx * p)
    m2 = np.sum(idx ** 2 * p)
    m3 = np.sum(idx ** 3 * p)

    # Solve for threshold using moment-preserving approach
    cd = m0 * m2 - m1 * m1
    if abs(cd) < 1e-15:
        return int(m1)

    c0 = (m1 * m3 - m2 * m2) / cd
    c1 = (m0 * m3 - m1 * m2) / cd  # noqa: F841 (used in discriminant)

    z0 = 0.5 * (-c1 - np.sqrt(max(0.0, c1 * c1 - 4.0 * c0)))

    # Find the threshold closest to z0
    threshold = max(0, min(n_bins - 1, int(z0 + 0.5)))
    return threshold


def _threshold_otsu(hist: np.ndarray) -> int:
    """Otsu's method (Otsu 1979): maximize between-class variance."""
    n_bins = len(hist)
    total = hist.sum()
    if total == 0:
        return 0

    sum_all = np.sum(np.arange(n_bins) * hist)

    sum_bg = 0.0
    w_bg = 0.0
    best_threshold = 0
    best_var = 0.0

    for t in range(n_bins):
        w_bg += hist[t]
        if w_bg == 0:
            continue
        w_fg = total - w_bg
        if w_fg == 0:
            break

        sum_bg += t * hist[t]
        mean_bg = sum_bg / w_bg
        mean_fg = (sum_all - sum_bg) / w_fg

        var_between = w_bg * w_fg * (mean_bg - mean_fg) ** 2
        if var_between > best_var:
            best_var = var_between
            best_threshold = t

    return best_threshold


def _threshold_percentile(hist: np.ndarray) -> int:
    """Percentile method (Doyle 1962): 50th percentile."""
    total = hist.sum()
    if total == 0:
        return 0

    target = total * 0.5
    cumsum = 0.0
    for i in range(len(hist)):
        cumsum += hist[i]
        if cumsum >= target:
            return i
    return len(hist) - 1


def _threshold_renyientropy(hist: np.ndarray) -> int:
    """Renyi Entropy method (Kapur, Sahoo & Wong 1985, generalized)."""
    n_bins = len(hist)
    total = hist.sum()
    if total == 0:
        return 0

    p = hist.astype(np.float64) / total
    best_threshold = 0
    best_ent = -np.inf

    for t in range(1, n_bins - 1):
        p_lo = p[:t + 1].sum()
        p_hi = 1.0 - p_lo
        if p_lo < 1e-15 or p_hi < 1e-15:
            continue

        # Renyi entropy of order 2
        h_lo = -np.log(np.sum((p[:t + 1] / p_lo) ** 2) + 1e-15)
        h_hi = -np.log(np.sum((p[t + 1:] / p_hi) ** 2) + 1e-15)

        ent = h_lo + h_hi
        if ent > best_ent:
            best_ent = ent
            best_threshold = t

    return best_threshold


def _threshold_shanbhag(hist: np.ndarray) -> int:
    """Shanbhag's method (Shanbhag 1994): fuzzy entropy."""
    n_bins = len(hist)
    total = hist.sum()
    if total == 0:
        return 0

    p = hist.astype(np.float64) / total
    cum = np.cumsum(p)

    best_threshold = 0
    best_ent = np.inf

    for t in range(1, n_bins - 1):
        p_lo = cum[t]
        p_hi = 1.0 - p_lo
        if p_lo < 1e-15 or p_hi < 1e-15:
            continue

        # Shanbhag's fuzzy entropy
        ent = 0.0
        for i in range(t + 1):
            if p[i] > 0:
                term = p[i] / p_lo
                term = min(term, 1.0)
                if 0 < term < 1:
                    ent -= term * np.log(term) + (1.0 - term) * np.log(1.0 - term)

        for i in range(t + 1, n_bins):
            if p[i] > 0:
                term = p[i] / p_hi
                term = min(term, 1.0)
                if 0 < term < 1:
                    ent -= term * np.log(term) + (1.0 - term) * np.log(1.0 - term)

        if ent < best_ent:
            best_ent = ent
            best_threshold = t

    return best_threshold


def _threshold_triangle(hist: np.ndarray) -> int:
    """Triangle method (Zack, Rogers & Latt 1977)."""
    n_bins = len(hist)
    nz = np.nonzero(hist)[0]
    if len(nz) < 2:
        return nz[0] if len(nz) == 1 else 0

    first = nz[0]
    last = nz[-1]

    peak_idx = first + np.argmax(hist[first:last + 1])

    # Determine which side is longer (line from peak to farthest end)
    if (peak_idx - first) > (last - peak_idx):
        # Peak is closer to the right, draw line from left to peak
        x1, y1 = first, hist[first]
        x2, y2 = peak_idx, hist[peak_idx]
        search_range = range(first, peak_idx)
        flip = False
    else:
        # Peak is closer to the left, draw line from peak to right
        x1, y1 = peak_idx, hist[peak_idx]
        x2, y2 = last, hist[last]
        search_range = range(peak_idx + 1, last + 1)
        flip = True

    # Distance from each histogram point to the line
    dx = float(x2 - x1)
    dy = float(y2 - y1)
    line_len = np.sqrt(dx * dx + dy * dy)
    if line_len < 1e-15:
        return peak_idx

    best_dist = 0.0
    best_t = x1

    for t in search_range:
        dist = abs(dy * t - dx * hist[t] + x2 * y1 - y2 * x1) / line_len
        if dist > best_dist:
            best_dist = dist
            best_t = t

    return best_t


def _threshold_yen(hist: np.ndarray) -> int:
    """Yen's method (Yen, Chang & Chang 1995)."""
    n_bins = len(hist)
    total = hist.sum()
    if total == 0:
        return 0

    p = hist.astype(np.float64) / total
    p_sq = p * p

    cum_p = np.cumsum(p)
    cum_p_sq = np.cumsum(p_sq)

    best_threshold = 0
    best_crit = -np.inf

    for t in range(n_bins - 1):
        s_lo = cum_p_sq[t]
        s_hi = cum_p_sq[-1] - cum_p_sq[t]
        p_lo = cum_p[t]
        p_hi = 1.0 - p_lo

        if p_lo < 1e-15 or p_hi < 1e-15:
            continue
        if s_lo < 1e-30 or s_hi < 1e-30:
            continue

        crit = -np.log(s_lo) - np.log(s_hi) + \
               2.0 * np.log(p_lo) + 2.0 * np.log(p_hi)
        if crit > best_crit:
            best_crit = crit
            best_threshold = t

    return best_threshold


_METHOD_DISPATCH = {
    'default': _threshold_default,
    'huang': _threshold_huang,
    'intermodes': _threshold_intermodes,
    'isodata': _threshold_default,
    'li': _threshold_li,
    'maxentropy': _threshold_maxentropy,
    'mean': _threshold_mean,
    'minerror': _threshold_minerror,
    'minimum': _threshold_minimum,
    'moments': _threshold_moments,
    'otsu': _threshold_otsu,
    'percentile': _threshold_percentile,
    'renyientropy': _threshold_renyientropy,
    'shanbhag': _threshold_shanbhag,
    'triangle': _threshold_triangle,
    'yen': _threshold_yen,
}


@processor_tags(modalities=['SAR', 'PAN', 'EO', 'MSI', 'HSI', 'thermal'],
                category='threshold')
@processor_version('1.54j')
class AutoThreshold(ImageTransform):
    """Global automatic thresholding with 16 methods, ported from Fiji.

    Computes a single global threshold from the image histogram and
    returns a binary mask. The threshold value is also accessible as
    an attribute after calling ``apply()``.

    Parameters
    ----------
    method : str
        Thresholding algorithm. One of: ``'default'`` (IsoData),
        ``'huang'``, ``'intermodes'``, ``'isodata'``, ``'li'``,
        ``'maxentropy'``, ``'mean'``, ``'minerror'``, ``'minimum'``,
        ``'moments'``, ``'otsu'``, ``'percentile'``,
        ``'renyientropy'``, ``'shanbhag'``, ``'triangle'``, ``'yen'``.
        Default is ``'otsu'``.
    n_bins : int
        Number of histogram bins. Default is 256.
    dark_background : bool
        If True (default), foreground pixels are brighter than
        background (mask = pixels > threshold). If False, foreground
        is darker (mask = pixels < threshold).

    Attributes
    ----------
    threshold_ : float
        The computed threshold value (set after ``apply()`` is called).
        In the original image's value range.
    threshold_bin_ : int
        The threshold bin index (0 to n_bins-1).

    Notes
    -----
    Independent reimplementation of ``fiji/threshold/Auto_Threshold.java``
    by Gabriel Landini (Fiji, GPL-2). Each method follows the published
    algorithm cited in the module docstring.

    Examples
    --------
    Otsu thresholding for water/land segmentation:

    >>> from grdl.imagej import AutoThreshold
    >>> ot = AutoThreshold(method='otsu')
    >>> mask = ot.apply(sar_db)
    >>> print(ot.threshold_)

    Triangle method for cloud detection:

    >>> tri = AutoThreshold(method='triangle', dark_background=False)
    >>> cloud_mask = tri.apply(pan_reflectance)
    """

    __imagej_source__ = 'fiji/threshold/Auto_Threshold.java'
    __imagej_version__ = '1.54j'
    __gpu_compatible__ = True

    def __init__(
        self,
        method: str = 'otsu',
        n_bins: int = 256,
        dark_background: bool = True,
    ) -> None:
        method_lower = method.lower()
        if method_lower not in THRESHOLD_METHODS:
            raise ValueError(
                f"Unknown method '{method}'. "
                f"Must be one of {THRESHOLD_METHODS}"
            )
        if n_bins < 2:
            raise ValueError(f"n_bins must be >= 2, got {n_bins}")

        self.method = method_lower
        self.n_bins = n_bins
        self.dark_background = dark_background
        self.threshold_: Optional[float] = None
        self.threshold_bin_: Optional[int] = None

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply global auto thresholding to a 2D image.

        Parameters
        ----------
        source : np.ndarray
            2D image array. Shape ``(rows, cols)``.

        Returns
        -------
        np.ndarray
            Binary mask, dtype float64, values 0.0 or 1.0.
            Same shape as input.

        Raises
        ------
        ValueError
            If source is not 2D.
        """
        if source.ndim != 2:
            raise ValueError(
                f"Expected 2D image, got shape {source.shape}"
            )

        image = source.astype(np.float64)
        vmin = image.min()
        vmax = image.max()

        if vmax - vmin < 1e-15:
            self.threshold_ = vmin
            self.threshold_bin_ = 0
            return np.zeros_like(image, dtype=np.float64)

        # Build histogram
        hist, bin_edges = np.histogram(image, bins=self.n_bins,
                                       range=(vmin, vmax))

        # Compute threshold bin
        func = _METHOD_DISPATCH[self.method]
        t_bin = func(hist.astype(np.float64))
        t_bin = max(0, min(self.n_bins - 1, t_bin))

        # Convert bin index to image value
        bin_width = (vmax - vmin) / self.n_bins
        threshold_value = vmin + (t_bin + 0.5) * bin_width

        self.threshold_bin_ = t_bin
        self.threshold_ = threshold_value

        if self.dark_background:
            return (image > threshold_value).astype(np.float64)
        else:
            return (image < threshold_value).astype(np.float64)
