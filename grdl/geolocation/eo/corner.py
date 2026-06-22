# -*- coding: utf-8 -*-
"""
Corner Geolocation - Approximate corner-coordinate projection for EO NITF.

Provides ``CornerGeolocation``, a fallback ``Geolocation`` for NITF imagery
that carries no RPC or RSM sensor model.  Fits a projective homography
between the four pixel corners and the four geographic corners found in
(priority order) the CSCRNA TRE, the BLOCKA TRE, or the NITF IGEOLO
header field (all ICORDS forms: G/C geographic, D decimal degrees,
N/S UTM, U MGRS).

This is an APPROXIMATE model: it has no terrain awareness and its
accuracy is limited by the precision of the corner coordinates
themselves (IGEOLO 'G' is whole arc-seconds, roughly 30 m).  Use it
only when no rigorous sensor model (RPC/RSM) is available.

Dependencies
------------
pyproj (only for ICORDS 'N'/'S' UTM and 'U' MGRS corner forms)

Author
------
Duane Smalley, PhD
170194430+DDSmalls@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-06-09

Modified
--------
2026-06-09
"""

# Standard library
import re
from typing import Optional, Tuple, Union, TYPE_CHECKING

# Third-party
import numpy as np

# GRDL internal
from grdl.exceptions import DependencyError
from grdl.geolocation.base import Geolocation

try:
    import pyproj
    _HAS_PYPROJ = True
except ImportError:
    _HAS_PYPROJ = False

if TYPE_CHECKING:
    from grdl.IO.base import ImageReader


# ---------------------------------------------------------------------------
# Corner-string parsing helpers
# ---------------------------------------------------------------------------

# BLOCKA 21-char corner: ddmmss.ssXdddmmss.ssY
_BLOCKA_DMS_RE = re.compile(
    r'^(\d{2})(\d{2})(\d{2}\.\d{2})([NS])(\d{3})(\d{2})(\d{2}\.\d{2})([EW])$'
)
# BLOCKA 21-char corner: ±dd.dddddd±ddd.dddddd
_BLOCKA_DEC_RE = re.compile(
    r'^([+-]\d{2}\.\d{6})([+-]\d{3}\.\d{6})$'
)

# IGEOLO 15-char corner forms
_IGEOLO_DMS_RE = re.compile(
    r'^(\d{2})(\d{2})(\d{2})([NS])(\d{3})(\d{2})(\d{2})([EW])$'
)
_IGEOLO_DEC_RE = re.compile(
    r'^([+-]\d{2}\.\d{3})([+-]\d{3}\.\d{3})$'
)
_IGEOLO_UTM_RE = re.compile(
    r'^(\d{2})(\d{6})(\d{7})$'
)
_IGEOLO_MGRS_RE = re.compile(
    r'^(\d{2})([C-HJ-NP-X])([A-HJ-NP-Z])([A-HJ-NP-V])(\d{5})(\d{5})$'
)

# MGRS 100 km square lettering (I and O are excluded throughout)
_MGRS_COL_SETS = ('ABCDEFGH', 'JKLMNPQR', 'STUVWXYZ')
_MGRS_ROW_LETTERS = 'ABCDEFGHJKLMNPQRSTUV'

# Minimum UTM northing (meters) for each MGRS latitude band, used to
# resolve the 2,000 km row-letter repeat cycle.  Southern-hemisphere
# bands (C-M) are in the 10,000 km false-northing frame; northern
# bands (N-X) start at the equator.  Values follow GEOTRANS / mgrs.js.
_MGRS_MIN_NORTHING = {
    'C': 1100000.0, 'D': 2000000.0, 'E': 2800000.0, 'F': 3700000.0,
    'G': 4600000.0, 'H': 5500000.0, 'J': 6400000.0, 'K': 7300000.0,
    'L': 8200000.0, 'M': 9100000.0,
    'N': 0.0, 'P': 800000.0, 'Q': 1700000.0, 'R': 2600000.0,
    'S': 3500000.0, 'T': 4400000.0, 'U': 5300000.0, 'V': 6200000.0,
    'W': 7000000.0, 'X': 7900000.0,
}


def _require_pyproj(context: str) -> None:
    """Raise a clear error if pyproj is not installed.

    Parameters
    ----------
    context : str
        Description of the operation that needs pyproj, used in the
        error message.

    Raises
    ------
    DependencyError
        If pyproj is not installed.
    """
    if not _HAS_PYPROJ:
        raise DependencyError(
            f"{context} requires pyproj. "
            f"Install with: pip install pyproj"
        )


def _utm_to_latlon(
    zone: int, north: bool, easting: float, northing: float,
) -> Tuple[float, float]:
    """Convert UTM coordinates to WGS84 geodetic latitude/longitude.

    Parameters
    ----------
    zone : int
        UTM zone number (1-60).
    north : bool
        True for the northern hemisphere (EPSG 326zz), False for the
        southern hemisphere (EPSG 327zz).
    easting : float
        UTM easting in meters.
    northing : float
        UTM northing in meters.

    Returns
    -------
    Tuple[float, float]
        (latitude, longitude) in degrees.

    Raises
    ------
    ValueError
        If the zone number is out of range.
    DependencyError
        If pyproj is not installed.
    """
    if not 1 <= zone <= 60:
        raise ValueError(f"UTM zone must be in 1..60, got {zone}")
    _require_pyproj('UTM corner conversion')
    epsg = (32600 if north else 32700) + zone
    transformer = pyproj.Transformer.from_crs(
        f'EPSG:{epsg}', 'EPSG:4326', always_xy=True
    )
    lon, lat = transformer.transform(easting, northing)
    return float(lat), float(lon)


def _mgrs_to_latlon(mgrs: str) -> Tuple[float, float]:
    """Convert a 15-character MGRS reference to WGS84 latitude/longitude.

    Decodes the grid-zone designator (zone number + latitude band),
    the 100 km square letters (zone-dependent column sets, even-zone
    row offset, 2,000 km row repeat resolved per latitude band), and
    the 1-meter-precision numeric easting/northing, then converts the
    resulting UTM coordinates to geodetic via pyproj.

    Parameters
    ----------
    mgrs : str
        15-character MGRS string ``zzBJKeeeeennnnn`` (zero-padded zone,
        band letter, two 100 km square letters, 5-digit easting,
        5-digit northing).

    Returns
    -------
    Tuple[float, float]
        (latitude, longitude) in degrees.

    Raises
    ------
    ValueError
        If the MGRS string is malformed or uses an illegal letter for
        its zone.
    DependencyError
        If pyproj is not installed.
    """
    m = _IGEOLO_MGRS_RE.match(mgrs)
    if m is None:
        raise ValueError(
            f"Invalid 15-character MGRS corner: {mgrs!r}. Expected "
            f"zzBJKeeeeennnnn (zone, band, 100 km square, easting, "
            f"northing)."
        )
    zone = int(m.group(1))
    if not 1 <= zone <= 60:
        raise ValueError(f"MGRS zone must be in 1..60, got {zone}")
    band = m.group(2)
    col_letter = m.group(3)
    row_letter = m.group(4)
    e_digits = int(m.group(5))
    n_digits = int(m.group(6))

    # 100 km column: zone-dependent letter set, easting 100-800 km
    col_set = _MGRS_COL_SETS[(zone - 1) % 3]
    if col_letter not in col_set:
        raise ValueError(
            f"MGRS column letter {col_letter!r} is not valid for "
            f"zone {zone} (legal set: {col_set})"
        )
    easting_100k = (col_set.index(col_letter) + 1) * 100000.0

    # 100 km row: 20-letter cycle, even zones offset by 5 letters
    row_idx = _MGRS_ROW_LETTERS.index(row_letter)
    if zone % 2 == 0:
        row_idx = (row_idx - 5) % 20
    northing_100k = row_idx * 100000.0

    # Resolve the 2,000 km row repeat using the band's minimum northing
    min_northing = _MGRS_MIN_NORTHING[band]
    while northing_100k < min_northing:
        northing_100k += 2000000.0

    easting = easting_100k + e_digits
    northing = northing_100k + n_digits
    north = band >= 'N'
    return _utm_to_latlon(zone, north, easting, northing)


def _parse_blocka_corner(corner: str) -> Tuple[float, float]:
    """Parse a 21-character BLOCKA TRE corner location string.

    Accepts both legal formats:

    - ``ddmmss.ssXdddmmss.ssY`` (X in {N, S}, Y in {E, W})
    - ``±dd.dddddd±ddd.dddddd`` (signed decimal degrees)

    Parameters
    ----------
    corner : str
        21-character corner location string.

    Returns
    -------
    Tuple[float, float]
        (latitude, longitude) in degrees.

    Raises
    ------
    ValueError
        If the string matches neither legal format.
    """
    s = corner.strip()
    m = _BLOCKA_DMS_RE.match(s)
    if m is not None:
        lat = int(m.group(1)) + int(m.group(2)) / 60.0 \
            + float(m.group(3)) / 3600.0
        if m.group(4) == 'S':
            lat = -lat
        lon = int(m.group(5)) + int(m.group(6)) / 60.0 \
            + float(m.group(7)) / 3600.0
        if m.group(8) == 'W':
            lon = -lon
        return lat, lon
    m = _BLOCKA_DEC_RE.match(s)
    if m is not None:
        return float(m.group(1)), float(m.group(2))
    raise ValueError(
        f"Invalid BLOCKA corner location: {corner!r}. Expected "
        f"'ddmmss.ssXdddmmss.ssY' or '±dd.dddddd±ddd.dddddd'."
    )


def _parse_igeolo_corner(corner: str, icords: str) -> Tuple[float, float]:
    """Parse one 15-character IGEOLO corner per the ICORDS form.

    Parameters
    ----------
    corner : str
        15-character corner substring of the IGEOLO field.
    icords : str
        ICORDS code: ``'G'``/``'C'`` (geographic ddmmssXdddmmssY),
        ``'D'`` (decimal ±dd.ddd±ddd.ddd), ``'N'``/``'S'`` (UTM
        zzeeeeeennnnnnn, northern/southern hemisphere), or ``'U'``
        (15-character MGRS).

    Returns
    -------
    Tuple[float, float]
        (latitude, longitude) in degrees.

    Raises
    ------
    ValueError
        If the corner string does not match the ICORDS form, or the
        ICORDS code is unsupported.
    DependencyError
        If pyproj is required (UTM/MGRS) but not installed.
    """
    if icords in ('G', 'C'):
        m = _IGEOLO_DMS_RE.match(corner)
        if m is None:
            raise ValueError(
                f"Invalid IGEOLO geographic corner: {corner!r}. "
                f"Expected ddmmssXdddmmssY."
            )
        lat = int(m.group(1)) + int(m.group(2)) / 60.0 \
            + int(m.group(3)) / 3600.0
        if m.group(4) == 'S':
            lat = -lat
        lon = int(m.group(5)) + int(m.group(6)) / 60.0 \
            + int(m.group(7)) / 3600.0
        if m.group(8) == 'W':
            lon = -lon
        return lat, lon

    if icords == 'D':
        m = _IGEOLO_DEC_RE.match(corner)
        if m is None:
            raise ValueError(
                f"Invalid IGEOLO decimal-degrees corner: {corner!r}. "
                f"Expected ±dd.ddd±ddd.ddd."
            )
        return float(m.group(1)), float(m.group(2))

    if icords in ('N', 'S'):
        m = _IGEOLO_UTM_RE.match(corner)
        if m is None:
            raise ValueError(
                f"Invalid IGEOLO UTM corner: {corner!r}. "
                f"Expected zzeeeeeennnnnnn."
            )
        zone = int(m.group(1))
        easting = float(m.group(2))
        northing = float(m.group(3))
        return _utm_to_latlon(zone, icords == 'N', easting, northing)

    if icords == 'U':
        return _mgrs_to_latlon(corner)

    raise ValueError(
        f"Unsupported ICORDS code: {icords!r}. Supported codes are "
        f"'G', 'C', 'D', 'N', 'S', and 'U'."
    )


def _parse_igeolo(igeolo: str, icords: str) -> np.ndarray:
    """Parse the 60-character NITF IGEOLO field into four corners.

    Parameters
    ----------
    igeolo : str
        60-character IGEOLO field: four 15-character corners in order
        UL, UR, LR, LL.
    icords : str
        ICORDS coordinate representation code.

    Returns
    -------
    np.ndarray
        Corner coordinates, shape ``(4, 2)``, columns ``[lat, lon]``,
        order UL, UR, LR, LL.

    Raises
    ------
    ValueError
        If the field length is wrong or any corner fails to parse.
    """
    s = igeolo.strip()
    if len(s) != 60:
        raise ValueError(
            f"IGEOLO must be 60 characters (four 15-character corners), "
            f"got {len(s)}"
        )
    code = icords.strip().upper()
    if not code:
        raise ValueError("ICORDS is blank — IGEOLO has no defined form")
    corners = [
        _parse_igeolo_corner(s[i * 15:(i + 1) * 15], code)
        for i in range(4)
    ]
    return np.asarray(corners, dtype=np.float64)


# ---------------------------------------------------------------------------
# CornerGeolocation
# ---------------------------------------------------------------------------


class CornerGeolocation(Geolocation):
    """Approximate geolocation from four geographic corner coordinates.

    Fits a projective homography (direct linear transform) between the
    four pixel corner centers and the four geographic corners, giving a
    smooth bidirectional pixel <-> lat/lon mapping across the image.

    .. warning::

        This is an **approximate** model intended as a last-resort
        fallback when no rigorous sensor model (RPC/RSM) is available.
        It has **no terrain awareness** — the homography is a flat
        plane fit — and its accuracy is bounded by the precision of
        the corner coordinates themselves (e.g., IGEOLO ``'G'`` corners
        are whole arc-seconds, roughly 30 m).  Do not use it for
        precision mensuration.

    Parameters
    ----------
    corners : np.ndarray
        Geographic corner coordinates, shape ``(4, 2)``, columns
        ``[lat, lon]`` in degrees, order UL, UR, LR, LL.
    shape : Tuple[int, int]
        Image shape ``(rows, cols)``.
    height : float, default=0.0
        Constant height above the WGS84 ellipsoid (meters) reported
        for all projected points.
    accuracy_source : str, default='corners'
        Provenance of the corner coordinates: ``'CSCRNA'``,
        ``'BLOCKA'``, ``'IGEOLO'``, or ``'corners'`` (direct
        construction).
    dem_path : str or Path, optional
        Path to DEM/DTED data for the base-class elevation hooks.
        Note the homography itself remains terrain-unaware.
    geoid_path : str or Path, optional
        Path to geoid correction file (EGM96/EGM2008).
    interpolation : int, default=3
        Spline interpolation order for DEM sampling.

    Attributes
    ----------
    corners : np.ndarray
        The ``(4, 2)`` corner array used for the fit.
    accuracy_source : str
        Provenance of the corner coordinates.

    Raises
    ------
    ValueError
        If *corners* is not a finite ``(4, 2)`` array, *shape* is
        smaller than 2 x 2 pixels, or the corners are degenerate
        (collinear/coincident, making the homography singular).

    Examples
    --------
    Direct construction:

    >>> import numpy as np
    >>> corners = np.array([[39.0, -77.1],   # UL
    ...                     [39.0, -77.0],   # UR
    ...                     [38.9, -77.0],   # LR
    ...                     [38.9, -77.1]])  # LL
    >>> geo = CornerGeolocation(corners, shape=(1024, 2048))
    >>> lat, lon, h = geo.image_to_latlon(0, 0)

    From an EO NITF reader with no RPC/RSM:

    >>> geo = CornerGeolocation.from_reader(reader)
    >>> geo.accuracy_source
    'IGEOLO'
    """

    def __init__(
        self,
        corners: np.ndarray,
        shape: Tuple[int, int],
        height: float = 0.0,
        accuracy_source: str = 'corners',
        dem_path: Optional[Union[str, object]] = None,
        geoid_path: Optional[Union[str, object]] = None,
        interpolation: int = 3,
    ) -> None:
        corners = np.asarray(corners, dtype=np.float64)
        if corners.shape != (4, 2):
            raise ValueError(
                f"corners must have shape (4, 2) [[lat, lon] x "
                f"UL, UR, LR, LL], got {corners.shape}"
            )
        if not np.all(np.isfinite(corners)):
            raise ValueError("corners contain non-finite values")

        rows, cols = int(shape[0]), int(shape[1])
        if rows < 2 or cols < 2:
            raise ValueError(
                f"shape must be at least (2, 2) to anchor four distinct "
                f"corner pixels, got ({rows}, {cols})"
            )

        self.corners = corners
        self.accuracy_source = accuracy_source
        self._height = float(height)

        # Pixel corner CENTERS in (row, col), order UL, UR, LR, LL
        src = np.array([
            [0.0, 0.0],
            [0.0, cols - 1.0],
            [rows - 1.0, cols - 1.0],
            [rows - 1.0, 0.0],
        ])

        # DLT: solve the 8x8 system for the homography
        #   lat = (h0*r + h1*c + h2) / (h6*r + h7*c + 1)
        #   lon = (h3*r + h4*c + h5) / (h6*r + h7*c + 1)
        r = src[:, 0]
        c = src[:, 1]
        lat = corners[:, 0]
        lon = corners[:, 1]
        a_mat = np.zeros((8, 8))
        a_mat[0::2, 0] = r
        a_mat[0::2, 1] = c
        a_mat[0::2, 2] = 1.0
        a_mat[0::2, 6] = -lat * r
        a_mat[0::2, 7] = -lat * c
        a_mat[1::2, 3] = r
        a_mat[1::2, 4] = c
        a_mat[1::2, 5] = 1.0
        a_mat[1::2, 6] = -lon * r
        a_mat[1::2, 7] = -lon * c
        b_vec = np.empty(8)
        b_vec[0::2] = lat
        b_vec[1::2] = lon

        try:
            h = np.linalg.solve(a_mat, b_vec)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "Degenerate corner coordinates: cannot fit a projective "
                "homography (corners are collinear or coincident)"
            ) from exc

        self._H = np.array([
            [h[0], h[1], h[2]],
            [h[3], h[4], h[5]],
            [h[6], h[7], 1.0],
        ])
        try:
            self._H_inv = np.linalg.inv(self._H)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "Degenerate corner coordinates: homography is not "
                "invertible"
            ) from exc

        super().__init__(
            (rows, cols), crs='WGS84', dem_path=dem_path,
            geoid_path=geoid_path, interpolation=interpolation,
        )

    @property
    def default_hae(self) -> float:
        """Default height above ellipsoid (meters) for this imagery.

        Returns
        -------
        float
            The constant ``height`` provided at construction.
        """
        return self._height

    def _image_to_latlon_array(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
        height: Union[float, np.ndarray] = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Transform pixel coordinate arrays to WGS84 coordinates.

        Applies the fitted homography. Fully vectorized.

        Parameters
        ----------
        rows : np.ndarray
            Row coordinates (1D array, float64).
        cols : np.ndarray
            Column coordinates (1D array, float64).
        height : float or np.ndarray, default=0.0
            Height above WGS84 ellipsoid in meters.  Scalar zero falls
            back to the construction-time height.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            ``(lats, lons, heights)`` arrays in WGS84 coordinates.
        """
        h_mat = self._H
        w = h_mat[2, 0] * rows + h_mat[2, 1] * cols + h_mat[2, 2]
        lats = (h_mat[0, 0] * rows + h_mat[0, 1] * cols + h_mat[0, 2]) / w
        lons = (h_mat[1, 0] * rows + h_mat[1, 1] * cols + h_mat[1, 2]) / w

        if np.ndim(height) > 0:
            heights = np.asarray(height, dtype=np.float64)
        else:
            heights = np.full_like(lats, self._resolve_height(height))
        return lats, lons, heights

    def _latlon_to_image_array(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        height: Union[float, np.ndarray] = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transform WGS84 coordinate arrays to pixel coordinates.

        Applies the analytic inverse homography. Fully vectorized.

        Parameters
        ----------
        lats : np.ndarray
            Latitudes in degrees North (1D array, float64).
        lons : np.ndarray
            Longitudes in degrees East (1D array, float64).
        height : float or np.ndarray, default=0.0
            Height above WGS84 ellipsoid in meters (unused by the flat
            homography inverse, included for ABC compatibility).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            ``(rows, cols)`` pixel coordinate arrays.
        """
        g = self._H_inv
        w = g[2, 0] * lats + g[2, 1] * lons + g[2, 2]
        rows = (g[0, 0] * lats + g[0, 1] * lons + g[0, 2]) / w
        cols = (g[1, 0] * lats + g[1, 1] * lons + g[1, 2]) / w
        return rows, cols

    @classmethod
    def from_reader(
        cls,
        reader: 'ImageReader',
        height: float = 0.0,
        dem_path: Optional[Union[str, object]] = None,
        geoid_path: Optional[Union[str, object]] = None,
        interpolation: int = 3,
    ) -> 'CornerGeolocation':
        """Create a CornerGeolocation from a GRDL imagery reader.

        Searches the reader's metadata for corner coordinates in
        priority order:

        1. ``metadata.cscrna.corners`` — CSCRNA TRE precision corners,
           ``(4, 2)`` array ``[lat, lon]`` in order UL, UR, LR, LL.
        2. ``metadata.blocka`` — BLOCKA TRE 21-character corner
           strings (FRFC=UL, FRLC=UR, LRLC=LR, LRFC=LL).
        3. ``metadata.igeolo`` + ``metadata.icords`` — NITF image
           subheader corners (ICORDS forms G/C/D/N/S/U).

        A source that is present but unparseable falls through to the
        next source.

        Parameters
        ----------
        reader : ImageReader
            A GRDL imagery reader with populated metadata and
            ``get_shape()``.
        height : float, default=0.0
            Constant height above the WGS84 ellipsoid (meters).
        dem_path : str or Path, optional
            Path to DEM/DTED data.
        geoid_path : str or Path, optional
            Path to geoid correction file.
        interpolation : int, default=3
            Spline interpolation order for DEM sampling.

        Returns
        -------
        CornerGeolocation
            Configured geolocation with ``accuracy_source`` set to
            ``'CSCRNA'``, ``'BLOCKA'``, or ``'IGEOLO'``.

        Raises
        ------
        ValueError
            If no usable corner coordinate source is found.
        """
        meta = reader.metadata
        shape = tuple(reader.get_shape()[:2])
        common = dict(
            shape=shape, height=height, dem_path=dem_path,
            geoid_path=geoid_path, interpolation=interpolation,
        )

        # 1. CSCRNA TRE precision corners (access via getattr only —
        #    the cscrna metadata model may not exist in this build).
        cscrna = getattr(meta, 'cscrna', None)
        if cscrna is not None:
            cscrna_corners = getattr(cscrna, 'corners', None)
            if cscrna_corners is not None:
                return cls(
                    corners=np.asarray(cscrna_corners, dtype=np.float64),
                    accuracy_source='CSCRNA',
                    **common,
                )

        # 2. BLOCKA TRE corner strings
        blocka = getattr(meta, 'blocka', None)
        if blocka is not None:
            corner_strs = (
                getattr(blocka, 'frfc_loc', None),  # UL
                getattr(blocka, 'frlc_loc', None),  # UR
                getattr(blocka, 'lrlc_loc', None),  # LR
                getattr(blocka, 'lrfc_loc', None),  # LL
            )
            if all(s is not None for s in corner_strs):
                try:
                    corners = np.asarray(
                        [_parse_blocka_corner(s) for s in corner_strs],
                        dtype=np.float64,
                    )
                    return cls(
                        corners=corners,
                        accuracy_source='BLOCKA',
                        **common,
                    )
                except ValueError:
                    pass  # malformed BLOCKA — fall through to IGEOLO

        # 3. NITF IGEOLO header corners
        igeolo = getattr(meta, 'igeolo', None)
        icords = getattr(meta, 'icords', None)
        if igeolo is not None and icords is not None and icords.strip():
            corners = _parse_igeolo(igeolo, icords)
            return cls(
                corners=corners,
                accuracy_source='IGEOLO',
                **common,
            )

        raise ValueError(
            "No corner coordinate source available: reader metadata has "
            "no CSCRNA TRE, no complete BLOCKA TRE, and no IGEOLO/ICORDS "
            "header fields"
        )
