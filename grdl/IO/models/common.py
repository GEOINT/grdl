# -*- coding: utf-8 -*-
"""
IO Models Common - Reusable primitive types for metadata dataclasses.

Provides building-block dataclasses shared across SICD, SIDD, and other
sensor-specific metadata: coordinate types (XYZ, LatLon, LatLonHAE,
RowCol) and polynomial types (Poly1D, Poly2D, XYZPoly).

Polynomial classes are callable and support analytical differentiation,
enabling direct use in R/Rdot geolocation computations per SICD Volume 3.

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
2026-02-10

Modified
--------
2026-03-16
"""

# Standard library
from dataclasses import dataclass
from typing import Optional, Tuple, Union

# Third-party
import numpy as np


# ── Coordinate types ─────────────────────────────────────────────────


@dataclass
class XYZ:
    """ECF (Earth-Centered Fixed) coordinate or 3D vector in meters.

    Supports conversion to/from numpy arrays and basic vector operations
    (norm, dot, cross) needed for R/Rdot geolocation geometry.

    Parameters
    ----------
    x : float
        X component (meters).
    y : float
        Y component (meters).
    z : float
        Z component (meters).
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_array(self) -> np.ndarray:
        """Return components as a shape ``(3,)`` float64 array."""
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'XYZ':
        """Create from a length-3 array-like.

        Parameters
        ----------
        arr : array-like
            Sequence of three values ``[x, y, z]``.

        Returns
        -------
        XYZ
        """
        arr = np.asarray(arr, dtype=np.float64).ravel()
        if arr.size != 3:
            raise ValueError(f"Expected 3 elements, got {arr.size}")
        return cls(float(arr[0]), float(arr[1]), float(arr[2]))

    def norm(self) -> float:
        """Euclidean norm (magnitude) of the vector."""
        return float(np.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2))

    def dot(self, other: 'XYZ') -> float:
        """Dot product with another XYZ vector.

        Parameters
        ----------
        other : XYZ
            Right-hand operand.

        Returns
        -------
        float
        """
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: 'XYZ') -> 'XYZ':
        """Cross product with another XYZ vector.

        Parameters
        ----------
        other : XYZ
            Right-hand operand.

        Returns
        -------
        XYZ
            ``self × other``.
        """
        return XYZ(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def unit(self) -> 'XYZ':
        """Return unit vector in the same direction.

        Returns
        -------
        XYZ

        Raises
        ------
        ValueError
            If the vector is zero-length.
        """
        n = self.norm()
        if n == 0.0:
            raise ValueError("Cannot normalize zero-length vector")
        return XYZ(self.x / n, self.y / n, self.z / n)

    def __sub__(self, other: 'XYZ') -> 'XYZ':
        return XYZ(self.x - other.x, self.y - other.y, self.z - other.z)

    def __add__(self, other: 'XYZ') -> 'XYZ':
        return XYZ(self.x + other.x, self.y + other.y, self.z + other.z)

    def __mul__(self, scalar: float) -> 'XYZ':
        return XYZ(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> 'XYZ':
        return self.__mul__(scalar)

    def __neg__(self) -> 'XYZ':
        return XYZ(-self.x, -self.y, -self.z)


@dataclass
class LatLon:
    """WGS-84 geographic point (2D).

    Parameters
    ----------
    lat : float
        Latitude in degrees.
    lon : float
        Longitude in degrees.
    """

    lat: float = 0.0
    lon: float = 0.0


@dataclass
class LatLonHAE:
    """WGS-84 geographic point with Height Above Ellipsoid.

    Parameters
    ----------
    lat : float
        Latitude in degrees.
    lon : float
        Longitude in degrees.
    hae : float
        Height Above Ellipsoid in meters.
    """

    lat: float = 0.0
    lon: float = 0.0
    hae: float = 0.0


@dataclass
class RowCol:
    """Row/column pixel coordinate.

    Parameters
    ----------
    row : float
        Row (line) coordinate.
    col : float
        Column (sample) coordinate.
    """

    row: float = 0.0
    col: float = 0.0


# ── Polynomial types ─────────────────────────────────────────────────


@dataclass
class Poly1D:
    """1D polynomial with evaluation and analytical differentiation.

    The polynomial value at point ``x`` is::

        P(x) = coefs[0] + coefs[1]*x + coefs[2]*x**2 + ...

    Callable: ``poly(x)`` evaluates the polynomial.  Supports scalar
    and array inputs via ``numpy.polyval``.

    Parameters
    ----------
    coefs : numpy.ndarray, optional
        1D array of polynomial coefficients, shape ``(order+1,)``.
        Index ``i`` is the coefficient of ``x**i`` (ascending order).
    """

    coefs: Optional[np.ndarray] = None

    @property
    def order(self) -> int:
        """Polynomial order (degree).  Returns -1 if no coefficients."""
        if self.coefs is None:
            return -1
        return len(self.coefs) - 1

    def __call__(
        self, x: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Evaluate the polynomial at ``x``.

        Parameters
        ----------
        x : float or np.ndarray
            Evaluation point(s).

        Returns
        -------
        float or np.ndarray
            ``P(x)``.

        Raises
        ------
        ValueError
            If coefficients are not set.
        """
        if self.coefs is None:
            raise ValueError("Poly1D coefficients are not set")
        # np.polyval expects descending order (highest power first)
        return np.polyval(self.coefs[::-1], x)

    def derivative(self, n: int = 1) -> 'Poly1D':
        """Return the *n*-th analytical derivative as a new Poly1D.

        Parameters
        ----------
        n : int
            Derivative order (default 1).

        Returns
        -------
        Poly1D
            Derivative polynomial.  A zero-order constant ``[0.0]``
            is returned when the derivative exceeds the polynomial order.

        Raises
        ------
        ValueError
            If coefficients are not set or *n* < 1.
        """
        if self.coefs is None:
            raise ValueError("Poly1D coefficients are not set")
        if n < 1:
            raise ValueError(f"Derivative order must be >= 1, got {n}")
        c = np.array(self.coefs, dtype=np.float64)
        for _ in range(n):
            if len(c) <= 1:
                c = np.array([0.0])
                break
            c = c[1:] * np.arange(1, len(c))
        return Poly1D(coefs=c)

    def derivative_eval(
        self, x: Union[float, np.ndarray], n: int = 1,
    ) -> Union[float, np.ndarray]:
        """Evaluate the *n*-th derivative at ``x``.

        Parameters
        ----------
        x : float or np.ndarray
            Evaluation point(s).
        n : int
            Derivative order (default 1).

        Returns
        -------
        float or np.ndarray
        """
        return self.derivative(n)(x)


@dataclass
class Poly2D:
    """2D polynomial with evaluation and analytical differentiation.

    The polynomial value at point ``(x, y)`` is::

        P(x, y) = sum_{i,j} coefs[i, j] * x**i * y**j

    Callable: ``poly(x, y)`` evaluates the polynomial.  Supports
    scalar and array inputs (broadcast-compatible).

    Parameters
    ----------
    coefs : numpy.ndarray, optional
        2D array of polynomial coefficients, shape
        ``(order_x+1, order_y+1)``.  ``coefs[i, j]`` is the
        coefficient of ``x**i * y**j``.
    """

    coefs: Optional[np.ndarray] = None

    @property
    def order(self) -> Tuple[int, int]:
        """Polynomial orders ``(order_x, order_y)``.

        Returns ``(-1, -1)`` if no coefficients.
        """
        if self.coefs is None:
            return (-1, -1)
        return (self.coefs.shape[0] - 1, self.coefs.shape[1] - 1)

    def __call__(
        self,
        x: Union[float, np.ndarray],
        y: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Evaluate the polynomial at ``(x, y)``.

        Uses Horner-like row-wise evaluation for efficiency:
        for each power of *x*, evaluate the inner 1D polynomial in *y*,
        then accumulate with powers of *x*.

        Parameters
        ----------
        x : float or np.ndarray
            First variable evaluation point(s).
        y : float or np.ndarray
            Second variable evaluation point(s).

        Returns
        -------
        float or np.ndarray
            ``P(x, y)``.

        Raises
        ------
        ValueError
            If coefficients are not set.
        """
        if self.coefs is None:
            raise ValueError("Poly2D coefficients are not set")
        c = np.asarray(self.coefs, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        # Evaluate: for each i, compute row_val[i] = sum_j c[i,j]*y**j
        # then result = sum_i row_val[i] * x**i
        # Use Horner in x (outer) with inner polyval in y
        nx = c.shape[0]
        # Start from highest x power and work down (Horner)
        result = np.polyval(c[nx - 1, ::-1], y)
        for i in range(nx - 2, -1, -1):
            result = result * x + np.polyval(c[i, ::-1], y)
        return result

    def derivative(self, dx: int = 0, dy: int = 0) -> 'Poly2D':
        """Return the analytical partial derivative as a new Poly2D.

        Parameters
        ----------
        dx : int
            Derivative order with respect to *x* (default 0).
        dy : int
            Derivative order with respect to *y* (default 0).

        Returns
        -------
        Poly2D
            Partial derivative polynomial.
        """
        if self.coefs is None:
            raise ValueError("Poly2D coefficients are not set")
        c = np.array(self.coefs, dtype=np.float64)
        # Differentiate in x (rows)
        for _ in range(dx):
            if c.shape[0] <= 1:
                c = np.zeros((1, c.shape[1]))
                break
            factors = np.arange(1, c.shape[0])
            c = c[1:, :] * factors[:, np.newaxis]
        # Differentiate in y (columns)
        for _ in range(dy):
            if c.shape[1] <= 1:
                c = np.zeros((c.shape[0], 1))
                break
            factors = np.arange(1, c.shape[1])
            c = c[:, 1:] * factors[np.newaxis, :]
        return Poly2D(coefs=c)

    def derivative_eval(
        self,
        x: Union[float, np.ndarray],
        y: Union[float, np.ndarray],
        dx: int = 0,
        dy: int = 0,
    ) -> Union[float, np.ndarray]:
        """Evaluate the partial derivative at ``(x, y)``.

        Parameters
        ----------
        x, y : float or np.ndarray
            Evaluation point(s).
        dx, dy : int
            Derivative orders.

        Returns
        -------
        float or np.ndarray
        """
        return self.derivative(dx, dy)(x, y)


@dataclass
class XYZPoly:
    """Position polynomial in three ECF components.

    Each component is a :class:`Poly1D` in time.  Calling ``xyz_poly(t)``
    returns a shape ``(3,)`` or ``(N, 3)`` array of ECF positions.

    Parameters
    ----------
    x : Poly1D, optional
        X-component polynomial.
    y : Poly1D, optional
        Y-component polynomial.
    z : Poly1D, optional
        Z-component polynomial.
    """

    x: Optional[Poly1D] = None
    y: Optional[Poly1D] = None
    z: Optional[Poly1D] = None

    def __call__(self, t: Union[float, np.ndarray]) -> np.ndarray:
        """Evaluate position at time ``t``.

        Parameters
        ----------
        t : float or np.ndarray
            Time value(s) (seconds, relative to collection start).

        Returns
        -------
        np.ndarray
            Shape ``(3,)`` for scalar *t*, or ``(N, 3)`` for array *t*.
            Columns are ``[X, Y, Z]`` in ECF meters.

        Raises
        ------
        ValueError
            If any component polynomial is not set.
        """
        if self.x is None or self.y is None or self.z is None:
            raise ValueError("All three XYZPoly components must be set")
        px = self.x(t)
        py = self.y(t)
        pz = self.z(t)
        scalar = np.ndim(t) == 0
        if scalar:
            return np.array([px, py, pz], dtype=np.float64)
        return np.column_stack([px, py, pz])

    def derivative(self, n: int = 1) -> 'XYZPoly':
        """Return the *n*-th time derivative as a new XYZPoly.

        For ``n=1`` this gives the velocity polynomial; for ``n=2``,
        the acceleration polynomial.

        Parameters
        ----------
        n : int
            Derivative order (default 1).

        Returns
        -------
        XYZPoly
            Derivative polynomial.

        Raises
        ------
        ValueError
            If any component polynomial is not set.
        """
        if self.x is None or self.y is None or self.z is None:
            raise ValueError("All three XYZPoly components must be set")
        return XYZPoly(
            x=self.x.derivative(n),
            y=self.y.derivative(n),
            z=self.z.derivative(n),
        )

    def derivative_eval(
        self, t: Union[float, np.ndarray], n: int = 1,
    ) -> np.ndarray:
        """Evaluate the *n*-th derivative at time ``t``.

        Parameters
        ----------
        t : float or np.ndarray
            Time value(s).
        n : int
            Derivative order (default 1).

        Returns
        -------
        np.ndarray
            Shape ``(3,)`` or ``(N, 3)`` — velocity, acceleration, etc.
        """
        return self.derivative(n)(t)
