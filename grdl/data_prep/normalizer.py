# -*- coding: utf-8 -*-
"""
Normalizer - Per-chip or per-image intensity normalization.

Provides configurable intensity normalization with fit/transform semantics.
Supports min-max scaling, z-score standardization, percentile clipping,
and unit-norm normalization. All operations are fully vectorized using
numpy and work on arrays of any shape.

Author
------
Steven Siebert

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-02-06

Modified
--------
2026-02-06
"""

# Standard library
from typing import Optional

# Third-party
import numpy as np


_VALID_METHODS = ('minmax', 'zscore', 'percentile', 'unit_norm')


class Normalizer:
    """Per-chip or per-image intensity normalization.

    Supports four normalization methods:

    - ``'minmax'``: Scale values to [0, 1] based on min/max.
    - ``'zscore'``: Subtract mean and divide by standard deviation.
    - ``'percentile'``: Clip to percentile range, then scale to [0, 1].
    - ``'unit_norm'``: Divide by the L2 norm of the array.

    Use ``normalize()`` for stateless one-shot normalization, or
    ``fit()`` / ``transform()`` for reusable normalization parameters
    (e.g., normalizing test data with training statistics).

    Parameters
    ----------
    method : str
        One of ``'minmax'``, ``'zscore'``, ``'percentile'``,
        ``'unit_norm'``.
    percentile_low : float
        Lower percentile for ``'percentile'`` method. Default ``2.0``.
    percentile_high : float
        Upper percentile for ``'percentile'`` method. Default ``98.0``.
    epsilon : float
        Small value to avoid division by zero. Default ``1e-10``.

    Raises
    ------
    ValueError
        If method is not one of the valid options, or if percentile
        bounds are invalid.

    Examples
    --------
    Stateless normalization:

    >>> import numpy as np
    >>> from grdl.data_prep import Normalizer
    >>> norm = Normalizer(method='minmax')
    >>> data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    >>> norm.normalize(data)
    array([0.  , 0.25, 0.5 , 0.75, 1.  ])

    Fit/transform pattern:

    >>> norm = Normalizer(method='zscore')
    >>> train = np.random.rand(1000)
    >>> norm.fit(train)
    >>> test_normalized = norm.transform(np.random.rand(100))
    """

    def __init__(
        self,
        method: str = 'minmax',
        percentile_low: float = 2.0,
        percentile_high: float = 98.0,
        epsilon: float = 1e-10,
    ) -> None:
        if method not in _VALID_METHODS:
            raise ValueError(
                f"method must be one of {_VALID_METHODS}, got '{method}'"
            )
        if percentile_low < 0.0 or percentile_low > 100.0:
            raise ValueError(
                f"percentile_low must be in [0, 100], got {percentile_low}"
            )
        if percentile_high < 0.0 or percentile_high > 100.0:
            raise ValueError(
                f"percentile_high must be in [0, 100], "
                f"got {percentile_high}"
            )
        if percentile_low >= percentile_high:
            raise ValueError(
                f"percentile_low ({percentile_low}) must be less than "
                f"percentile_high ({percentile_high})"
            )

        self._method = method
        self._percentile_low = percentile_low
        self._percentile_high = percentile_high
        self._epsilon = epsilon

        # Fitted parameters (set by fit())
        self._min: Optional[float] = None
        self._max: Optional[float] = None
        self._mean: Optional[float] = None
        self._std: Optional[float] = None
        self._pct_low_val: Optional[float] = None
        self._pct_high_val: Optional[float] = None
        self._l2_norm: Optional[float] = None
        self._is_fitted: bool = False

    @property
    def method(self) -> str:
        """The normalization method.

        Returns
        -------
        str
            One of ``'minmax'``, ``'zscore'``, ``'percentile'``,
            ``'unit_norm'``.
        """
        return self._method

    @property
    def is_fitted(self) -> bool:
        """Whether ``fit()`` has been called.

        Returns
        -------
        bool
            True if normalization parameters have been computed.
        """
        return self._is_fitted

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize array values using statistics computed from the input.

        This is a stateless operation: statistics are computed from
        ``data`` and applied immediately. Does not store parameters.
        Works on any shape array.

        Parameters
        ----------
        data : np.ndarray
            Input array of any shape.

        Returns
        -------
        np.ndarray
            Normalized array with same shape as input, dtype float64.

        Raises
        ------
        TypeError
            If data is not a numpy ndarray.
        """
        self._validate_data(data)
        result = data.astype(np.float64)

        if self._method == 'minmax':
            dmin = result.min()
            dmax = result.max()
            drange = dmax - dmin
            if drange < self._epsilon:
                return np.zeros_like(result)
            return (result - dmin) / drange

        if self._method == 'zscore':
            dmean = result.mean()
            dstd = result.std()
            if dstd < self._epsilon:
                return np.zeros_like(result)
            return (result - dmean) / dstd

        if self._method == 'percentile':
            plow = np.percentile(result, self._percentile_low)
            phigh = np.percentile(result, self._percentile_high)
            prange = phigh - plow
            if prange < self._epsilon:
                return np.zeros_like(result)
            result = np.clip(result, plow, phigh)
            return (result - plow) / prange

        # unit_norm
        l2 = np.linalg.norm(result)
        if l2 < self._epsilon:
            return np.zeros_like(result)
        return result / l2

    def fit(self, data: np.ndarray) -> 'Normalizer':
        """Compute normalization parameters from data without transforming.

        Stores statistics (min, max, mean, std, percentiles, L2 norm) so
        that ``transform()`` can be called later with the same parameters.
        Returns self for method chaining.

        Parameters
        ----------
        data : np.ndarray
            Input array of any shape. Statistics are computed over the
            full flattened array.

        Returns
        -------
        Normalizer
            Self, for method chaining.

        Raises
        ------
        TypeError
            If data is not a numpy ndarray.
        """
        self._validate_data(data)
        arr = data.astype(np.float64)

        self._min = float(arr.min())
        self._max = float(arr.max())
        self._mean = float(arr.mean())
        self._std = float(arr.std())
        self._pct_low_val = float(
            np.percentile(arr, self._percentile_low)
        )
        self._pct_high_val = float(
            np.percentile(arr, self._percentile_high)
        )
        self._l2_norm = float(np.linalg.norm(arr))
        self._is_fitted = True

        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply previously fitted normalization parameters.

        Uses statistics computed during ``fit()`` to normalize the input.
        This allows normalizing new data (e.g., test sets) with the same
        parameters used for training data.

        Parameters
        ----------
        data : np.ndarray
            Input array of any shape.

        Returns
        -------
        np.ndarray
            Normalized array with same shape as input, dtype float64.

        Raises
        ------
        RuntimeError
            If ``fit()`` was not called first.
        TypeError
            If data is not a numpy ndarray.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Normalizer has not been fitted. Call fit() first."
            )
        self._validate_data(data)
        result = data.astype(np.float64)

        if self._method == 'minmax':
            drange = self._max - self._min
            if drange < self._epsilon:
                return np.zeros_like(result)
            return (result - self._min) / drange

        if self._method == 'zscore':
            if self._std < self._epsilon:
                return np.zeros_like(result)
            return (result - self._mean) / self._std

        if self._method == 'percentile':
            prange = self._pct_high_val - self._pct_low_val
            if prange < self._epsilon:
                return np.zeros_like(result)
            result = np.clip(result, self._pct_low_val, self._pct_high_val)
            return (result - self._pct_low_val) / prange

        # unit_norm
        if self._l2_norm < self._epsilon:
            return np.zeros_like(result)
        return result / self._l2_norm

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Convenience: fit then transform in one call.

        Equivalent to calling ``fit(data)`` followed by ``transform(data)``.

        Parameters
        ----------
        data : np.ndarray
            Input array of any shape.

        Returns
        -------
        np.ndarray
            Normalized array with same shape as input, dtype float64.

        Raises
        ------
        TypeError
            If data is not a numpy ndarray.
        """
        self.fit(data)
        return self.transform(data)

    @staticmethod
    def _validate_data(data: np.ndarray) -> None:
        """Validate that input is a numpy ndarray.

        Parameters
        ----------
        data : np.ndarray
            Array to validate.

        Raises
        ------
        TypeError
            If data is not a numpy ndarray.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError(
                f"data must be np.ndarray, got {type(data).__name__}"
            )

    def __repr__(self) -> str:
        return (
            f"Normalizer(method='{self._method}', "
            f"percentile_low={self._percentile_low}, "
            f"percentile_high={self._percentile_high}, "
            f"epsilon={self._epsilon})"
        )
