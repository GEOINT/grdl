# -*- coding: utf-8 -*-
"""
Polarimetric Decomposition Base Classes - Abstract interfaces for decompositions.

Defines the abstract base class for polarimetric decomposition methods.
Concrete implementations (Pauli, Freeman-Durden, Yamaguchi, etc.) inherit
from PolarimetricDecomposition and implement the decompose() method.

All decompositions operate on the complex-valued scattering matrix elements
(S_HH, S_HV, S_VH, S_VV) and return a dictionary of named complex-valued
components. Phase and interference are fully preserved in the output.
Conversion to magnitude, power, or dB is a separate explicit step.

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
2026-01-30

Modified
--------
2026-02-06
"""

# Standard library
import dataclasses
from abc import abstractmethod
from typing import Any, Dict, Tuple, TYPE_CHECKING

# Third-party
import numpy as np

# GRDL internal
from grdl.image_processing.base import ImageProcessor

if TYPE_CHECKING:
    from grdl.IO.models.base import ImageMetadata


class PolarimetricDecomposition(ImageProcessor):
    """
    Abstract base class for polarimetric decomposition methods.

    Decompositions take the four complex-valued elements of the 2x2
    scattering matrix [S] and produce named complex-valued components.
    Phase information from complex channel mixing (constructive and
    destructive interference) is preserved in the output.

    The scattering matrix is::

        [S] = [[S_HH, S_HV],
               [S_VH, S_VV]]

    Subclasses implement ``decompose`` which performs the decomposition,
    ``component_names`` which lists the output keys, and ``to_rgb``
    which maps components to an RGB composite.

    Concrete convenience methods ``to_power``, ``to_magnitude``, and
    ``to_db`` are provided for converting complex components to
    real-valued representations.

    Examples
    --------
    >>> from grdl.image_processing import PauliDecomposition
    >>> pauli = PauliDecomposition()
    >>> components = pauli.decompose(shh, shv, svh, svv)
    >>> db = pauli.to_db(components)
    >>> rgb = pauli.to_rgb(components)
    """

    def execute(
        self,
        metadata: 'ImageMetadata',
        source: np.ndarray,
        **kwargs: Any,
    ) -> tuple:
        """Execute the decomposition via the universal protocol.

        Quad-pol scattering matrix channels can be provided as keyword
        arguments (``shh``, ``shv``, ``svh``, ``svv``) or extracted from
        a 4-band source array (last axis).

        Parameters
        ----------
        metadata : ImageMetadata
            Input image metadata.
        source : np.ndarray
            Input array — either a 2-D single-channel or 3-D with 4+
            bands containing the quad-pol channels.

        Returns
        -------
        tuple[Dict[str, np.ndarray], ImageMetadata]
        """
        self._metadata = metadata
        shh = kwargs.pop('shh', None)
        shv = kwargs.pop('shv', None)
        svh = kwargs.pop('svh', None)
        svv = kwargs.pop('svv', None)
        if shh is None and source.ndim == 3 and source.shape[-1] >= 4:
            shh = source[..., 0]
            shv = source[..., 1]
            svh = source[..., 2]
            svv = source[..., 3]
        components = self.decompose(shh, shv, svh, svv)
        updated = dataclasses.replace(metadata, bands=len(components))
        return components, updated

    @abstractmethod
    def decompose(
        self,
        shh: np.ndarray,
        shv: np.ndarray,
        svh: np.ndarray,
        svv: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Decompose quad-pol scattering matrix into named complex components.

        Parameters
        ----------
        shh : np.ndarray
            Complex S_HH channel. Shape (rows, cols).
        shv : np.ndarray
            Complex S_HV channel. Shape (rows, cols).
        svh : np.ndarray
            Complex S_VH channel. Shape (rows, cols).
        svv : np.ndarray
            Complex S_VV channel. Shape (rows, cols).

        Returns
        -------
        Dict[str, np.ndarray]
            Named decomposition components. All values are complex-valued
            arrays with the same shape as the inputs. Keys are
            decomposition-specific (see ``component_names``).

        Raises
        ------
        TypeError
            If any input is not complex-valued.
        ValueError
            If inputs are not 2D or have mismatched shapes.
        """
        ...

    @property
    @abstractmethod
    def component_names(self) -> Tuple[str, ...]:
        """
        Names of the decomposition components.

        Returns
        -------
        Tuple[str, ...]
            Ordered tuple of component names matching the keys
            returned by ``decompose()``.
        """
        ...

    @abstractmethod
    def to_rgb(
        self,
        components: Dict[str, np.ndarray],
        representation: str = 'db',
        percentile_low: float = 2.0,
        percentile_high: float = 98.0,
    ) -> np.ndarray:
        """
        Create an RGB composite from decomposition components.

        The mapping from components to R/G/B channels is
        decomposition-specific.

        Parameters
        ----------
        components : Dict[str, np.ndarray]
            Output of ``decompose()``. Complex-valued component arrays.
        representation : str
            How to convert complex components before stretching.
            One of ``'db'`` (20*log10(|z|)), ``'magnitude'`` (|z|),
            or ``'power'`` (|z|^2). Default ``'db'``.
        percentile_low : float
            Lower percentile for contrast stretch. Default 2.0.
        percentile_high : float
            Upper percentile for contrast stretch. Default 98.0.

        Returns
        -------
        np.ndarray
            RGB image, shape (rows, cols, 3), dtype float32,
            values in [0, 1].
        """
        ...

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_scattering_matrix(
        self,
        shh: np.ndarray,
        shv: np.ndarray,
        svh: np.ndarray,
        svv: np.ndarray,
    ) -> None:
        """
        Validate scattering matrix inputs.

        Parameters
        ----------
        shh, shv, svh, svv : np.ndarray
            The four scattering matrix channels.

        Raises
        ------
        TypeError
            If any input is not a numpy array or not complex-valued.
        ValueError
            If any input is not 2D or shapes do not match.
        """
        channels = {'shh': shh, 'shv': shv, 'svh': svh, 'svv': svv}

        for name, arr in channels.items():
            if not isinstance(arr, np.ndarray):
                raise TypeError(
                    f"{name} must be a numpy ndarray, got {type(arr).__name__}"
                )
            if not np.iscomplexobj(arr):
                raise TypeError(
                    f"{name} must be complex-valued (complex64 or complex128), "
                    f"got {arr.dtype}. Pass complex arrays from the SAR reader."
                )
            if arr.ndim != 2:
                raise ValueError(
                    f"{name} must be 2D (rows, cols), got {arr.ndim}D "
                    f"with shape {arr.shape}"
                )

        shape = shh.shape
        for name, arr in channels.items():
            if arr.shape != shape:
                raise ValueError(
                    f"All channels must have the same shape. "
                    f"shh has shape {shape}, but {name} has shape {arr.shape}"
                )

    # ------------------------------------------------------------------
    # Conversion methods (concrete)
    # ------------------------------------------------------------------

    def to_power(
        self, components: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Convert complex components to power (|z|^2).

        Parameters
        ----------
        components : Dict[str, np.ndarray]
            Output of ``decompose()``.

        Returns
        -------
        Dict[str, np.ndarray]
            Real-valued power arrays. Same keys, dtype float.
        """
        return {k: np.abs(v) ** 2 for k, v in components.items()}

    def to_magnitude(
        self, components: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Convert complex components to magnitude (|z|).

        Parameters
        ----------
        components : Dict[str, np.ndarray]
            Output of ``decompose()``.

        Returns
        -------
        Dict[str, np.ndarray]
            Real-valued magnitude arrays. Same keys, dtype float.
        """
        return {k: np.abs(v) for k, v in components.items()}

    def to_db(
        self,
        components: Dict[str, np.ndarray],
        floor: float = -50.0,
    ) -> Dict[str, np.ndarray]:
        """
        Convert complex components to magnitude in dB.

        Computes 20 * log10(|z|), clamped to a floor value.

        Parameters
        ----------
        components : Dict[str, np.ndarray]
            Output of ``decompose()``.
        floor : float
            Minimum dB value. Pixels below this are clamped.
            Default -50.0.

        Returns
        -------
        Dict[str, np.ndarray]
            Real-valued dB arrays. Same keys, dtype float.
        """
        result = {}
        for k, v in components.items():
            mag = np.abs(v)
            db = 20.0 * np.log10(mag + np.finfo(mag.dtype).tiny)
            np.maximum(db, floor, out=db)
            result[k] = db
        return result

    def _percentile_stretch(
        self,
        arr: np.ndarray,
        percentile_low: float = 2.0,
        percentile_high: float = 98.0,
    ) -> np.ndarray:
        """
        Percentile-stretch an array to [0, 1].

        Parameters
        ----------
        arr : np.ndarray
            Real-valued 2D array.
        percentile_low : float
            Lower percentile. Default 2.0.
        percentile_high : float
            Upper percentile. Default 98.0.

        Returns
        -------
        np.ndarray
            Stretched array, dtype float32, values clipped to [0, 1].
        """
        finite_mask = np.isfinite(arr)
        if not np.any(finite_mask):
            return np.zeros_like(arr, dtype=np.float32)

        vals = arr[finite_mask]
        vmin = np.percentile(vals, percentile_low)
        vmax = np.percentile(vals, percentile_high)
        span = vmax - vmin
        if span < np.finfo(np.float32).eps:
            return np.zeros_like(arr, dtype=np.float32)

        out = (arr - vmin) / span
        return np.clip(out, 0.0, 1.0).astype(np.float32)
