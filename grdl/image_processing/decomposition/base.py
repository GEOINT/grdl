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
170194430+DDSmalls@users.noreply.github.com

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
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

# Third-party
import numpy as np

# GRDL internal
from grdl.image_processing.base import ImageProcessor

if TYPE_CHECKING:
    from grdl.IO.models.base import ChannelMetadata, ImageMetadata


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
        if shh is None and source.ndim == 3:
            axis_order = getattr(metadata, 'axis_order', None)

            if axis_order in ('CYX', 'YXC'):
                inferred_order = axis_order
            else:
                inferred_order = None
                channel_metadata = getattr(metadata, 'channel_metadata', None)
                if channel_metadata is not None:
                    n_channels = len(channel_metadata)
                    if source.shape[0] == n_channels and source.shape[-1] != n_channels:
                        inferred_order = 'CYX'
                    elif source.shape[-1] == n_channels and source.shape[0] != n_channels:
                        inferred_order = 'YXC'
                if inferred_order is None:
                    bands = getattr(metadata, 'bands', None)
                    if bands is not None:
                        if source.shape[0] == bands and source.shape[-1] != bands:
                            inferred_order = 'CYX'
                        elif source.shape[-1] == bands and source.shape[0] != bands:
                            inferred_order = 'YXC'

            if inferred_order == 'CYX' and source.shape[0] >= 4:
                shh = source[0]
                shv = source[1]
                svh = source[2]
                svv = source[3]
            elif inferred_order == 'YXC' and source.shape[-1] >= 4:
                shh = source[..., 0]
                shv = source[..., 1]
                svh = source[..., 2]
                svv = source[..., 3]
            elif source.shape[0] >= 4:
                shh = source[0]
                shv = source[1]
                svh = source[2]
                svv = source[3]
            elif source.shape[-1] >= 4:
                shh = source[..., 0]
                shv = source[..., 1]
                svh = source[..., 2]
                svv = source[..., 3]
        components = self.decompose(shh, shv, svh, svv)
        updated = dataclasses.replace(
            metadata,
            bands=len(components),
            axis_order='CYX',
            channel_metadata=self._build_component_metadata(metadata),
        )
        return components, updated

    def _build_component_metadata(
        self,
        metadata: 'ImageMetadata',
    ) -> list['ChannelMetadata']:
        """Build per-component metadata for decomposition outputs."""
        from grdl.IO.models.base import ChannelMetadata

        return [
            ChannelMetadata(
                index=i,
                name=name,
                role='decomposition',
                source_indices=[],
            )
            for i, name in enumerate(self.component_names)
        ]

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
        color_mode: str = 'standard',
        channels: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, 'ImageMetadata']:
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
        color_mode : str
            ``'standard'`` stacks the selected channels directly as RGB.
            ``'perceptual'`` blends the channels in CIELAB space before
            converting back to display RGB.
        channels : list of str, optional
            Override which 3 component keys map to R, G, B (in that order).
            When provided, must contain exactly 3 valid component key names.
            Defaults to the decomposition's canonical R/G/B mapping.

        Returns
        -------
        tuple[np.ndarray, ImageMetadata]
            ``(rgb, metadata)`` — rgb is shape (3, rows, cols), dtype
            float32, values in [0, 1]; metadata carries the channel
            descriptors for the three output bands.
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

    def _validate_internal_matrix_window_size(
        self,
        precomputed_method: str,
    ) -> None:
        """Reject ``decompose()`` calls that try to build a 1x1 averaged matrix."""
        window_size = getattr(self, 'window_size', None)
        if window_size is not None and window_size < 3:
            raise ValueError(
                f"{type(self).__name__}.decompose() requires window_size >= 3 "
                f"when building the averaging matrix internally; got "
                f"window_size={window_size}. Use {precomputed_method}(...) "
                f"with a precomputed matrix if you need window_size=1."
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

    def _apply_power_representation(
        self,
        arr: np.ndarray,
        representation: str = 'db',
    ) -> np.ndarray:
        """
        Convert a real-valued power-like image for display normalization.

        Parameters
        ----------
        arr : np.ndarray
            Real-valued component image (typically power/intensity).
        representation : str
            One of ``'db'``, ``'magnitude'``, or ``'power'``.

        Returns
        -------
        np.ndarray
            Converted real-valued image.
        """
        arr = np.real(np.asarray(arr, dtype=np.float64))
        if representation == 'power':
            return arr
        if representation == 'magnitude':
            return np.sqrt(np.clip(arr, 0.0, None))
        if representation == 'db':
            tiny = np.finfo(np.float64).tiny
            with np.errstate(divide='ignore', invalid='ignore'):
                return 10.0 * np.log10(np.maximum(arr, tiny))
        raise ValueError(
            f"representation must be one of ['db', 'magnitude', 'power'], got {representation!r}"
        )

    def _percentile_stretch(
        self,
        arr: np.ndarray,
        percentile_low: float = 2.0,
        percentile_high: float = 98.0,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> np.ndarray:
        """
        Percentile-stretch an array to [0, 1].

        Parameters
        ----------
        arr : np.ndarray
            Real-valued 2D array.
        percentile_low : float
            Lower percentile used to derive ``vmin`` when ``vmin`` is not
            supplied explicitly. Default 2.0.
        percentile_high : float
            Upper percentile used to derive ``vmax`` when ``vmax`` is not
            supplied explicitly. Default 98.0.
        vmin : float, optional
            Explicit lower clip value.  When provided, overrides
            ``percentile_low`` for this call.  Pass the same value to
            multiple ``_percentile_stretch`` calls to produce RGB bands
            that share a common stretch range, enabling colour-comparable
            composites across different images or algorithms.
        vmax : float, optional
            Explicit upper clip value.  When provided, overrides
            ``percentile_high`` for this call.

        Returns
        -------
        np.ndarray
            Stretched array, dtype float32, values clipped to [0, 1].
        """
        finite_mask = np.isfinite(arr)
        if not np.any(finite_mask):
            return np.zeros_like(arr, dtype=np.float32)

        if vmin is None or vmax is None:
            vals = arr[finite_mask]
            if vmin is None:
                vmin = float(np.percentile(vals, percentile_low))
            if vmax is None:
                vmax = float(np.percentile(vals, percentile_high))

        span = vmax - vmin
        if span < np.finfo(np.float32).eps:
            return np.zeros_like(arr, dtype=np.float32)

        out = (arr - vmin) / span
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    def _build_power_rgb(
        self,
        components: Dict[str, np.ndarray],
        channel_map: List[Tuple[str, str]],
        format_name: str,
        representation: str = 'db',
        percentile_low: float = 2.0,
        percentile_high: float = 98.0,
        color_mode: str = 'standard',
        channels: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, 'ImageMetadata']:
        """Build an RGB composite from real-valued power-like components.

        Applies ``_apply_power_representation`` then ``_percentile_stretch``
        to each channel independently, stacks them into ``(3, rows, cols)``,
        and wraps the result in an ``ImageMetadata`` object.

        This helper is intended for decompositions whose R/G/B channels are
        real-valued power (or power-derived) quantities with the same semantic
        role — i.e. FreemanDurden, ModelFree3C/4C, CompactPolModelFree3C, and
        Yamaguchi.  H/A/alpha, Touzi, Praks, Neumann, and DoP use
        physics-specific normalisation that does not fit this pattern and
        should continue to override ``to_rgb`` directly.

        Parameters
        ----------
        components : Dict[str, np.ndarray]
            Output of ``decompose()`` or equivalent.
        channel_map : list of (component_key, rgb_role)
            Exactly 3 entries in R/G/B order used when ``channels`` is None.
        format_name : str
            Value for ``ImageMetadata.format`` (e.g. ``'FreemanDurdenRGB'``).
        representation : str
            ``'db'`` (default), ``'power'``, or ``'magnitude'``.
        percentile_low : float
            Lower percentile for stretch. Default 2.0.
        percentile_high : float
            Upper percentile for stretch. Default 98.0.
        color_mode : str
            ``'standard'`` stacks the three selected channels directly as RGB.
            ``'perceptual'`` blends the channels in CIELAB space before
            converting back to display RGB.
        channels : list of str, optional
            Override the R/G/B channel selection.  Exactly 3 component key
            names in R, G, B order.  When provided, *channel_map* is
            ignored.  Useful for visualising non-default components, e.g.
            ``channels=['helix', 'volume', 'surface']``.

        Returns
        -------
        tuple[np.ndarray, ImageMetadata]
            ``(rgb, metadata)`` — rgb shape ``(3, rows, cols)``, float32.

        Raises
        ------
        ValueError
            If any component key is missing from *components*, or the
            resolved channel selection does not contain exactly 3 entries.
        """
        from grdl.IO.models.base import ImageMetadata, ChannelMetadata

        _roles = ('rgb_red', 'rgb_green', 'rgb_blue')
        if channels is not None:
            if len(channels) != 3:
                raise ValueError(
                    f"channels must have exactly 3 entries, got {len(channels)}"
                )
            channel_map = list(zip(channels, _roles))

        if len(channel_map) != 3:
            raise ValueError(
                f"channel_map must have exactly 3 entries, got {len(channel_map)}"
            )

        missing = {k for k, _ in channel_map} - set(components.keys())
        if missing:
            raise ValueError(f"Missing component keys: {missing}")

        bands = [
            self._percentile_stretch(
                self._apply_power_representation(components[key], representation),
                percentile_low,
                percentile_high,
            )
            for key, _ in channel_map
        ]
        rgb = self._bands_to_rgb(
            bands,
            color_mode=color_mode,
            channel_keys=[key for key, _ in channel_map],
        )

        meta = ImageMetadata(
            format=format_name,
            rows=rgb.shape[1],
            cols=rgb.shape[2],
            bands=3,
            dtype='float32',
            axis_order='CYX',
            channel_metadata=[
                ChannelMetadata(index=i, name=key, role=role)
                for i, (key, role) in enumerate(channel_map)
            ],
        )
        return rgb, meta

    def _bands_to_rgb(
        self,
        bands: List[np.ndarray],
        color_mode: str = 'standard',
        channel_keys: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Convert three display bands to an RGB cube.

        ``standard`` mode stacks the bands directly. ``perceptual`` mode
        blends the bands in CIELAB space using canonical red/green/blue
        anchors and then converts back to RGB.
        """
        if len(bands) != 3:
            raise ValueError(f"bands must have exactly 3 entries, got {len(bands)}")

        stack = np.stack([np.asarray(b, dtype=np.float64) for b in bands], axis=-1)

        if color_mode == 'standard':
            return np.moveaxis(stack.astype(np.float32, copy=False), -1, 0)

        if color_mode != 'perceptual':
            raise ValueError(
                "color_mode must be one of ['standard', 'perceptual'], "
                f"got {color_mode!r}"
            )

        if channel_keys is not None and len(set(channel_keys)) == 1:
            gray = np.clip(np.mean(stack, axis=-1), 0.0, 1.0).astype(np.float32)
            return np.stack([gray, gray, gray], axis=0)

        try:
            from skimage.color import lab2rgb, rgb2lab
        except ImportError as exc:  # pragma: no cover - dependency check
            raise ImportError(
                "perceptual color_mode requires scikit-image"
            ) from exc

        anchors_rgb = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        anchors_lab = rgb2lab(anchors_rgb[np.newaxis, :, :])[0]

        total = np.sum(stack, axis=-1, keepdims=True)
        safe_total = np.maximum(total, np.finfo(np.float64).eps)
        weights = stack / safe_total

        lab = np.einsum('...k,kj->...j', weights, anchors_lab)
        lab[..., 0] = 100.0 * np.clip(np.mean(stack, axis=-1), 0.0, 1.0)
        lab[..., 1:] *= np.clip(total, 0.0, 1.0)

        rgb = lab2rgb(lab)
        return np.moveaxis(np.clip(rgb, 0.0, 1.0).astype(np.float32), -1, 0)
