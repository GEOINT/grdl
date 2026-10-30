# -*- coding: utf-8 -*-
"""
Polarimetric Filter Base - Shared array and execution contracts.

Provides canonical apply() dispatch for Refined Lee, IDAN, and Wishart.
Matrix preparation and storage conversion live in decomposition.pol_matrix;
concrete filters implement only the matrix kernel and optional scalar hook.

Author
------
GRDL contributors

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-10-29

Modified
--------
2026-10-29
"""

import dataclasses
import warnings
from abc import abstractmethod
from typing import Any, Optional

import numpy as np

from grdl.IO.models.base import ChannelMetadata, ImageMetadata
from grdl.exceptions import ValidationError
from grdl.image_processing.base import BandwiseTransformMixin, ImageTransform
from grdl.image_processing.decomposition.pol_matrix import (
    _make_matrix_channel_metadata,
    pack_matrix,
    prepare_polarimetric_matrix,
    validate_matrix_type,
)
from grdl.image_processing.filters._validation import validate_kernel_size
from grdl.image_processing.filters.sar_base import SARFilter


class PolarimetricFilter(SARFilter):
    """Filter CCYX matrices or form them from explicitly described CYX channels.

    Concrete filters implement only ``_filter_matrix``. IDAN additionally
    supports independent image bands; only Wishart retains implicit legacy
    packed input. Layout conversion belongs to ``execute``'s input boundary.
    """

    _supports_images = False
    _legacy_packed = False
    matrix_type: Optional[str] = None

    def _resolve_params(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Validate effective settings without mutating instance defaults."""
        try:
            params = super()._resolve_params(kwargs)
        except (ValueError, TypeError) as error:
            raise ValidationError(str(error)) from error
        for name, value in params.items():
            if isinstance(value, (int, float)) and not np.isfinite(value):
                raise ValidationError(f'{name} must be finite, got {value!r}')
        validate_kernel_size(params['kernel_size'])
        if 'nlook' in params and params['nlook'] <= 0:
            raise ValidationError('nlook must be > 0')
        if 'similarity_threshold' in params and params['similarity_threshold'] <= 0:
            raise ValidationError('similarity_threshold must be in (0, 1]')
        return params

    def apply(
        self,
        source: np.ndarray,
        *,
        input_kind: Optional[str] = None,
        matrix_type: Optional[str] = None,
        channel_metadata: Optional[list[ChannelMetadata]] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Filter canonical arrays without changing a formed matrix's basis.

        ``input_kind`` is ``matrix`` (CCYX), ``channels`` (complex CYX),
        ``packed`` (real CYX upper triangle), or IDAN's ``image`` (YX/CYX).
        Matrix rank is unambiguous. Channel formation requires an explicit
        C2/T2/C3/T3 ``matrix_type``; C4/T4 are accepted only when preformed.
        Optional channel descriptors resolve polarization order, not layout.
        Algorithm overrides are validated per call without changing the instance.

        Parameters
        ----------
        source : np.ndarray
            Formed ``(N, N, rows, cols)`` matrix, complex channel cube,
            real packed cube, or IDAN image. N must be 2, 3, or 4.
        input_kind : {'matrix', 'channels', 'packed', 'image'}, optional
            Four-dimensional input defaults to matrix. IDAN defaults to image
            for other ranks; Wishart also recognizes real 3D packed input.
            Otherwise specify the representation explicitly.
        matrix_type : str, optional
            C2/T2/C3/T3/C4/T4 descriptor. Required for channel formation unless
            configured on the instance. For formed matrices, checks dimension
            only: it neither converts nor verifies the physical basis.
        channel_metadata : list of ChannelMetadata, optional
            Polarization names and roles used to order raw channels. Without
            descriptors, use co/cross for dual-pol or HH, HV, [VH,] VV.
        **kwargs : Any
            Annotated algorithm parameters overriding instance defaults.

        Returns
        -------
        np.ndarray
            CCYX matrix for matrix/channel input, real CYX packed bands for
            packed input, or independently filtered YX/CYX images for IDAN.
            Matrix output retains the prepared matrix's shape and dtype.

        Raises
        ------
        ValidationError
            If settings, representation, dimensions, dtype, or finite-value
            requirements are violated. Layout arguments are not accepted.
        """
        if 'axis_order' in kwargs:
            raise ValidationError('apply() expects canonical CYX/CCYX; normalize layout at the IO boundary')
        params = self._resolve_params(kwargs)
        kind = validate_matrix_type(self.matrix_type if matrix_type is None else matrix_type)
        if input_kind is None:
            if source.ndim == 4:
                input_kind = 'matrix'
            elif self._supports_images:
                input_kind = 'image'
            elif self._legacy_packed and source.ndim == 3 and not np.iscomplexobj(source):
                input_kind = 'packed'
            else:
                raise ValidationError(
                    'Specify input_kind="channels" and explicit matrix_type for SLC input; '
                    'filter_channels() is deprecated'
                )
        if input_kind == 'image':
            if not self._supports_images:
                raise ValidationError(f'{type(self).__name__} does not support independent images')
            if source.dtype.kind not in 'fciub' or not np.isfinite(source).all():
                raise ValidationError('Image input must be numeric and contain no non-finite values')
            return BandwiseTransformMixin.apply(self, source, **params)
        matrix = prepare_polarimetric_matrix(
            source, input_kind=input_kind, matrix_type=kind,
            channel_metadata=channel_metadata,
        )
        result = self._filter_matrix(matrix, params)
        return pack_matrix(result) if input_kind == 'packed' else result

    def execute(self, metadata: ImageMetadata, source: np.ndarray, **kwargs: Any) -> tuple:
        """Adapt labelled input to apply(), returning canonical output metadata.

        Parameters
        ----------
        metadata : ImageMetadata
            Input dimensions, layout, and optional channel descriptors. The
            shared transform boundary normalizes 3D YXC input to CYX.
        source : np.ndarray
            Input array corresponding to metadata.
        **kwargs : Any
            Input descriptors and effective algorithm parameters for apply().
            Complex 3D input defaults to channels for Refined Lee and Wishart;
            IDAN requires input_kind='channels' to select joint filtering.

        Returns
        -------
        tuple[np.ndarray, ImageMetadata]
            Result and copied metadata. Matrix results carry CCYX layout,
            N*N bands, element descriptors, spatial dimensions, and dtype.
            This adapter does not restore channel-last output.
        """
        try:
            source, canonical_metadata = ImageTransform._canonicalize_source(metadata, source)
        except ValueError as error:
            raise ValidationError(str(error)) from error
        options = dict(kwargs)
        if source.ndim == 3 and np.iscomplexobj(source) and not self._supports_images:
            options.setdefault('input_kind', 'channels')
        options.setdefault('channel_metadata', canonical_metadata.channel_metadata)
        result = self.apply(source, **options)
        if result.ndim != 4:
            return result, self._update_metadata_for_result(canonical_metadata, result)
        dimension = result.shape[0]
        kind = validate_matrix_type(options.get('matrix_type', self.matrix_type))
        descriptors = canonical_metadata.channel_metadata if source.ndim == 4 else None
        if descriptors is None or len(descriptors) != dimension * dimension:
            descriptors = _make_matrix_channel_metadata(dimension, kind or 'M')
        updated = dataclasses.replace(
            canonical_metadata, rows=result.shape[-2], cols=result.shape[-1],
            bands=dimension * dimension, axis_order='CCYX', dtype=str(result.dtype),
            channel_metadata=descriptors,
        )
        return result, updated

    def filter_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Filter a CCYX tensor through the deprecated positional adapter.

        Use ``apply(matrix, input_kind='matrix')`` instead. Emits
        DeprecationWarning and delegates validation to the canonical method.
        """
        warnings.warn(
            'filter_matrix() is deprecated; use apply(..., input_kind="matrix")',
            DeprecationWarning, stacklevel=2,
        )
        return self.apply(matrix, input_kind='matrix')

    def filter_channels(
        self,
        shh: np.ndarray,
        shv: np.ndarray,
        svh: Optional[np.ndarray] = None,
        svv: Optional[np.ndarray] = None,
        matrix_type: str = 'C3',
    ) -> np.ndarray:
        """Stack matching 2D SLC arrays through the deprecated adapter.

        Two channels form C2/T2. Three reciprocal channels are interpreted
        positionally as HH, HV, VV; four as HH, HV, VH, VV. For new code use
        ``apply(cube, input_kind='channels', matrix_type=...)`` so array order
        and optional descriptors are explicit. Emits DeprecationWarning.
        """
        warnings.warn(
            'filter_channels() is deprecated; use apply(..., input_kind="channels")',
            DeprecationWarning, stacklevel=2,
        )
        channels = [shh, shv]
        if svh is not None:
            channels.append(svh)
        if svv is not None:
            channels.append(svv)
        if any(channel.ndim != 2 or channel.shape != shh.shape for channel in channels):
            raise ValidationError('SLC channels must be 2D with matching shapes')
        return self.apply(
            np.stack(channels), input_kind='channels', matrix_type=matrix_type,
        )

    @abstractmethod
    def _filter_matrix(self, matrix: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        """Apply the algorithm to a validated CCYX tensor and effective settings."""
        ...