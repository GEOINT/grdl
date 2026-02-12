# -*- coding: utf-8 -*-
"""
Image Processing Base Classes - Abstract interfaces for image processors.

Defines the ``ImageProcessor`` common base class for all image processor types
(transforms, detectors, decompositions, SAR-specific processors) and the
``ImageTransform`` ABC for dense raster transforms. ``ImageProcessor``
provides version checking at first instantiation, detection input
declaration/validation, and ``typing.Annotated``-based tunable parameter
declarations with automatic ``__init__`` generation and runtime resolution
through ``**kwargs``.

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
2026-02-10
"""

# Standard library
import logging
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

# Third-party
import numpy as np

logger = logging.getLogger(__name__)

from grdl.image_processing.params import ParamSpec, collect_param_specs, _make_init

if TYPE_CHECKING:
    from grdl.image_processing.versioning import DetectionInputSpec
    from grdl.image_processing.detection.models import DetectionSet


class ImageProcessor(ABC):
    """
    Common base class for all image processors.

    All processor types -- dense raster transforms (``ImageTransform``),
    sparse vector detectors (``ImageDetector``), polarimetric
    decompositions (``PolarimetricDecomposition``), and SAR-specific
    processors (``SublookDecomposition``) -- inherit from this class.

    Provides three cross-cutting capabilities:

    **Version checking**: Concrete subclasses that do not declare a processor
    version via ``@processor_version('x.y.z')`` will trigger a
    ``UserWarning`` at first instantiation.  The check uses ``__new__``
    rather than ``__init_subclass__`` so that decorators have been applied
    by the time the check runs.

    **Detection input flow**: Any processor can declare that it accepts
    ``DetectionSet`` inputs from upstream detectors by overriding
    ``detection_input_specs``.  Detection inputs are passed as keyword
    arguments to ``apply()`` or ``detect()``, flowing through the
    existing ``**kwargs`` mechanism.

    **Tunable parameter flow**: Subclasses declare tunable parameters as
    ``typing.Annotated`` class-body fields using constraint markers from
    :mod:`grdl.image_processing.params` (``Range``, ``Options``, ``Desc``).
    ``__init_subclass__`` collects these into ``__param_specs__`` and
    auto-generates an ``__init__`` (unless the subclass defines its own).
    At runtime, ``_resolve_params(kwargs)`` merges instance defaults with
    keyword-argument overrides and validates constraints.
    """

    # Track which classes have been checked to warn only once per class.
    _version_warned_classes: set = set()

    #: Whether this processor uses only numpy operations (True) or
    #: depends on scipy/other CPU-only libraries (False).  Used by
    #: GRDK's GpuBackend to skip futile GPU attempts.
    __gpu_compatible__: bool = False

    #: Tuple of :class:`~grdl.image_processing.params.ParamSpec` built
    #: automatically by ``__init_subclass__`` from ``Annotated`` fields.
    __param_specs__: Tuple[ParamSpec, ...] = ()

    # -----------------------------------------------------------------
    # Subclass hook: collect Annotated params & generate __init__
    # -----------------------------------------------------------------
    #: Tuple of method names decorated with ``@globalprocessor``.
    __global_callbacks__: Tuple[str, ...] = ()

    #: Whether this processor class has any global-pass callbacks.
    __has_global_pass__: bool = False

    @property
    def has_global_pass(self) -> bool:
        """Whether this processor requires a global pass before transforms.

        Returns ``True`` when at least one method is decorated with
        ``@globalprocessor``, meaning the executor must stream the full
        image through the global callbacks before running ``apply()``.

        Returns
        -------
        bool
        """
        return type(self).__has_global_pass__

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.__param_specs__ = collect_param_specs(cls)
        # Auto-generate __init__ only when the subclass has Annotated
        # params and did NOT define its own __init__.
        if cls.__param_specs__ and '__init__' not in cls.__dict__:
            cls.__init__ = _make_init(cls.__param_specs__)

        # Collect @globalprocessor-decorated methods
        new_callbacks = []
        for name in list(cls.__dict__):
            attr = cls.__dict__[name]
            if callable(attr) and getattr(attr, '__is_global_callback__', False):
                new_callbacks.append(name)
        parent_callbacks = getattr(
            super(cls, cls), '__global_callbacks__', ()
        )
        # Preserve order: parent callbacks first, then new ones (dedup)
        all_callbacks = tuple(
            dict.fromkeys((*parent_callbacks, *new_callbacks))
        )
        cls.__global_callbacks__ = all_callbacks
        cls.__has_global_pass__ = bool(all_callbacks)

    # -----------------------------------------------------------------
    # Version checking (fires once per class at first instantiation)
    # -----------------------------------------------------------------
    def __new__(cls, *args: Any, **kwargs: Any) -> 'ImageProcessor':
        if cls not in ImageProcessor._version_warned_classes:
            ImageProcessor._version_warned_classes.add(cls)
            if (
                not getattr(cls, '__processor_version__', None)
                and not getattr(cls, '__abstractmethods__', None)
            ):
                warnings.warn(
                    f"{cls.__qualname__} does not declare a processor version. "
                    f"Use @processor_version('x.y.z') to declare one.",
                    UserWarning,
                    stacklevel=2,
                )
        logger.debug("Instantiating %s", cls.__qualname__)
        return super().__new__(cls)

    # -----------------------------------------------------------------
    # Detection input specs (unchanged)
    # -----------------------------------------------------------------
    @property
    def detection_input_specs(self) -> Tuple['DetectionInputSpec', ...]:
        """
        Declare detection inputs this processor accepts.

        Override in subclasses that consume ``DetectionSet`` objects from
        upstream detectors.  Each spec describes a keyword argument name,
        whether it is required, and what it is used for.

        Returns
        -------
        Tuple[DetectionInputSpec, ...]
            Detection input declarations.  Default is an empty tuple
            (processor accepts no detection inputs).
        """
        return ()

    def _validate_detection_inputs(self, kwargs: Dict[str, Any]) -> None:
        """
        Check that required detection inputs are present in *kwargs*.

        Call this at the start of ``apply()`` or ``detect()`` in processors
        that declare detection inputs.

        Parameters
        ----------
        kwargs : Dict[str, Any]
            The keyword arguments passed to the processor method.

        Raises
        ------
        ValueError
            If a required detection input is missing from *kwargs*.
        """
        for spec in self.detection_input_specs:
            if spec.required and spec.name not in kwargs:
                raise ValueError(
                    f"Required detection input '{spec.name}' not provided. "
                    f"Description: {spec.description}"
                )

    def _get_detection_input(
        self, name: str, kwargs: Dict[str, Any]
    ) -> Optional['DetectionSet']:
        """
        Extract a named detection input from *kwargs*.

        Parameters
        ----------
        name : str
            The keyword argument name for the detection input.
        kwargs : Dict[str, Any]
            The keyword arguments passed to the processor method.

        Returns
        -------
        Optional[DetectionSet]
            The detection set if present, otherwise ``None``.
        """
        return kwargs.get(name)

    # -----------------------------------------------------------------
    # Tunable parameter resolution
    # -----------------------------------------------------------------
    def _resolve_params(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge instance defaults with runtime *kwargs* overrides.

        For each declared parameter in ``__param_specs__``:

        1. If present in *kwargs*, use the *kwargs* value.
        2. Otherwise use the instance attribute (``self.<name>``).

        Every resolved value is validated against its spec's type, range,
        and choices constraints.

        Parameters
        ----------
        kwargs : Dict[str, Any]
            Runtime keyword arguments.  May contain non-param keys
            (e.g. ``progress_callback``, detection inputs); those are
            ignored.

        Returns
        -------
        Dict[str, Any]
            ``{param_name: resolved_value}`` for every declared param.

        Raises
        ------
        TypeError
            If a value has the wrong type.
        ValueError
            If a value violates range or choices constraints.
        """
        resolved: Dict[str, Any] = {}
        for spec in type(self).__param_specs__:
            if spec.name in kwargs:
                value = kwargs[spec.name]
            else:
                value = getattr(self, spec.name)
            spec.validate(value)
            resolved[spec.name] = value
        return resolved

    # -----------------------------------------------------------------
    # Progress reporting (unchanged)
    # -----------------------------------------------------------------
    def _report_progress(
        self, kwargs: Dict[str, Any], fraction: float
    ) -> None:
        """Report progress to an optional callback.

        Call this from long-running ``apply()`` or ``detect()`` methods.
        If the caller provided a ``progress_callback`` keyword argument,
        it is called with the current fraction (0.0 to 1.0). If no
        callback is provided, this is a no-op.

        Parameters
        ----------
        kwargs : Dict[str, Any]
            The keyword arguments passed to the processor method.
        fraction : float
            Progress fraction in [0.0, 1.0].
        """
        cb = kwargs.get('progress_callback')
        if cb is not None:
            cb(float(fraction))


class ImageTransform(ImageProcessor):
    """
    Abstract base class for image transforms.

    Provides interface for transforms that take a source image array and
    produce a transformed output array. Covers both geometric transforms
    (orthorectification, reprojection) and radiometric transforms
    (filtering, enhancement).

    Subclasses implement ``apply`` which operates on numpy arrays.
    Detection inputs from upstream detectors can be passed through
    ``**kwargs`` -- see ``detection_input_specs`` on ``ImageProcessor``.
    """

    @abstractmethod
    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Apply the transform to a source image array.

        Parameters
        ----------
        source : np.ndarray
            Input image. Shape depends on the specific transform:
            (rows, cols) for single-band, (bands, rows, cols) or
            (rows, cols, bands) for multi-band.

        Returns
        -------
        np.ndarray
            Transformed image.
        """
        ...


class BandwiseTransformMixin:
    """Mixin that auto-applies a 2D transform across bands of a 3D stack.

    When mixed into an ``ImageTransform`` subclass, this overrides
    ``apply()`` to accept 3D ``(bands, rows, cols)`` arrays by
    iterating over the band axis and applying the subclass's 2D
    implementation to each band independently.

    2D inputs pass through unchanged. The mixin delegates to
    ``_apply_2d()``, which subclasses must implement (typically
    by renaming their existing ``apply()`` method or having it
    called by the mixin).

    Usage
    -----
    Subclasses should inherit from both the mixin and ``ImageTransform``::

        class MyFilter(BandwiseTransformMixin, ImageTransform):
            def _apply_2d(self, source, **kwargs):
                # 2D-only implementation
                ...

    When ``apply()`` is called with a ``(bands, rows, cols)`` array,
    the mixin calls ``_apply_2d()`` for each band and stacks results.
    """

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply the transform, handling both 2D and 3D inputs.

        Parameters
        ----------
        source : np.ndarray
            2D ``(rows, cols)`` or 3D ``(bands, rows, cols)`` array.

        Returns
        -------
        np.ndarray
            Transformed image with same dimensionality as input.
        """
        if source.ndim == 3:
            return np.stack(
                [self._apply_2d(source[b], **kwargs)
                 for b in range(source.shape[0])]
            )
        return self._apply_2d(source, **kwargs)

    @abstractmethod
    def _apply_2d(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply the transform to a single 2D band.

        Parameters
        ----------
        source : np.ndarray
            2D image array, shape ``(rows, cols)``.

        Returns
        -------
        np.ndarray
            Transformed 2D image.
        """
        ...
