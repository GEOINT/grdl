# -*- coding: utf-8 -*-
"""
Image Processing Base Classes - Abstract interfaces for image processors.

Defines the ``ImageProcessor`` common base class for all image processor types
(transforms, detectors, decompositions) and the ``ImageTransform`` ABC for
dense raster transforms. ``ImageProcessor`` provides version checking at
first instantiation and detection input declaration/validation so that any
processor can accept upstream ``DetectionSet`` objects through ``**kwargs``.

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
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

# Third-party
import numpy as np

if TYPE_CHECKING:
    from grdl.image_processing.versioning import DetectionInputSpec, TunableParameterSpec
    from grdl.image_processing.detection.models import DetectionSet


class ImageProcessor(ABC):
    """
    Common base class for all image processors.

    All processor types -- dense raster transforms (``ImageTransform``),
    sparse vector detectors (``ImageDetector``), and polarimetric
    decompositions (``PolarimetricDecomposition``) -- inherit from this
    class.

    Provides three cross-cutting capabilities:

    **Version checking**: Concrete subclasses that do not declare a processor
    version via ``@processor_version('x.y.z')`` will trigger a
    ``UserWarning`` at first instantiation. The check uses ``__new__``
    rather than ``__init_subclass__`` so that decorators have been applied
    by the time the check runs.

    **Detection input flow**: Any processor can declare that it accepts
    ``DetectionSet`` inputs from upstream detectors by overriding
    ``detection_input_specs``. Detection inputs are passed as keyword
    arguments to ``apply()`` or ``detect()``, flowing through the
    existing ``**kwargs`` mechanism.

    **Tunable parameter flow**: Any processor can declare runtime-adjustable
    parameters by overriding ``tunable_parameter_specs``. Tunable parameters
    are passed as keyword arguments to ``apply()``, ``detect()``, or
    ``decompose()``, flowing through the existing ``**kwargs`` mechanism.
    Parameters are validated for type, range, and allowed values.

    Notes
    -----
    This class intentionally has no ``__init__`` method so that existing
    subclasses (which may not call ``super().__init__()``) remain
    backward-compatible. Abstract classes cannot be instantiated, so
    the version warning only fires for concrete processors.
    """

    # Track which classes have been checked to warn only once per class.
    _version_warned_classes: set = set()

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
        return super().__new__(cls)

    @property
    def detection_input_specs(self) -> Tuple['DetectionInputSpec', ...]:
        """
        Declare detection inputs this processor accepts.

        Override in subclasses that consume ``DetectionSet`` objects from
        upstream detectors. Each spec describes a keyword argument name,
        whether it is required, and what it is used for.

        Returns
        -------
        Tuple[DetectionInputSpec, ...]
            Detection input declarations. Default is an empty tuple
            (processor accepts no detection inputs).
        """
        return ()

    def _validate_detection_inputs(self, kwargs: Dict[str, Any]) -> None:
        """
        Check that required detection inputs are present in kwargs.

        Call this at the start of ``apply()`` or ``detect()`` in processors
        that declare detection inputs.

        Parameters
        ----------
        kwargs : Dict[str, Any]
            The keyword arguments passed to the processor method.

        Raises
        ------
        ValueError
            If a required detection input is missing from kwargs.
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
        Extract a named detection input from kwargs.

        Parameters
        ----------
        name : str
            The keyword argument name for the detection input.
        kwargs : Dict[str, Any]
            The keyword arguments passed to the processor method.

        Returns
        -------
        Optional[DetectionSet]
            The detection set if present, otherwise None.
        """
        return kwargs.get(name)

    @property
    def tunable_parameter_specs(self) -> Tuple['TunableParameterSpec', ...]:
        """
        Declare tunable parameters this processor accepts.

        Override in subclasses that accept runtime-adjustable parameters
        through ``**kwargs``. Each spec describes a keyword argument name,
        expected type, default value, and optional constraints.

        Returns
        -------
        Tuple[TunableParameterSpec, ...]
            Tunable parameter declarations. Default is an empty tuple
            (processor accepts no tunable parameters).
        """
        return ()

    def _validate_tunable_parameters(self, kwargs: Dict[str, Any]) -> None:
        """
        Check tunable parameter types, required presence, and constraints.

        Call this at the start of ``apply()``, ``detect()``, or
        ``decompose()`` in processors that declare tunable parameters.

        For each declared tunable parameter:

        1. If required and missing from kwargs, raises ``ValueError``.
        2. If present, checks ``isinstance`` against the spec's
           ``param_type``. ``int`` is accepted for ``float`` parameters.
        3. If range constraints (``min_value``/``max_value``) are set,
           validates the value falls within bounds (inclusive).
        4. If ``choices`` is set, validates the value is in the tuple.

        Parameters
        ----------
        kwargs : Dict[str, Any]
            The keyword arguments passed to the processor method.

        Raises
        ------
        ValueError
            If a required parameter is missing, or a value violates
            range or choices constraints.
        TypeError
            If a value has the wrong type.
        """
        for spec in self.tunable_parameter_specs:
            if spec.name not in kwargs:
                if spec.required:
                    raise ValueError(
                        f"Required tunable parameter '{spec.name}' not provided. "
                        f"Description: {spec.description}"
                    )
                continue

            value = kwargs[spec.name]

            # Type check: allow int for float
            if spec.param_type is float:
                if not isinstance(value, (int, float)):
                    raise TypeError(
                        f"Tunable parameter '{spec.name}' must be "
                        f"{spec.param_type.__name__}, got {type(value).__name__}"
                    )
            else:
                if not isinstance(value, spec.param_type):
                    raise TypeError(
                        f"Tunable parameter '{spec.name}' must be "
                        f"{spec.param_type.__name__}, got {type(value).__name__}"
                    )

            # Range constraints
            if spec.min_value is not None and value < spec.min_value:
                raise ValueError(
                    f"Tunable parameter '{spec.name}' value {value!r} "
                    f"is below minimum {spec.min_value!r}"
                )
            if spec.max_value is not None and value > spec.max_value:
                raise ValueError(
                    f"Tunable parameter '{spec.name}' value {value!r} "
                    f"is above maximum {spec.max_value!r}"
                )

            # Choices constraint
            if spec.choices is not None and value not in spec.choices:
                raise ValueError(
                    f"Tunable parameter '{spec.name}' value {value!r} "
                    f"is not in allowed choices {spec.choices!r}"
                )

    def _get_tunable_parameter(
        self, name: str, kwargs: Dict[str, Any]
    ) -> Any:
        """
        Extract a tunable parameter from kwargs, falling back to its default.

        Looks up the parameter by name in kwargs. If not present, returns
        the default value from the matching ``TunableParameterSpec``. If
        no matching spec exists, returns ``None``.

        Parameters
        ----------
        name : str
            The keyword argument name for the tunable parameter.
        kwargs : Dict[str, Any]
            The keyword arguments passed to the processor method.

        Returns
        -------
        Any
            The parameter value if present in kwargs, the spec's default
            if not present and a default exists, or None if no matching
            spec is found.
        """
        if name in kwargs:
            return kwargs[name]
        for spec in self.tunable_parameter_specs:
            if spec.name == name:
                if not spec.required:
                    return spec.default
                return None
        return None


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
