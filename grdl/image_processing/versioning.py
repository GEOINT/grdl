# -*- coding: utf-8 -*-
"""
Processor Versioning - Version decorator, detection input, and tunable parameter declarations.

Provides the ``@processor_version`` class decorator for stamping semantic
version strings on any image processor class (ImageTransform, ImageDetector,
PolarimetricDecomposition, etc.). Also provides ``DetectionInputSpec`` for
processors to declare what DetectionSet inputs they accept, and
``TunableParameterSpec`` for processors to declare runtime-adjustable
parameters with type checking and constraint validation.

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
from typing import Any, Optional, Sequence, Tuple, Type, TypeVar, Union, overload
import importlib.metadata

# GRDL vocabulary
from grdl.vocabulary import (
    DetectionType,
    ImageModality,
    ProcessorCategory,
    SegmentationType,
)

T = TypeVar('T')


@overload
def processor_version(version: str):
    ...

@overload
def processor_version():
    ...

def processor_version(version: Optional[str] = None):
    """Class decorator that stamps a processor version on an image processor.

    Applies to any image processor class (ImageTransform, ImageDetector,
    PolarimetricDecomposition, etc.). Sets ``__processor_version__`` as a
    class attribute. This version serves as the single source of truth for
    both the algorithm version and the output format version.

    If a version is not provided, it will be inferred from the package
    metadata. This is useful for "rapid" development processors where

    Parameters
    ----------
    version : str
        Semantic version string (e.g., ``'1.0.0'``).

    Returns
    -------
    Callable
        Class decorator that sets ``__processor_version__`` on the class.

    Examples
    --------
    >>> from grdl.image_processing.versioning import processor_version
    >>> from grdl.image_processing.base import ImageTransform
    >>>
    >>> @processor_version('1.0.0')
    ... class MyFilter(ImageTransform):
    ...     def apply(self, source, **kwargs):
    ...         return source
    >>>
    >>> MyFilter.__processor_version__
    '1.0.0'
    """
    def decorator(cls: Type[T]) -> Type[T]:
        if version:
            cls.__processor_version__ = version
        else:
            try:
                cls.__processor_version__ = importlib.metadata.version('grdl')
            except importlib.metadata.PackageNotFoundError:
                cls.__processor_version__ = "unknown"
        return cls
    return decorator


def processor_tags(
    modalities: Optional[Sequence[ImageModality]] = None,
    category: Optional[ProcessorCategory] = None,
    description: Optional[str] = None,
    detection_types: Optional[Sequence[DetectionType]] = None,
    segmentation_types: Optional[Sequence[SegmentationType]] = None,
):
    """Class decorator for processor capability metadata.

    Stamps ``__processor_tags__`` on the class with modality, category,
    and description metadata. Used by downstream tools (e.g., GRDK's
    OWProcessor widget) to filter and discover processors by capability.

    Parameters
    ----------
    modalities : Sequence[ImageModality], optional
        Imagery modalities this processor is designed for.
        Use members of :class:`~grdl.vocabulary.ImageModality`.
    category : ProcessorCategory, optional
        Processing category.
        Use a member of :class:`~grdl.vocabulary.ProcessorCategory`.
    description : str, optional
        Short human-readable description of the processor's purpose.
    detection_types : Sequence[DetectionType], optional
        Types of detection this processor performs. Applicable to
        ``ImageDetector`` subclasses.
    segmentation_types : Sequence[SegmentationType], optional
        Types of segmentation this processor produces. Applicable to
        segmentation processors.

    Raises
    ------
    TypeError
        If any element of *modalities* is not an ``ImageModality``, or
        *category* is not a ``ProcessorCategory``, or any element of
        *detection_types*/*segmentation_types* is not the correct enum.

    Examples
    --------
    >>> from grdl.vocabulary import ImageModality as IM, ProcessorCategory as PC
    >>> @processor_version('1.0.0')
    ... @processor_tags(modalities=[IM.SAR, IM.PAN], category=PC.FILTERS)
    ... class MyFilter(ImageTransform):
    ...     def apply(self, source, **kwargs):
    ...         return source
    >>> MyFilter.__processor_tags__['category']
    <ProcessorCategory.FILTERS: 'filters'>
    """
    # -- Validate enum types eagerly so typos fail at import time ----------
    if modalities is not None:
        for m in modalities:
            if not isinstance(m, ImageModality):
                raise TypeError(
                    f"modalities must be ImageModality members, got {m!r}"
                )
    if category is not None and not isinstance(category, ProcessorCategory):
        raise TypeError(
            f"category must be a ProcessorCategory member, got {category!r}"
        )
    if detection_types is not None:
        for d in detection_types:
            if not isinstance(d, DetectionType):
                raise TypeError(
                    f"detection_types must be DetectionType members, got {d!r}"
                )
    if segmentation_types is not None:
        for s in segmentation_types:
            if not isinstance(s, SegmentationType):
                raise TypeError(
                    f"segmentation_types must be SegmentationType members, "
                    f"got {s!r}"
                )

    def decorator(cls: Type[T]) -> Type[T]:
        cls.__processor_tags__ = {
            'modalities': tuple(modalities) if modalities else (),
            'category': category,
            'description': description,
            'detection_types': (
                tuple(detection_types) if detection_types else ()
            ),
            'segmentation_types': (
                tuple(segmentation_types) if segmentation_types else ()
            ),
        }
        return cls
    return decorator


class DetectionInputSpec:
    """Declaration of a detection input that a processor accepts.

    Processors that consume DetectionSet objects from upstream detectors
    declare their inputs via ``detection_input_specs``. Each spec describes
    a named keyword argument that the processor's ``apply()`` or ``detect()``
    method accepts.

    Parameters
    ----------
    name : str
        Keyword argument name used to pass the DetectionSet
        (e.g., ``'prior_detections'``).
    required : bool
        If True, the processor raises ``ValueError`` when this input
        is not provided.
    description : str
        Human-readable description of what this detection input is used for.

    Examples
    --------
    >>> spec = DetectionInputSpec(
    ...     name='prior_detections',
    ...     required=False,
    ...     description='Detections from a prior pass to bias filter weights',
    ... )
    >>> spec.name
    'prior_detections'
    """

    def __init__(
        self,
        name: str,
        required: bool,
        description: str,
    ) -> None:
        self.name = name
        self.required = required
        self.description = description

    def __repr__(self) -> str:
        return (
            f"DetectionInputSpec(name={self.name!r}, "
            f"required={self.required!r}, "
            f"description={self.description!r})"
        )


class _NoDefault:
    """Sentinel indicating a tunable parameter has no default value.

    Singleton so that ``isinstance(obj, _NoDefault)`` works reliably.
    Needed because ``None`` can be a valid default for an optional parameter.
    """

    _instance: Optional['_NoDefault'] = None

    def __new__(cls) -> '_NoDefault':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return '<NO_DEFAULT>'


_NO_DEFAULT = _NoDefault()


class TunableParameterSpec:
    """Declaration of a tunable parameter that a processor accepts.

    Processors that accept runtime-adjustable parameters declare them
    via ``tunable_parameter_specs``. Each spec describes a keyword
    argument that the processor's ``apply()``, ``detect()``, or
    ``decompose()`` method accepts through ``**kwargs``.

    Whether the parameter is required is determined by the ``default``
    argument: if no default is provided (i.e., ``default`` is the
    sentinel ``_NO_DEFAULT``), the parameter is required.

    Parameters
    ----------
    name : str
        Keyword argument name (e.g., ``'threshold'``).
    param_type : type
        Expected Python type (e.g., ``float``, ``int``, ``str``).
        For numeric parameters, ``int`` values are accepted when
        ``param_type`` is ``float``.
    default : Any, optional
        Default value when the kwarg is not provided. Omit to make
        the parameter required.
    description : str, optional
        Human-readable description of the parameter.
    min_value : int or float, optional
        Minimum allowed value (inclusive). Only for numeric types.
    max_value : int or float, optional
        Maximum allowed value (inclusive). Only for numeric types.
    choices : tuple, optional
        Allowed values. If set, the parameter value must be in this
        tuple. Mutually exclusive with ``min_value``/``max_value``.

    Raises
    ------
    ValueError
        If both range constraints (``min_value``/``max_value``) and
        ``choices`` are specified.

    Examples
    --------
    >>> spec = TunableParameterSpec(
    ...     name='threshold',
    ...     param_type=float,
    ...     default=0.5,
    ...     description='Detection confidence threshold',
    ...     min_value=0.0,
    ...     max_value=1.0,
    ... )
    >>> spec.required
    False
    """

    def __init__(
        self,
        name: str,
        param_type: type,
        default: Any = _NO_DEFAULT,
        description: str = '',
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        choices: Optional[Tuple] = None,
    ) -> None:
        if (min_value is not None or max_value is not None) and choices is not None:
            raise ValueError(
                f"Parameter '{name}': cannot specify both "
                f"range (min_value/max_value) and choices constraints."
            )
        self.name = name
        self.param_type = param_type
        self.default = default
        self.description = description
        self.min_value = min_value
        self.max_value = max_value
        self.choices = choices

    @property
    def required(self) -> bool:
        """Whether this parameter is required (has no default value).

        Returns
        -------
        bool
            True if no default value was specified.
        """
        return isinstance(self.default, _NoDefault)

    def __repr__(self) -> str:
        parts = (
            f"TunableParameterSpec(name={self.name!r}, "
            f"param_type={self.param_type.__name__}, "
            f"required={self.required!r}"
        )
        if not self.required:
            parts += f", default={self.default!r}"
        if self.min_value is not None:
            parts += f", min_value={self.min_value!r}"
        if self.max_value is not None:
            parts += f", max_value={self.max_value!r}"
        if self.choices is not None:
            parts += f", choices={self.choices!r}"
        return parts + ")"
