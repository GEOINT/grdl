# -*- coding: utf-8 -*-
"""
Processor Versioning - Version decorator and detection input declarations.

Provides the ``@processor_version`` class decorator for stamping semantic
version strings on any image processor class (ImageTransform, ImageDetector,
PolarimetricDecomposition, etc.). Also provides ``DetectionInputSpec`` for
processors to declare what DetectionSet inputs they accept.

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
from typing import Optional, Sequence, Type, TypeVar, overload
import importlib.metadata

# GRDL vocabulary
from grdl.vocabulary import (
    DetectionType,
    ExecutionPhase,
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
    phases: Optional[Sequence[ExecutionPhase]] = None,
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
    phases : Sequence[ExecutionPhase], optional
        Pipeline execution phases this processor is compatible with.
        When not provided, defaults to an empty tuple (no phase
        restriction -- compatible with any phase).

    Raises
    ------
    TypeError
        If any element of *modalities* is not an ``ImageModality``, or
        *category* is not a ``ProcessorCategory``, or any element of
        *detection_types*/*segmentation_types*/*phases* is not the
        correct enum type.

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
    if phases is not None:
        for p in phases:
            if not isinstance(p, ExecutionPhase):
                raise TypeError(
                    f"phases must be ExecutionPhase members, got {p!r}"
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
            'phases': tuple(phases) if phases else (),
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


def globalprocessor(method):
    """Mark a method as a global-pass callback on an ImageProcessor subclass.

    Applied to **methods** on ``ImageProcessor`` subclasses.  At class
    definition time, ``ImageProcessor.__init_subclass__`` collects all
    methods decorated with ``@globalprocessor`` into
    ``cls.__global_callbacks__`` and sets ``cls.__has_global_pass__``.

    Parameters
    ----------
    method : callable
        The method to mark as a global-pass callback.

    Returns
    -------
    callable
        The same method with ``__is_global_callback__`` set to ``True``.

    Examples
    --------
    >>> from grdl.image_processing.versioning import globalprocessor
    >>> from grdl.image_processing.base import ImageTransform
    >>>
    >>> class MyProcessor(ImageTransform):
    ...     @globalprocessor
    ...     def compute_stats(self, source):
    ...         return source.mean()
    ...     def apply(self, source, **kwargs):
    ...         return source
    >>>
    >>> MyProcessor.__has_global_pass__
    True
    >>> MyProcessor.__global_callbacks__
    ('compute_stats',)
    """
    method.__is_global_callback__ = True
    return method


