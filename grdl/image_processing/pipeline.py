# -*- coding: utf-8 -*-
"""
Pipeline - Composable sequence of image transforms.

Chains multiple ``ImageTransform`` instances into a single callable
pipeline. The output of each transform feeds into the next. Supports
progress reporting across the full chain and preserves processor
version metadata.

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
import logging
from typing import Any, List, Optional, Sequence

# Third-party
import numpy as np

# GRDL internal
from grdl.image_processing.base import ImageTransform

logger = logging.getLogger(__name__)


class Pipeline(ImageTransform):
    """Sequential chain of image transforms.

    Applies a sequence of ``ImageTransform`` instances in order,
    passing the output of each as the input to the next. The
    pipeline itself is an ``ImageTransform``, so it can be nested
    inside other pipelines.

    Parameters
    ----------
    steps : Sequence[ImageTransform]
        Ordered list of transforms to apply. Must contain at least
        one transform.

    Examples
    --------
    >>> from grdl.imagej import GammaCorrection, UnsharpMask, EdgeDetector
    >>> from grdl.image_processing import Pipeline
    >>>
    >>> pipe = Pipeline([
    ...     GammaCorrection(gamma=0.5),
    ...     UnsharpMask(sigma=2.0, weight=0.6),
    ...     EdgeDetector(method='sobel'),
    ... ])
    >>> result = pipe.apply(image)

    With progress reporting:

    >>> result = pipe.apply(image, progress_callback=lambda f: print(f"{f:.0%}"))
    """

    __processor_version__ = '1.0.0'
    __gpu_compatible__ = False

    def __init__(self, steps: Sequence[ImageTransform]) -> None:
        if not steps:
            raise ValueError("Pipeline requires at least one transform")
        for i, step in enumerate(steps):
            if not isinstance(step, ImageTransform):
                raise TypeError(
                    f"Step {i} is not an ImageTransform: {type(step).__name__}"
                )
        self._steps: List[ImageTransform] = list(steps)

    @property
    def steps(self) -> List[ImageTransform]:
        """The ordered list of transforms in this pipeline.

        Returns
        -------
        List[ImageTransform]
            Shallow copy of the step list.
        """
        return list(self._steps)

    def __len__(self) -> int:
        return len(self._steps)

    def __repr__(self) -> str:
        step_names = [type(s).__name__ for s in self._steps]
        return f"Pipeline({step_names})"

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply all transforms in sequence.

        Parameters
        ----------
        source : np.ndarray
            Input image array.
        **kwargs
            Keyword arguments forwarded to each transform's ``apply()``.
            ``progress_callback`` is intercepted and rescaled so each
            step reports its proportional share of overall progress.

        Returns
        -------
        np.ndarray
            Output after all transforms have been applied.
        """
        n = len(self._steps)
        outer_cb = kwargs.pop('progress_callback', None)

        result = source
        for i, step in enumerate(self._steps):
            logger.debug("Pipeline step %d/%d: %s", i + 1, n,
                         type(step).__qualname__)

            step_kwargs = dict(kwargs)
            if outer_cb is not None:
                base = i / n
                scale = 1.0 / n
                step_kwargs['progress_callback'] = (
                    lambda f, _b=base, _s=scale: outer_cb(_b + f * _s)
                )

            result = step.apply(result, **step_kwargs)

            if outer_cb is not None:
                outer_cb((i + 1) / n)

        return result
