# -*- coding: utf-8 -*-
"""
Detection cueing: confine detectors to a geographic region of interest.

:func:`cued_detect` is a convenience wrapper around
``ImageDetector.detect`` that rasterises a :class:`GeographicShape` into
a pixel-space boolean mask and passes it as the ``valid_mask`` kwarg.
Two modes:

- ``chip=False`` (default) -- full-image detection with the mask applied
  at candidate-labelling time. Background statistics (where applicable)
  are drawn from the full image, which keeps CFAR's adaptive noise
  floor physically meaningful.

- ``chip=True`` -- extract the shape's bounding-box chip first, run
  detection on the chip with a chip-local mask, then offset detections
  back to full-image coordinates. Cheap for small ROIs inside large
  images; trades adaptive background for compute.

Dependencies
------------
numpy

Author
------
Duane Smalley
170194430+DDSmalls@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-04-18

Modified
--------
2026-04-18
"""

# Standard library
from typing import TYPE_CHECKING, Tuple

# Third-party
import numpy as np

# GRDL internal
from grdl.shapes.base import GeographicShape


if TYPE_CHECKING:
    from grdl.geolocation.base import Geolocation
    from grdl.image_processing.detection.base import ImageDetector
    from grdl.image_processing.detection.models import DetectionSet


def cued_detect(
    detector: 'ImageDetector',
    source: np.ndarray,
    shape: GeographicShape,
    geolocation: 'Geolocation',
    chip: bool = False,
    n_initial: int = 128,
    pixel_tolerance: float = 0.5,
    **detect_kwargs,
) -> 'DetectionSet':
    """Run ``detector`` restricted to the interior of ``shape``.

    Parameters
    ----------
    detector : ImageDetector
        Any GRDL detector whose ``detect()`` method honours a
        ``valid_mask`` kwarg. Detectors that ignore the kwarg still
        return correct-within-image detections; the shape cueing is
        lost for those.
    source : np.ndarray
        Image array, shape ``(rows, cols)`` or ``(rows, cols, bands)``.
    shape : GeographicShape
        Region of interest.
    geolocation : Geolocation
        Geolocation for ``source``.
    chip : bool
        When False (default), run detection over the full image with
        the mask applied at candidate-labelling time. When True,
        extract the shape's bounding box, detect on the chip, and
        offset coordinates back to full-image frame.
    n_initial, pixel_tolerance : see :meth:`GeographicShape.to_pixels`.
    **detect_kwargs : forwarded to ``detector.detect``.

    Returns
    -------
    DetectionSet
    """
    image_shape = source.shape[:2]

    if not chip:
        mask = shape.rasterize(
            geolocation=geolocation,
            image_shape=image_shape,
            fill=True,
            outline=False,
            n_initial=n_initial,
            pixel_tolerance=pixel_tolerance,
        )
        return detector.detect(
            source=source,
            geolocation=geolocation,
            valid_mask=mask,
            **detect_kwargs,
        )

    # Chip path: extract bbox, detect local, offset back.
    pixels = shape.to_pixels(
        geolocation=geolocation,
        n_initial=n_initial,
        pixel_tolerance=pixel_tolerance,
    )
    row_lo, row_hi, col_lo, col_hi = _bbox_in_image(pixels, image_shape)
    if row_hi <= row_lo or col_hi <= col_lo:
        # Shape falls entirely outside the image -- return empty.
        from grdl.image_processing.detection.models import DetectionSet
        detector_name = type(detector).__name__
        version = getattr(detector, '__processor_version__', 'unknown')
        return DetectionSet(
            detections=[],
            detector_name=detector_name,
            detector_version=version,
            output_fields=tuple(detector.output_fields),
        )

    if source.ndim == 2:
        chip_src = source[row_lo:row_hi, col_lo:col_hi]
    else:
        chip_src = source[row_lo:row_hi, col_lo:col_hi, ...]
    chip_shape = chip_src.shape[:2]

    # Chip-local mask: shift the projected polygon into chip coordinates
    # and rasterise without re-running to_pixels (avoids second
    # ``latlon_to_image`` call).
    from grdl.shapes.rasterize import rasterize_polygon
    chip_pixels = pixels.copy()
    chip_pixels[:, 0] -= row_lo
    chip_pixels[:, 1] -= col_lo
    chip_mask = rasterize_polygon(
        pixels=chip_pixels,
        image_shape=chip_shape,
        fill=True,
        outline=False,
        outline_thickness=1,
        closed=shape.is_closed,
    )

    # Wrap the geolocation with a pixel offset so detections get
    # geographic coordinates consistent with the full image.
    try:
        from grdl.geolocation import ChipGeolocation
        chip_geo = ChipGeolocation(
            geolocation, row_offset=row_lo, col_offset=col_lo,
            shape=chip_shape,
        )
    except Exception:
        chip_geo = None

    chip_result = detector.detect(
        source=chip_src,
        geolocation=chip_geo,
        valid_mask=chip_mask,
        **detect_kwargs,
    )
    return _offset_detections(chip_result, row_offset=row_lo, col_offset=col_lo)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _bbox_in_image(
    pixels: np.ndarray,
    image_shape: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    """Return ``(row_lo, row_hi, col_lo, col_hi)`` clipped to image bounds."""
    rows_total, cols_total = image_shape
    row_lo = int(max(0, np.floor(np.nanmin(pixels[:, 0]))))
    row_hi = int(min(rows_total, np.ceil(np.nanmax(pixels[:, 0])) + 1))
    col_lo = int(max(0, np.floor(np.nanmin(pixels[:, 1]))))
    col_hi = int(min(cols_total, np.ceil(np.nanmax(pixels[:, 1])) + 1))
    return row_lo, row_hi, col_lo, col_hi


def _offset_detections(
    detset: 'DetectionSet',
    row_offset: int,
    col_offset: int,
) -> 'DetectionSet':
    """Return a DetectionSet with pixel geometries shifted by (row, col)."""
    from shapely.affinity import translate
    from grdl.image_processing.detection.models import (
        Detection, DetectionSet,
    )

    new_dets = []
    for d in detset.detections:
        pix_geom = d.pixel_geometry
        if pix_geom is not None:
            # shapely x=col, y=row
            pix_geom = translate(pix_geom, xoff=col_offset, yoff=row_offset)
        new_dets.append(Detection(
            pixel_geometry=pix_geom,
            properties=dict(d.properties) if d.properties else None,
            confidence=d.confidence,
            geo_geometry=d.geo_geometry,
        ))
    return DetectionSet(
        detections=new_dets,
        detector_name=detset.detector_name,
        detector_version=detset.detector_version,
        output_fields=detset.output_fields,
        metadata=dict(detset.metadata),
    )
