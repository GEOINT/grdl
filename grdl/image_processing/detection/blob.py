# -*- coding: utf-8 -*-
"""
BlobDetector - Connected-component blob detection from binary masks.

Converts a binary mask (output of any threshold- or segmentation-based
processor) into a :class:`~grdl.image_processing.detection.models.DetectionSet`
by labeling connected components and extracting convex-hull polygons,
area, and perimeter for each blob.

This processor completes the raster → binary_mask → detection_set chain
described in the GRDL workflow vision.

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
2026-04-20
"""

# Standard library
import logging
from typing import Any, Optional, Tuple, List

# Third-party
import numpy as np

# GRDL internal
from grdl.image_processing.detection.base import ImageDetector
from grdl.image_processing.detection.fields import Fields
from grdl.image_processing.detection.models import Detection, DetectionSet
from grdl.image_processing.versioning import processor_tags, processor_version
from grdl.vocabulary import DataPortType, ProcessorCategory

logger = logging.getLogger(__name__)


@processor_version('0.1.0')
@processor_tags(
    input_type=DataPortType.BINARY_MASK,
    output_type=DataPortType.DETECTION_SET,
    category=ProcessorCategory.FIND_MAXIMA,
    required_bands=1,
    description='Connected-component blob detection from a binary mask',
)
class BlobDetector(ImageDetector):
    """Detect blobs in a binary mask using connected-component labeling.

    Converts a boolean or 0/1 binary mask array into a
    :class:`~grdl.image_processing.detection.models.DetectionSet` by:

    1. Labeling connected components (scipy.ndimage.label, 4-connectivity
       with 8-connectivity available via ``connectivity=8``).
    2. Filtering components by ``min_area`` / ``max_area`` (pixel count).
    3. Computing the convex hull of each component's pixels as the
       detection geometry.
    4. Recording ``physical.area`` and ``physical.perimeter`` in each
       detection's properties.

    **Strict 2-D input policy**: ``source`` must be a 2-D array of shape
    ``(rows, cols)``.  No automatic band collapsing or shape normalization
    is performed.  If you have a multi-polarization stack (e.g., Sentinel-1
    VV+VH or NISAR quad-pol), apply a dimensionality-reduction step first
    — such as Dual-Pol H-Alpha decomposition, Pauli decomposition, or
    intensity averaging — to produce a single 2-D raster, then threshold it
    before calling this detector.

    The ``required_bands=1`` processor tag communicates this constraint to
    grdl-runtime so that the graph validator rejects multi-band inputs at
    workflow-construction time.

    Parameters
    ----------
    min_area : int
        Minimum blob area in pixels (inclusive).  Blobs smaller than
        this are discarded.  Default: 4.
    max_area : int or None
        Maximum blob area in pixels (inclusive).  ``None`` means no
        upper limit.  Default: None.
    connectivity : int
        Connected-component connectivity: 4 (cross-shaped) or 8
        (including diagonals).  Default: 4.
    """

    __param_specs__ = {
        'min_area': {
            'type': 'int',
            'default': 4,
            'min': 1,
            'max': 10_000,
            'description': 'Minimum blob area in pixels',
        },
        'max_area': {
            'type': 'int',
            'default': 0,          # 0 is treated as "no limit"
            'min': 0,
            'max': 1_000_000,
            'description': 'Maximum blob area in pixels (0 = no limit)',
        },
        'connectivity': {
            'type': 'int',
            'default': 4,
            'min': 4,
            'max': 8,
            'description': 'Connectivity: 4 (cross) or 8 (full)',
        },
    }

    def __init__(
        self,
        min_area: int = 4,
        max_area: Optional[int] = None,
        connectivity: int = 4,
    ) -> None:
        self.min_area = max(1, int(min_area))
        self.max_area = int(max_area) if max_area and max_area > 0 else None
        if connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8")
        self.connectivity = connectivity

    @property
    def output_fields(self) -> Tuple[str, ...]:
        return (
            Fields.physical.AREA,
            Fields.physical.PERIMETER,
        )

    def detect(
        self,
        source: np.ndarray,
        geolocation: Optional[Any] = None,
        valid_mask: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> DetectionSet:
        """Run blob detection on a binary mask.

        Parameters
        ----------
        source : np.ndarray
            Binary mask array, shape ``(rows, cols)`` **only**.
            Values > 0 are treated as foreground.

            This detector enforces a strict 2-D input policy.  No automatic
            band collapsing or shape normalization is performed, so sensor
            geometry (e.g., multi-swath Sentinel-1, multi-pol NISAR) is
            never silently misrepresented.

            If you need to pass multi-polarization data, first apply a
            dimensionality-reduction step (Dual-Pol H-Alpha decomposition,
            Pauli decomposition, intensity averaging, etc.) to obtain a
            single 2-D channel, then threshold it to produce a binary mask
            before calling this detector.
        geolocation : Geolocation, optional
            Not used by this detector (no geo-registration performed here).
            Detections carry pixel-space geometries only.
        valid_mask : np.ndarray, optional
            Boolean 2-D gate, same shape as ``source``.  Applied after
            binarization; pixels where the gate is ``False`` are excluded
            from connected-component labeling.

        Returns
        -------
        DetectionSet

        Raises
        ------
        ValueError
            If ``source`` is not exactly 2-D.
        """
        from scipy.ndimage import label as nd_label

        arr = np.asarray(source)

        # ------------------------------------------------------------------
        # Strict 2-D input policy — no automatic normalization.
        # Raise with actionable guidance so the caller knows exactly what
        # pre-processing step is missing.
        # ------------------------------------------------------------------
        if arr.ndim != 2:
            if arr.ndim == 3:
                n_bands = arr.shape[2] if arr.shape[0] > arr.shape[2] else arr.shape[0]
                raise ValueError(
                    f"BlobDetector requires a single-channel 2-D array "
                    f"(rows, cols), but received a {arr.ndim}-D array with "
                    f"shape {arr.shape}.\n\n"
                    f"Multi-band inputs are rejected without normalization to "
                    f"preserve sensor geometry and data provenance.  If your "
                    f"data is a multi-polarization stack (e.g., Sentinel-1 "
                    f"VV+VH, NISAR quad-pol), apply a dimensionality-reduction "
                    f"step first — for example:\n"
                    f"  • Dual-Pol H-Alpha decomposition (DualPolHAlpha)\n"
                    f"  • Pauli decomposition\n"
                    f"  • Single-band extraction (select one polarization)\n"
                    f"  • Coherence or intensity averaging\n"
                    f"then threshold the resulting 2-D raster to obtain a "
                    f"binary mask before passing it to BlobDetector."
                )
            raise ValueError(
                f"BlobDetector requires a 2-D array (rows, cols), "
                f"got {arr.ndim}-D array with shape {arr.shape}."
            )

        mask = arr > 0

        if valid_mask is not None:
            mask = mask & np.asarray(valid_mask, dtype=bool)

        # Connected-component labeling
        struct = (
            np.ones((3, 3), dtype=np.int8)
            if self.connectivity == 8
            else np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int8)
        )
        labeled, n_components = nd_label(mask, structure=struct)

        detections: List[Detection] = []
        for comp_id in range(1, n_components + 1):
            ys, xs = np.where(labeled == comp_id)
            area = int(len(ys))
            if area < self.min_area:
                continue
            if self.max_area is not None and area > self.max_area:
                continue

            pixel_geometry = _hull_polygon(xs, ys)
            perimeter = _approx_perimeter(xs, ys)

            detections.append(
                Detection(
                    pixel_geometry=pixel_geometry,
                    properties={
                        Fields.physical.AREA: area,
                        Fields.physical.PERIMETER: round(perimeter, 2),
                    },
                )
            )

        logger.debug(
            "BlobDetector: %d blobs (from %d components, min_area=%d)",
            len(detections), n_components, self.min_area,
        )

        return DetectionSet(
            detections=detections,
            detector_name=self.__class__.__name__,
            detector_version=self.__processor_version__,
            output_fields=self.output_fields,
            metadata={
                'min_area': self.min_area,
                'max_area': self.max_area,
                'connectivity': self.connectivity,
                'n_components_raw': n_components,
            },
        )


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _hull_polygon(xs: np.ndarray, ys: np.ndarray):
    """Return a shapely Polygon for the convex hull of pixel coordinates."""
    from shapely.geometry import MultiPoint, Point

    if len(xs) == 0:
        return Point(0, 0).buffer(0)
    pts = np.column_stack((xs.astype(float), ys.astype(float)))
    if len(pts) == 1:
        from shapely.geometry import Point as SPoint
        return SPoint(float(pts[0, 0]), float(pts[0, 1])).buffer(0.5)
    if len(pts) == 2:
        from shapely.geometry import LineString
        return LineString(pts).buffer(0.5)
    return MultiPoint(pts).convex_hull


def _approx_perimeter(xs: np.ndarray, ys: np.ndarray) -> float:
    """Estimate the perimeter of a blob via its convex hull."""
    try:
        hull = _hull_polygon(xs, ys)
        return float(hull.length)
    except Exception:
        return float(len(xs))
