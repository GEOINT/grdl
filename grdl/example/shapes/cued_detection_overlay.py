# -*- coding: utf-8 -*-
"""
Cued-detection overlay demo.

Demonstrates the end-to-end grdl.shapes workflow on a synthetic image:

1. Build a basemap geolocation and a synthetic bright-target scene.
2. Draw a geographic Circle as the region of interest.
3. Run CACFARDetector via cued_detect -- detections only fire inside
   the circle even though there are bright targets elsewhere.
4. Overlay the circle and detections on the image via matplotlib.

No external imagery needed; the scene is generated in-process so the
example runs anywhere.

Dependencies
------------
matplotlib
numpy
rasterio
shapely
scikit-image

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
2026-04-18

Modified
--------
2026-04-18
"""

import numpy as np
import matplotlib.pyplot as plt
from rasterio.transform import Affine

from grdl.geolocation.eo.affine import AffineGeolocation
from grdl.image_processing.detection.cfar import CACFARDetector
from grdl.shapes import Circle, cued_detect, overlay_shape


def make_scene(shape=(512, 512)) -> np.ndarray:
    rng = np.random.default_rng(2026)
    img = rng.normal(10.0, 1.0, shape).astype(np.float64)
    # Target A: inside the ROI
    img[80:90, 80:90] = 200.0
    # Target B: outside the ROI
    img[380:390, 380:390] = 200.0
    return img


def main() -> None:
    image = make_scene()
    transform = Affine(1e-3, 0.0, -118.2, 0.0, -1e-3, 34.1)
    geo = AffineGeolocation(transform, image.shape, 'EPSG:4326')

    roi = Circle(
        center_lat=34.1 - 85 * 1e-3,
        center_lon=-118.2 + 85 * 1e-3,
        radius_m=15_000.0,
    )
    detector = CACFARDetector(
        guard_cells=2, training_cells=4, pfa=1e-3, min_pixels=4,
        assumption='gaussian',
    )
    detections = cued_detect(detector, image, roi, geo)
    print(f"{len(detections)} detections inside the cued region")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, cmap='gray', vmin=0, vmax=50)
    overlay_shape(ax, roi, geo, color='cyan', linewidth=2, label='ROI')

    for d in detections:
        if d.pixel_geometry is None:
            continue
        x, y = d.pixel_geometry.centroid.x, d.pixel_geometry.centroid.y
        ax.plot(x, y, 'r+', markersize=12, markeredgewidth=2)
    ax.set_title(
        f"Cued CA-CFAR: {len(detections)} detection(s) inside ROI"
    )
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
