# -*- coding: utf-8 -*-
"""
Error-budget overlay demo: combining independent uncertainty ellipses.

A detection is localised with three independent error sources:

- Pixel-localisation: a small isotropic ellipse.
- Sensor pointing: a larger ellipse aligned with the look direction.
- Georegistration: a modest ellipse aligned east-north.

The three sources are convolved via grdl.shapes.convolve_ellipses to
produce the total 1-sigma error ellipse. Both the individual inputs
(thin lines) and combined 1/2/3-sigma ellipses (thick lines) are
overlaid on the imagery.

Dependencies
------------
matplotlib
numpy
rasterio
shapely
scikit-image

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

import numpy as np
import matplotlib.pyplot as plt
from rasterio.transform import Affine

from grdl.geolocation.eo.affine import AffineGeolocation
from grdl.shapes import Ellipse, convolve_ellipses, overlay_shape


def main() -> None:
    shape = (512, 512)
    image = np.zeros(shape, dtype=np.float64)
    transform = Affine(1e-3, 0.0, -118.2, 0.0, -1e-3, 34.1)
    geo = AffineGeolocation(transform, shape, 'EPSG:4326')

    det_lat, det_lon = 34.05, -118.15
    sources = [
        Ellipse(det_lat, det_lon, 8.0, 8.0, rotation_deg=0.0),
        Ellipse(det_lat, det_lon, 40.0, 20.0, rotation_deg=30.0),
        Ellipse(det_lat, det_lon, 25.0, 15.0, rotation_deg=0.0),
    ]
    labels = ['Pixel localisation', 'Sensor pointing', 'Georegistration']
    colors = ['yellow', 'magenta', 'cyan']

    combined_1s = convolve_ellipses(sources)
    combined_2s = Ellipse(
        det_lat, det_lon,
        semi_major_m=combined_1s.semi_major_m * 2.0,
        semi_minor_m=combined_1s.semi_minor_m * 2.0,
        rotation_deg=combined_1s.rotation_deg,
    )
    combined_3s = Ellipse(
        det_lat, det_lon,
        semi_major_m=combined_1s.semi_major_m * 3.0,
        semi_minor_m=combined_1s.semi_minor_m * 3.0,
        rotation_deg=combined_1s.rotation_deg,
    )

    print(
        f"Combined 1-sigma: a={combined_1s.semi_major_m:.2f} m, "
        f"b={combined_1s.semi_minor_m:.2f} m, "
        f"rot={combined_1s.rotation_deg:.1f} deg"
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, cmap='gray', vmin=0, vmax=1)

    for src, label, color in zip(sources, labels, colors):
        overlay_shape(
            ax, src, geo, color=color, linewidth=1.0, label=label,
        )
    for sigma_shape, color, label in (
        (combined_1s, 'lime', '1-sigma total'),
        (combined_2s, 'orange', '2-sigma total'),
        (combined_3s, 'red', '3-sigma total'),
    ):
        overlay_shape(
            ax, sigma_shape, geo, color=color, linewidth=2.5, label=label,
        )

    ax.set_title('Error-budget overlay: sources + convolved total')
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
