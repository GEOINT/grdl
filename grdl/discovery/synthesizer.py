# -*- coding: utf-8 -*-
"""
Discovery Synthesizer - Generate synthetic test imagery with metadata.

Creates geocoded GeoTIFF files with known affine transforms and
controllable spatial patterns for testing the scanner, catalog, and
visualization pipeline.

Dependencies
------------
rasterio (for GeoTIFF writing with CRS/transform)

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
2026-03-29

Modified
--------
2026-03-29
"""

# Standard library
from pathlib import Path
from typing import Optional

# Third-party
import numpy as np


class DataSynthesizer:
    """Generate synthetic test imagery with known metadata.

    Produces geocoded GeoTIFF files that ``open_any()`` can read
    back with full geolocation via affine transforms.
    """

    def synthesize_sar(
        self,
        output_path: str | Path,
        rows: int = 1024,
        cols: int = 1024,
        center_lat: float = 0.0,
        center_lon: float = 0.0,
        pixel_spacing: float = 1.0,
        noise_sigma: float = 0.1,
    ) -> Path:
        """Generate a synthetic complex SAR image.

        Parameters
        ----------
        output_path : str or Path
            Output GeoTIFF path.
        rows : int
            Number of rows.
        cols : int
            Number of columns.
        center_lat : float
            Center latitude in degrees.
        center_lon : float
            Center longitude in degrees.
        pixel_spacing : float
            Ground pixel spacing in meters.
        noise_sigma : float
            Noise standard deviation for complex Gaussian noise.

        Returns
        -------
        Path
            Path to the created file.
        """
        output_path = Path(output_path)

        # Generate complex Gaussian noise
        real = np.random.randn(rows, cols).astype(np.float32) * noise_sigma
        imag = np.random.randn(rows, cols).astype(np.float32) * noise_sigma
        data = real + 1j * imag

        # Add some point targets
        for r_frac, c_frac, amp in [
            (0.25, 0.25, 10.0),
            (0.25, 0.75, 10.0),
            (0.75, 0.5, 15.0),
            (0.5, 0.5, 20.0),
        ]:
            r, c = int(r_frac * rows), int(c_frac * cols)
            data[max(0, r-1):r+2, max(0, c-1):c+2] += amp

        transform, crs = self._build_geotransform(
            center_lat, center_lon, rows, cols, pixel_spacing,
        )

        self._write_geotiff(output_path, data, transform, crs)
        return output_path

    def synthesize_eo(
        self,
        output_path: str | Path,
        rows: int = 1024,
        cols: int = 1024,
        bands: int = 3,
        center_lat: float = 0.0,
        center_lon: float = 0.0,
        gsd: float = 0.5,
        pattern: str = 'checkerboard',
    ) -> Path:
        """Generate a synthetic EO image.

        Parameters
        ----------
        output_path : str or Path
            Output GeoTIFF path.
        rows : int
            Number of rows.
        cols : int
            Number of columns.
        bands : int
            Number of spectral bands.
        center_lat : float
            Center latitude in degrees.
        center_lon : float
            Center longitude in degrees.
        gsd : float
            Ground sample distance in meters.
        pattern : str
            Spatial pattern: ``'checkerboard'``, ``'gradient'``,
            ``'noise'``, ``'resolution_target'``.

        Returns
        -------
        Path
            Path to the created file.
        """
        output_path = Path(output_path)

        data = self._generate_pattern(rows, cols, bands, pattern)

        transform, crs = self._build_geotransform(
            center_lat, center_lon, rows, cols, gsd,
        )

        self._write_geotiff(output_path, data, transform, crs)
        return output_path

    def synthesize_multispectral(
        self,
        output_path: str | Path,
        rows: int = 512,
        cols: int = 512,
        bands: int = 13,
        center_lat: float = 0.0,
        center_lon: float = 0.0,
        gsd: float = 10.0,
    ) -> Path:
        """Generate a synthetic multispectral image.

        Parameters
        ----------
        output_path : str or Path
            Output GeoTIFF path.
        rows : int
            Number of rows.
        cols : int
            Number of columns.
        bands : int
            Number of spectral bands.
        center_lat : float
            Center latitude in degrees.
        center_lon : float
            Center longitude in degrees.
        gsd : float
            Ground sample distance in meters.

        Returns
        -------
        Path
            Path to the created file.
        """
        output_path = Path(output_path)

        # Each band gets a slightly different gradient + noise
        data = np.zeros((bands, rows, cols), dtype=np.float32)
        rr = np.linspace(0, 1, rows, dtype=np.float32)
        cc = np.linspace(0, 1, cols, dtype=np.float32)
        row_grid, col_grid = np.meshgrid(rr, cc, indexing='ij')

        for b in range(bands):
            phase = b * 2 * np.pi / bands
            data[b] = (
                0.5 * np.sin(row_grid * 4 * np.pi + phase)
                + 0.5 * np.cos(col_grid * 4 * np.pi + phase)
                + np.random.randn(rows, cols).astype(np.float32) * 0.05
            )

        transform, crs = self._build_geotransform(
            center_lat, center_lon, rows, cols, gsd,
        )

        self._write_geotiff(output_path, data, transform, crs)
        return output_path

    # ── Internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _build_geotransform(
        center_lat: float,
        center_lon: float,
        rows: int,
        cols: int,
        gsd_meters: float,
    ) -> tuple:
        """Build a rasterio-compatible Affine transform and CRS.

        Returns
        -------
        tuple
            ``(transform, crs)`` where transform is an Affine and crs
            is a rasterio CRS object.
        """
        from rasterio.transform import from_bounds
        from rasterio.crs import CRS

        from grdl.geolocation.coordinates import meters_per_degree
        mpd = meters_per_degree(center_lat)
        deg_per_pix_lat = gsd_meters / mpd[0]
        deg_per_pix_lon = gsd_meters / mpd[1]

        half_lat = deg_per_pix_lat * rows / 2
        half_lon = deg_per_pix_lon * cols / 2

        west = center_lon - half_lon
        east = center_lon + half_lon
        south = center_lat - half_lat
        north = center_lat + half_lat

        transform = from_bounds(west, south, east, north, cols, rows)
        crs = CRS.from_epsg(4326)

        return transform, crs

    @staticmethod
    def _write_geotiff(
        path: Path,
        data: np.ndarray,
        transform: object,
        crs: object,
    ) -> None:
        """Write array to GeoTIFF with geolocation."""
        import rasterio

        path.parent.mkdir(parents=True, exist_ok=True)

        if np.iscomplexobj(data):
            # Write magnitude for GeoTIFF (complex not supported)
            write_data = np.abs(data).astype(np.float32)
            if write_data.ndim == 2:
                write_data = write_data[np.newaxis]
            count = write_data.shape[0]
            dtype = 'float32'
        elif data.ndim == 2:
            write_data = data[np.newaxis].astype(np.float32)
            count = 1
            dtype = 'float32'
        elif data.ndim == 3:
            write_data = data.astype(np.float32)
            count = data.shape[0]
            dtype = 'float32'
        else:
            raise ValueError(f"Unexpected data shape: {data.shape}")

        height, width = write_data.shape[1], write_data.shape[2]

        with rasterio.open(
            str(path), 'w',
            driver='GTiff',
            height=height,
            width=width,
            count=count,
            dtype=dtype,
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(write_data)

    @staticmethod
    def _generate_pattern(
        rows: int,
        cols: int,
        bands: int,
        pattern: str,
    ) -> np.ndarray:
        """Generate a spatial pattern.

        Returns
        -------
        np.ndarray
            Shape ``(bands, rows, cols)`` float32 array.
        """
        data = np.zeros((bands, rows, cols), dtype=np.float32)

        if pattern == 'checkerboard':
            block = max(rows // 16, 1)
            rr = np.arange(rows) // block
            cc = np.arange(cols) // block
            checker = ((rr[:, None] + cc[None, :]) % 2).astype(np.float32)
            for b in range(bands):
                data[b] = checker * (0.5 + 0.5 * b / max(bands - 1, 1))

        elif pattern == 'gradient':
            rr = np.linspace(0, 1, rows, dtype=np.float32)
            cc = np.linspace(0, 1, cols, dtype=np.float32)
            for b in range(bands):
                angle = b * np.pi / bands
                data[b] = (
                    np.cos(angle) * rr[:, None]
                    + np.sin(angle) * cc[None, :]
                )

        elif pattern == 'noise':
            data = np.random.randn(bands, rows, cols).astype(np.float32)

        elif pattern == 'resolution_target':
            # Siemens star pattern
            cy, cx = rows // 2, cols // 2
            yy, xx = np.ogrid[-cy:rows-cy, -cx:cols-cx]
            theta = np.arctan2(yy, xx)
            r = np.sqrt(yy**2 + xx**2).astype(np.float32)
            for b in range(bands):
                spokes = 16 + b * 4
                star = (np.sin(theta * spokes) > 0).astype(np.float32)
                # Fade at center and edges
                fade = np.clip(r / max(rows, cols) * 4, 0, 1)
                data[b] = star * fade

        else:
            raise ValueError(
                f"Unknown pattern: {pattern!r}. "
                "Options: 'checkerboard', 'gradient', 'noise', 'resolution_target'"
            )

        return data
