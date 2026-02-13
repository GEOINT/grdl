# -*- coding: utf-8 -*-
"""
Collection Geometry - Coordinate systems and angles from CPHD metadata.

Computes reference vectors, per-pulse geometry (graze angle, incidence
angle, Doppler cone angle, twist, slope, azimuth, layover), image/slant
plane coordinate systems, and ARP polynomials required by the Polar
Format Algorithm and SICD metadata population.

Follows the NGA monostatic CPHD standard NGA.STND.0068-1_1.1.0,
section 6.5.1.

Dependencies
------------
sarpy (for ``wgs_84_norm``, ``ecf_to_geodetic``)
cartopy (optional, for ``plot()``)

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
2026-02-12

Modified
--------
2026-02-12
"""

# Standard library
from typing import Optional

# Third-party
import numpy as np
from numpy.linalg import norm

from sarpy.geometry.geocoords import wgs_84_norm, ecf_to_geodetic

# GRDL internal
from grdl.IO.models.cphd import CPHDMetadata, CPHDPVP


# Speed of light (m/s)
_C = 299792458.0


class CollectionGeometry:
    """Collection geometry from CPHD per-vector parameters.

    Computes coordinate systems, per-pulse angles, image/slant plane
    definitions, and ARP polynomials. All results are stored as
    instance attributes for downstream use by ``PolarGrid`` and
    ``PolarFormatAlgorithm``.

    Parameters
    ----------
    metadata : CPHDMetadata
        Typed CPHD metadata with populated PVP arrays.
    slant : bool
        If True (default), use slant plane. If False, use ground plane.

    Attributes
    ----------
    srp : np.ndarray
        Scene Reference Point ECF, shape ``(N, 3)``.
    srp_llh : np.ndarray
        SRP in geodetic (lat, lon, hae) degrees, shape ``(N, 3)``.
    arp : np.ndarray
        Antenna Reference Point ECF, shape ``(N, 3)``.
    varp : np.ndarray
        ARP velocity ECF, shape ``(N, 3)``.
    range_vec : np.ndarray
        Bistatic range vectors, shape ``(N, 3)``.
    phi : np.ndarray
        Polar angle per pulse (radians), shape ``(N,)``.
    k_sf : np.ndarray
        K-space scale factor per pulse, shape ``(N,)``.
    graz_ang : np.ndarray
        Grazing angle per pulse (radians).
    incd_ang : np.ndarray
        Incidence angle per pulse (radians).
    dop_cone_ang : np.ndarray
        Doppler cone angle per pulse (radians).
    twist_ang : np.ndarray
        Twist angle per pulse (radians).
    slope_ang : np.ndarray
        Slope angle per pulse (radians).
    azim_ang : np.ndarray
        Azimuth angle per pulse (radians).
    layover_ang : np.ndarray
        Layover angle per pulse (radians).
    side_of_track : str
        ``'L'`` or ``'R'``.
    coa_time : float
        Center-of-aperture time (seconds).
    tstart : float
        First pulse time.
    tend : float
        Last pulse time.
    image_plane : str
        ``'SLANT'`` or ``'GROUND'``.
    rg_uvect_ecf : np.ndarray
        Range unit vector in ECF at COA, shape ``(3,)``.
    az_uvect_ecf : np.ndarray
        Azimuth unit vector in ECF at COA, shape ``(3,)``.
    arp_poly_x, arp_poly_y, arp_poly_z : np.polynomial.Polynomial
        5th-order ARP position polynomials in time.
    c : float
        Speed of light (m/s).

    Examples
    --------
    >>> from grdl.IO.sar import CPHDReader
    >>> from grdl.image_processing.sar.image_formation import CollectionGeometry
    >>> with CPHDReader('data.cphd') as reader:
    ...     geo = CollectionGeometry(reader.metadata)
    ...     print(f"Graze angle at COA: {np.degrees(geo.graz_ang_coa):.2f} deg")
    """

    def __init__(
        self,
        metadata: CPHDMetadata,
        slant: bool = True,
    ) -> None:
        pvp = metadata.pvp
        if pvp is None:
            raise ValueError("CPHDMetadata must have populated PVP arrays")

        self._pvp = pvp
        self._metadata = metadata
        self.c = _C

        # Derive time array
        self.time = 0.5 * (pvp.tx_time + pvp.rcv_time)
        self.npulses = len(self.time)
        self.nsamples = metadata.cols

        # Frequency parameters
        self.fx1 = pvp.fx1
        self.fx2 = pvp.fx2
        self.fxss = pvp.scss if pvp.scss is not None else np.zeros(self.npulses)
        self.fx0 = pvp.sc0 if pvp.sc0 is not None else pvp.fx1

        # Build geometry
        self._build_reference_vectors(pvp)
        self._build_reference_geometry(pvp)
        self._build_coordinates(slant)

    # ------------------------------------------------------------------
    # Reference vectors
    # ------------------------------------------------------------------

    def _build_reference_vectors(self, pvp: CPHDPVP) -> None:
        """Compute SRP, ENU unit vectors."""
        self.srp = pvp.srp_pos.copy()
        self.srp_llh = ecf_to_geodetic(self.srp, ordering='latlong')
        self.srp_dec = norm(self.srp, axis=1).reshape(-1, 1)
        self.u_ec_scp = self.srp / self.srp_dec

        # ENU unit vectors at the SRP
        srp_rad = self.srp_llh[0].copy()
        srp_rad[0] = np.radians(srp_rad[0])
        srp_rad[1] = np.radians(srp_rad[1])

        self.u_east = np.array([
            -np.sin(srp_rad[1]),
            np.cos(srp_rad[1]),
            0.0,
        ])
        self.u_north = np.array([
            -np.sin(srp_rad[0]) * np.cos(srp_rad[1]),
            -np.sin(srp_rad[0]) * np.sin(srp_rad[1]),
            np.cos(srp_rad[0]),
        ])
        self.u_up = np.array([
            np.cos(srp_rad[0]) * np.cos(srp_rad[1]),
            np.cos(srp_rad[0]) * np.sin(srp_rad[1]),
            np.sin(srp_rad[0]),
        ])

    # ------------------------------------------------------------------
    # Reference geometry (per-pulse angles)
    # ------------------------------------------------------------------

    def _build_reference_geometry(self, pvp: CPHDPVP) -> None:
        """Compute per-pulse angles following NGA monostatic standard."""
        tx_range = pvp.tx_pos - pvp.srp_pos
        rcv_range = pvp.rcv_pos - pvp.srp_pos

        tx_mag = norm(tx_range, axis=1).reshape(-1, 1)
        rcv_mag = norm(rcv_range, axis=1).reshape(-1, 1)

        tx_unit = tx_range / tx_mag
        rcv_unit = rcv_range / rcv_mag

        bisector = tx_unit + rcv_unit
        bisector = bisector / norm(bisector, axis=1).reshape(-1, 1)

        self.range_vec = bisector * (tx_mag + rcv_mag) / 2
        u_range_vec = self.range_vec / norm(
            self.range_vec, axis=1,
        ).reshape(-1, 1)

        # ARP (monostatic: midpoint of tx and rcv)
        self.arp = np.mean([pvp.tx_pos, pvp.rcv_pos], axis=0)
        self.varp = np.mean([pvp.tx_vel, pvp.rcv_vel], axis=0)

        # ARP w.r.t. SRP
        r_arp_srp = norm(self.arp - pvp.srp_pos, axis=1).reshape(-1, 1)
        u_arp = (self.arp - pvp.srp_pos) / r_arp_srp

        # Range rate
        rdot_arp_srp = np.sum(u_arp * self.varp, axis=1)

        # ARP in ECF
        arp_dec = norm(self.arp, axis=1).reshape(-1, 1)
        u_ec_arp = self.arp / arp_dec

        # Earth angle ARP → SRP
        ea_arp = np.arccos(
            np.clip(np.sum(self.u_ec_scp * u_ec_arp, axis=1), -1, 1),
        )
        self.rg_arp_scp = np.hstack(self.srp_dec) * ea_arp

        # Velocity magnitude and unit
        varp_m = norm(self.varp, axis=1).reshape(-1, 1)
        u_varp = self.varp / varp_m

        # Side of track
        left = np.cross(u_ec_arp, u_varp)
        look = -1
        self.side_of_track = 'R'
        if np.sum(left[0] * u_arp[0]) < 0:
            look = 1
            self.side_of_track = 'L'

        # Doppler cone angle
        self.dop_cone_ang = np.arccos(
            np.clip(-rdot_arp_srp / np.hstack(varp_m), -1, 1),
        )

        # Ground plane
        u_gpz = wgs_84_norm(self.srp)
        u_gpy = np.cross(u_gpz, u_range_vec)
        u_gpy = u_gpy / norm(u_gpy, axis=1).reshape(-1, 1)
        u_gpx = np.cross(u_gpy, u_gpz)

        # Graze and incidence angles
        self.graz_ang = np.arccos(
            np.clip(np.sum(u_range_vec * u_gpx, axis=1), -1, 1),
        )
        self.incd_ang = np.pi / 2 - self.graz_ang

        # Azimuth angle
        gpx_n = np.dot(u_gpx, self.u_north)
        gpx_e = np.dot(u_gpx, self.u_east)
        self.azim_ang = np.arctan2(gpx_e, gpx_n)

        # Slant plane normal
        spn = look * np.cross(u_arp, u_varp)
        u_spn = spn / norm(spn, axis=1).reshape(-1, 1)

        # Twist and slope
        self.twist_ang = -np.arcsin(
            np.clip(np.sum(u_spn * u_gpy, axis=1), -1, 1),
        )
        self.slope_ang = np.arccos(
            np.clip(np.sum(u_gpz * u_spn, axis=1), -1, 1),
        )

        # Layover angle
        lodir_n = -np.dot(u_spn, self.u_north)
        lodir_e = -np.dot(u_spn, self.u_east)
        self.layover_ang = np.arctan2(lodir_e, lodir_n)

    # ------------------------------------------------------------------
    # Coordinate system and image plane
    # ------------------------------------------------------------------

    def _build_coordinates(self, slant: bool = True) -> None:
        """Build image plane coordinate system and COA parameters."""
        center_idx = self.npulses // 2
        self.coa_time = float(self.time[center_idx])
        self.coa_arp_vel = self.varp[center_idx] - self.srp[center_idx]
        self.range_vec_coa = self.range_vec[center_idx]
        self.slant_range_coa = float(norm(self.range_vec_coa))

        # Focal plane normal
        fpn = wgs_84_norm(self.srp)

        # Image plane normal
        if slant:
            ipn = np.cross(self.range_vec[0], self.range_vec[-1])
            self.ipn = ipn / norm(ipn)
            self.image_plane = 'SLANT'
        else:
            self.ipn = self.u_up.copy()
            self.image_plane = 'GROUND'

        # Project range vectors onto image plane
        self._compute_image_plane_geometry(
            self.range_vec,
            np.array([0.0, 0.0, 0.0]),
            self.range_vec_coa,
            self.ipn,
            fpn,
        )

        # COA values
        self.graz_ang_coa = float(self.graz_ang[center_idx])
        self.incd_ang_coa = float(self.incd_ang[center_idx])
        self.dop_cone_ang_coa = float(self.dop_cone_ang[center_idx])
        self.twist_ang_coa = float(self.twist_ang[center_idx])
        self.slope_ang_coa = float(self.slope_ang[center_idx])
        self.azim_ang_coa = float(self.azim_ang[center_idx])
        self.layover_ang_coa = float(self.layover_ang[center_idx])
        self.ground_range_coa = float(self.rg_arp_scp[center_idx])

        self.tstart = float(self.time[0])
        self.tend = float(self.time[-1])

        # Bandwidth
        self.bandwidth = float(abs(self.fx2[0] - self.fx1[0]))

    def _compute_image_plane_geometry(
        self,
        pv: np.ndarray,
        scp: np.ndarray,
        pv_coa: np.ndarray,
        ipn: np.ndarray,
        fpn: np.ndarray,
    ) -> None:
        """Compute polar angles, k-space scale, unit vectors, and ARP polys."""
        # Project onto image plane
        ip_pos = self._project(pv, fpn, scp, ipn)
        ip_coa_pos = self._project(pv_coa, fpn, scp, ipn)

        # Image plane axes
        ipx = ip_coa_pos - scp
        ipx = ipx / norm(ipx)
        ipy = np.cross(ipx, ipn)

        # Polar angles
        ip_range = ip_pos - scp
        self.phi = -np.arctan2(
            np.sum(ip_range * ipy, axis=1),
            np.sum(ip_range * ipx, axis=1),
        )

        # K-space scale factor
        range_vecs_unit = pv / norm(pv, axis=1).reshape(-1, 1)
        sin_graze = np.sum(range_vecs_unit * fpn, axis=1)
        ip_range_unit = ip_range / norm(ip_range, axis=1).reshape(-1, 1)
        sin_graze_ip = np.sum(ip_range_unit * fpn, axis=1)
        self.k_sf = np.sqrt(1 - sin_graze**2) / np.sqrt(1 - sin_graze_ip**2)

        self.theta = float(abs(self.phi[-1] - self.phi[0]))
        self.az_uvect_ecf = ipy[0].copy()
        self.rg_uvect_ecf = ipx[0].copy()

        # ARP polynomials (5th order)
        self.arp_poly_x = np.polynomial.Polynomial.fit(
            self.time, self.arp[:, 0], 5,
        )
        self.arp_poly_y = np.polynomial.Polynomial.fit(
            self.time, self.arp[:, 1], 5,
        )
        self.arp_poly_z = np.polynomial.Polynomial.fit(
            self.time, self.arp[:, 2], 5,
        )

    @staticmethod
    def _project(
        points: np.ndarray,
        line_dir: np.ndarray,
        plane_point: np.ndarray,
        plane_normal: np.ndarray,
    ) -> np.ndarray:
        """Project points along a direction onto a plane."""
        plane_normal = np.atleast_1d(plane_normal).ravel()
        dot = np.dot(line_dir, plane_normal)
        d = np.dot(plane_point - points, plane_normal) / dot
        if np.ndim(d) == 0:
            return points + d * line_dir
        return points + d.reshape(-1, 1) * line_dir

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot(self, ax=None) -> None:
        """Plot collection geometry on a map.

        Shows the SCP location and platform track on a Lambert
        Conformal projection with coastlines and borders. Requires
        ``matplotlib`` and ``cartopy``.

        Parameters
        ----------
        ax : cartopy.mpl.geoaxes.GeoAxes or None, optional
            GeoAxes to plot on. If None, creates a new figure with
            a Lambert Conformal projection centered on the SCP.
        """
        try:
            import matplotlib
            matplotlib.use("QtAgg")
            import matplotlib.pyplot as plt
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
        except ImportError:
            raise ImportError(
                "matplotlib and cartopy are required for plotting. "
                "Install with: conda install -c conda-forge cartopy"
            )

        lat0 = float(self.srp_llh[0, 0])
        lon0 = float(self.srp_llh[0, 1])

        proj = ccrs.LambertConformal(
            central_longitude=lon0, central_latitude=lat0,
        )

        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(1, 1, 1, projection=proj)

        # Map extent: ~1000 km around SCP
        extent_deg = 10.0
        ax.set_extent([
            lon0 - extent_deg, lon0 + extent_deg,
            lat0 - extent_deg, lat0 + extent_deg,
        ], crs=ccrs.PlateCarree())

        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='--')
        ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)

        # Plot SCP
        ax.plot(
            lon0, lat0, 'ro', markersize=10, transform=ccrs.PlateCarree(),
            label='SCP', zorder=5,
        )

        # Plot platform track (ARP positions converted to lat/lon)
        arp_llh = ecf_to_geodetic(self.arp, ordering='latlong')
        ax.plot(
            arp_llh[:, 1], arp_llh[:, 0], 'b-', linewidth=1.5,
            transform=ccrs.PlateCarree(), label='Platform track',
        )
        # Mark first and last pulse
        ax.plot(
            arp_llh[0, 1], arp_llh[0, 0], 'g^', markersize=8,
            transform=ccrs.PlateCarree(), label='First pulse',
        )
        ax.plot(
            arp_llh[-1, 1], arp_llh[-1, 0], 'ms', markersize=8,
            transform=ccrs.PlateCarree(), label='Last pulse',
        )

        ax.legend(loc='upper left', fontsize=9)
        ax.set_title('Collection Geometry')

        # Annotate with key angles at COA
        info = (
            f"Graze: {np.degrees(self.graz_ang_coa):.1f}\u00b0  "
            f"Azim: {np.degrees(self.azim_ang_coa):.1f}\u00b0  "
            f"Side: {self.side_of_track}  "
            f"SCP: ({lat0:.4f}, {lon0:.4f})"
        )
        ax.text(
            0.02, 0.02, info,
            transform=ax.transAxes, fontsize=9,
            color='white', backgroundcolor='black',
        )
