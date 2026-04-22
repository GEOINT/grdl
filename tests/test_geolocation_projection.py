# -*- coding: utf-8 -*-
"""
Tests for grdl.geolocation.projection — R/Rdot projection engine.

Uses synthetic geometry to validate the core R/Rdot contour intersection
and round-trip image↔ground consistency without requiring real SAR data.

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
2026-03-16

Modified
--------
2026-03-16
"""

# Third-party
import numpy as np
import pytest

# GRDL
from grdl.IO.models.common import Poly1D, Poly2D, XYZ, XYZPoly, RowCol
from grdl.geolocation.coordinates import geodetic_to_ecef, ecef_to_geodetic
from grdl.geolocation.projection import (
    COAProjection,
    image_to_ground_hae,
    image_to_ground_plane,
    ground_to_image,
    wgs84_norm,
    _plane_projector,
    _rgazcomp_projector,
    _inca_projector,
)


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def synthetic_geometry():
    """Synthetic SAR collection geometry for testing.

    Models a side-looking sensor at ~600 km altitude with a realistic
    grazing angle (~45 deg).  The sensor flies along the Y axis and
    looks sideways toward a ground point offset in Z.
    """
    from grdl.geolocation.coordinates import WGS84_A

    # SCP: offset from sub-satellite point to create side-looking angle
    # Place SCP at ~45 deg latitude on the ellipsoid surface
    scp_lat, scp_lon = 0.0, 0.0
    scp_ecf = np.array([WGS84_A, 0.0, 0.0])

    # ARP at 600 km altitude, offset in Z to create ~45 deg graze
    # This puts the sensor north of nadir, looking south-sideways
    alt = 600e3
    arp_pos = np.array([WGS84_A + alt, 0.0, alt])

    # Velocity: ~7.5 km/s along Y axis
    arp_vel = np.array([0.0, 7500.0, 0.0])

    # Slant range
    delta = arp_pos - scp_ecf
    slant_range = np.linalg.norm(delta)

    # Range rate at SCP
    rdot_scp = np.dot(arp_vel, delta) / slant_range

    return {
        'scp_ecf': scp_ecf,
        'arp_pos': arp_pos,
        'arp_vel': arp_vel,
        'slant_range': slant_range,
        'rdot_scp': rdot_scp,
        'alt': alt,
    }


@pytest.fixture
def plane_coa(synthetic_geometry):
    """COAProjection with PLANE-type projector for synthetic geometry."""
    g = synthetic_geometry
    scp = g['scp_ecf']

    # Constant TimeCOAPoly: t_COA = 2.5 sec everywhere
    time_coa_poly = Poly2D(coefs=np.array([[2.5]]))

    # Constant ARP trajectory (stationary + linear velocity)
    # ARP(t) = arp_pos + arp_vel * (t - 2.5)
    arp_poly = XYZPoly(
        x=Poly1D(coefs=np.array([
            g['arp_pos'][0] - g['arp_vel'][0] * 2.5,
            g['arp_vel'][0],
        ])),
        y=Poly1D(coefs=np.array([
            g['arp_pos'][1] - g['arp_vel'][1] * 2.5,
            g['arp_vel'][1],
        ])),
        z=Poly1D(coefs=np.array([
            g['arp_pos'][2] - g['arp_vel'][2] * 2.5,
            g['arp_vel'][2],
        ])),
    )

    # Image grid: row = range (increasing away from sensor), col = azimuth
    # u_row points from ARP toward ground (range increasing = farther)
    u_row = (scp - g['arp_pos']) / np.linalg.norm(scp - g['arp_pos'])
    u_col = g['arp_vel'] / np.linalg.norm(g['arp_vel'])

    row_ss = 1.0  # 1 meter per pixel
    col_ss = 1.0

    projector = _plane_projector(
        scp_ecf=scp,
        u_row=u_row,
        u_col=u_col,
        row_ss=row_ss,
        col_ss=col_ss,
    )

    coa = COAProjection(
        time_coa_poly=time_coa_poly,
        arp_poly=arp_poly,
        method_projection=projector,
        scp_pixel=(500.0, 500.0),
        row_ss=row_ss,
        col_ss=col_ss,
    )

    return coa, g, u_row, u_col


# ── wgs84_norm ───────────────────────────────────────────────────────


class TestWGS84Norm:
    """Tests for ellipsoid normal computation."""

    def test_equator_prime_meridian(self):
        """Normal at equator/prime meridian should point along +X."""
        from grdl.geolocation.coordinates import WGS84_A
        ecf = np.array([WGS84_A, 0.0, 0.0])
        n = wgs84_norm(ecf)
        assert n[0] == pytest.approx(1.0, abs=1e-6)
        assert abs(n[1]) < 1e-10
        assert abs(n[2]) < 1e-10

    def test_north_pole(self):
        """Normal at north pole should point along +Z."""
        from grdl.geolocation.coordinates import WGS84_B
        ecf = np.array([0.0, 0.0, WGS84_B])
        n = wgs84_norm(ecf)
        assert abs(n[0]) < 1e-10
        assert abs(n[1]) < 1e-10
        assert n[2] == pytest.approx(1.0, abs=1e-6)

    def test_unit_length(self):
        """Normal should be a unit vector."""
        ecf = np.array([4e6, 3e6, 5e6])
        n = wgs84_norm(ecf)
        assert np.linalg.norm(n) == pytest.approx(1.0, abs=1e-12)

    def test_batch(self):
        """Should handle (N, 3) input."""
        from grdl.geolocation.coordinates import WGS84_A, WGS84_B
        ecf = np.array([
            [WGS84_A, 0.0, 0.0],
            [0.0, 0.0, WGS84_B],
        ])
        n = wgs84_norm(ecf)
        assert n.shape == (2, 3)
        np.testing.assert_allclose(
            np.linalg.norm(n, axis=-1), [1.0, 1.0], atol=1e-12)


# ── image_to_ground_plane ────────────────────────────────────────────


class TestImageToGroundPlane:
    """Tests for R/Rdot contour-plane intersection."""

    def test_scp_projects_to_scp(self, synthetic_geometry):
        """R/Rdot at SCP should project back to SCP on the ground plane."""
        g = synthetic_geometry
        scp = g['scp_ecf']
        u_z = wgs84_norm(scp)

        gpp = image_to_ground_plane(
            r=np.array([g['slant_range']]),
            rdot=np.array([g['rdot_scp']]),
            arp=g['arp_pos'].reshape(1, 3),
            varp=g['arp_vel'].reshape(1, 3),
            gref=scp,
            u_z=u_z,
        )

        # Should be close to SCP (within meters for this simple geometry)
        dist = np.linalg.norm(gpp[0] - scp)
        assert dist < 10.0, f"SCP projection error: {dist:.1f} m"

    def test_infeasible_returns_nan(self, synthetic_geometry):
        """When range is too small, result should be NaN."""
        g = synthetic_geometry
        scp = g['scp_ecf']
        u_z = wgs84_norm(scp)

        # Range much smaller than ARP altitude → infeasible
        gpp = image_to_ground_plane(
            r=np.array([1.0]),  # 1 meter range from 600 km altitude
            rdot=np.array([0.0]),
            arp=g['arp_pos'].reshape(1, 3),
            varp=g['arp_vel'].reshape(1, 3),
            gref=scp,
            u_z=u_z,
        )
        assert np.all(np.isnan(gpp[0]))

    def test_batch_consistency(self, synthetic_geometry):
        """Multiple points should project independently."""
        g = synthetic_geometry
        scp = g['scp_ecf']
        u_z = wgs84_norm(scp)
        n = 5
        r_arr = np.full(n, g['slant_range'])
        rdot_arr = np.full(n, g['rdot_scp'])
        arp_arr = np.tile(g['arp_pos'], (n, 1))
        varp_arr = np.tile(g['arp_vel'], (n, 1))

        gpp = image_to_ground_plane(
            r_arr, rdot_arr, arp_arr, varp_arr, scp, u_z)

        # All points should be approximately equal
        for i in range(1, n):
            np.testing.assert_allclose(gpp[i], gpp[0], atol=1e-6)


# ── COAProjection ────────────────────────────────────────────────────


class TestCOAProjection:
    """Tests for COAProjection hub."""

    def test_scp_returns_correct_range(self, plane_coa):
        """At SCP pixel, R should equal the slant range."""
        coa, g, _, _ = plane_coa
        im_points = np.array([[500.0, 500.0]])  # SCP pixel
        r, rdot, t_coa, arp, varp = coa.projection(im_points)

        assert r[0] == pytest.approx(g['slant_range'], rel=1e-6)
        assert t_coa[0] == pytest.approx(2.5)

    def test_arp_position_at_coa(self, plane_coa):
        """ARP position at COA should match synthetic geometry."""
        coa, g, _, _ = plane_coa
        im_points = np.array([[500.0, 500.0]])
        _, _, _, arp, varp = coa.projection(im_points)

        np.testing.assert_allclose(arp[0], g['arp_pos'], atol=1.0)
        np.testing.assert_allclose(varp[0], g['arp_vel'], atol=1.0)

    def test_range_increases_along_row(self, plane_coa):
        """Moving along row direction (range) should increase R."""
        coa, g, _, _ = plane_coa
        pts = np.array([
            [500.0, 500.0],
            [600.0, 500.0],  # 100 pixels further in range
        ])
        r, _, _, _, _ = coa.projection(pts)
        assert r[1] > r[0]

    def test_range_bias(self, plane_coa):
        """Range bias should add a constant offset."""
        coa_base, g, u_row, u_col = plane_coa
        bias = 5.0

        coa_biased = COAProjection(
            time_coa_poly=coa_base._time_coa_poly,
            arp_poly=coa_base._arp_poly,
            method_projection=coa_base._method_projection,
            scp_pixel=(500.0, 500.0),
            row_ss=1.0,
            col_ss=1.0,
            range_bias=bias,
        )

        pts = np.array([[500.0, 500.0]])
        r_base, _, _, _, _ = coa_base.projection(pts)
        r_biased, _, _, _, _ = coa_biased.projection(pts)

        assert r_biased[0] == pytest.approx(r_base[0] + bias, rel=1e-10)


# ── image_to_ground_hae ──────────────────────────────────────────────


class TestImageToGroundHAE:
    """Tests for iterative HAE projection."""

    def test_scp_projects_near_scp(self, plane_coa):
        """SCP pixel projected at HAE=0 should land near equator/PM."""
        coa, g, _, _ = plane_coa
        im_points = np.array([[500.0, 500.0]])
        gpp = image_to_ground_hae(coa, im_points, hae=0.0,
                                   scp_ecf=g['scp_ecf'],
                                   max_iter=15, tol=1.0)

        # Convert to lat/lon
        geo = ecef_to_geodetic(gpp)
        lats, lons, heights = geo[:, 0], geo[:, 1], geo[:, 2]

        # Should be near (0, 0) for our synthetic geometry
        assert abs(lats[0]) < 5.0, f"Latitude off: {lats[0]:.2f}"
        assert abs(lons[0]) < 5.0, f"Longitude off: {lons[0]:.2f}"
        # Height should be near 0 (within tolerance)
        assert abs(heights[0]) < 100.0, f"Height off: {heights[0]:.1f} m"

    def test_hae_offset_changes_position(self, plane_coa):
        """Projecting to different HAE should give different positions."""
        coa, g, _, _ = plane_coa
        im_points = np.array([[500.0, 500.0]])

        gpp_0 = image_to_ground_hae(coa, im_points, hae=0.0,
                                     scp_ecf=g['scp_ecf'],
                                     max_iter=15, tol=1.0)
        gpp_1000 = image_to_ground_hae(coa, im_points, hae=1000.0,
                                        scp_ecf=g['scp_ecf'],
                                        max_iter=15, tol=1.0)

        dist = np.linalg.norm(gpp_1000[0] - gpp_0[0])
        assert dist > 100.0, "HAE difference should produce offset"


# ── INCA projector ───────────────────────────────────────────────────


class TestINCAProjector:
    """Tests for INCA (RMA) formation-specific projector."""

    def test_scp_returns_r_ca_scp(self):
        """At SCP (row_t=0, col_t=0), range should be R_CA_SCP."""
        r_ca_scp = 800e3
        time_ca_poly = Poly1D(coefs=np.array([2.5]))  # constant TCA
        d_rate_sf_poly = Poly2D(coefs=np.array([[1.0]]))  # DRSF = 1

        proj = _inca_projector(time_ca_poly, r_ca_scp, d_rate_sf_poly)

        # At SCP: row_t=0, col_t=0, time_coa=2.5 (= TCA), so dt=0
        # R = sqrt(R_CA² + 0) = R_CA
        arp = np.array([[6.978e6, 0.0, 0.0]])
        varp = np.array([[0.0, 7500.0, 0.0]])

        r, rdot = proj(
            np.array([0.0]),
            np.array([0.0]),
            np.array([2.5]),
            arp,
            varp,
        )
        assert r[0] == pytest.approx(r_ca_scp, rel=1e-10)
        assert rdot[0] == pytest.approx(0.0, abs=1e-6)


# ── RgAzComp projector ──────────────────────────────────────────────


class TestRgAzCompProjector:
    """Tests for RgAzComp formation-specific projector."""

    def test_scp_returns_slant_range(self):
        """At SCP (row_t=0, col_t=0), R and Rdot offsets are zero."""
        from grdl.geolocation.coordinates import WGS84_A
        scp_ecf = np.array([WGS84_A, 0.0, 0.0])
        arp = np.array([[WGS84_A + 600e3, 0.0, 0.0]])
        varp = np.array([[0.0, 7500.0, 0.0]])

        proj = _rgazcomp_projector(scp_ecf, az_sf=0.5)

        r, rdot = proj(
            np.array([0.0]),
            np.array([0.0]),
            np.array([2.5]),
            arp,
            varp,
        )
        expected_r = np.linalg.norm(arp[0] - scp_ecf)
        assert r[0] == pytest.approx(expected_r, rel=1e-10)

    def test_azimuth_offset_changes_rdot(self):
        """Non-zero col_t should change Rdot via azimuth scale factor."""
        from grdl.geolocation.coordinates import WGS84_A
        scp_ecf = np.array([WGS84_A, 0.0, 0.0])
        arp = np.array([[WGS84_A + 600e3, 0.0, 0.0]])
        varp = np.array([[0.0, 7500.0, 0.0]])

        proj = _rgazcomp_projector(scp_ecf, az_sf=0.5)

        _, rdot_0 = proj(
            np.array([0.0]), np.array([0.0]),
            np.array([2.5]), arp, varp)
        _, rdot_1 = proj(
            np.array([0.0]), np.array([100.0]),
            np.array([2.5]), arp, varp)

        assert rdot_0[0] != rdot_1[0]


# ── Round-trip: ground_to_image ↔ image_to_ground ────────────────────


class TestGroundToImage:
    """Tests for inverse projection (ground → image)."""

    def test_round_trip_scp(self, plane_coa):
        """SCP pixel → ground → image should return SCP pixel."""
        coa, g, u_row, u_col = plane_coa
        scp_pixel = (500.0, 500.0)

        # Forward: image → ground
        im_pts = np.array([[500.0, 500.0]])
        gpp = image_to_ground_hae(coa, im_pts, hae=0.0,
                                   scp_ecf=g['scp_ecf'],
                                   max_iter=15, tol=1.0)

        # Inverse: ground → image
        im_back = ground_to_image(
            coa, gpp, g['scp_ecf'], u_row, u_col,
            row_ss=1.0, col_ss=1.0,
            scp_pixel=scp_pixel,
            max_iter=15,
            tol=1.0,
        )

        # Should be close to original pixel
        assert im_back[0, 0] == pytest.approx(500.0, abs=5.0)
        assert im_back[0, 1] == pytest.approx(500.0, abs=5.0)
