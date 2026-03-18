# -*- coding: utf-8 -*-
"""
Tests for grdl.IO.models.common polynomial and coordinate primitives.

Verifies evaluation, differentiation, vector arithmetic, and edge cases
for Poly1D, Poly2D, XYZPoly, and XYZ.

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
from grdl.IO.models.common import Poly1D, Poly2D, XYZ, XYZPoly


# ── XYZ ──────────────────────────────────────────────────────────────


class TestXYZ:
    """Tests for XYZ vector operations."""

    def test_to_array(self):
        v = XYZ(1.0, 2.0, 3.0)
        arr = v.to_array()
        np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])
        assert arr.dtype == np.float64

    def test_from_array(self):
        v = XYZ.from_array([4.0, 5.0, 6.0])
        assert v.x == 4.0
        assert v.y == 5.0
        assert v.z == 6.0

    def test_from_array_wrong_size(self):
        with pytest.raises(ValueError, match="Expected 3"):
            XYZ.from_array([1.0, 2.0])

    def test_norm(self):
        v = XYZ(3.0, 4.0, 0.0)
        assert v.norm() == pytest.approx(5.0)

    def test_dot(self):
        a = XYZ(1.0, 0.0, 0.0)
        b = XYZ(0.0, 1.0, 0.0)
        assert a.dot(b) == pytest.approx(0.0)
        assert a.dot(a) == pytest.approx(1.0)

    def test_cross(self):
        x = XYZ(1.0, 0.0, 0.0)
        y = XYZ(0.0, 1.0, 0.0)
        z = x.cross(y)
        assert z.x == pytest.approx(0.0)
        assert z.y == pytest.approx(0.0)
        assert z.z == pytest.approx(1.0)

    def test_unit(self):
        v = XYZ(0.0, 0.0, 5.0)
        u = v.unit()
        assert u.z == pytest.approx(1.0)
        assert u.norm() == pytest.approx(1.0)

    def test_unit_zero_raises(self):
        with pytest.raises(ValueError, match="zero-length"):
            XYZ(0.0, 0.0, 0.0).unit()

    def test_arithmetic(self):
        a = XYZ(1.0, 2.0, 3.0)
        b = XYZ(4.0, 5.0, 6.0)
        s = a + b
        assert (s.x, s.y, s.z) == (5.0, 7.0, 9.0)
        d = b - a
        assert (d.x, d.y, d.z) == (3.0, 3.0, 3.0)

    def test_scalar_multiply(self):
        v = XYZ(1.0, 2.0, 3.0)
        r = v * 2.0
        assert (r.x, r.y, r.z) == (2.0, 4.0, 6.0)
        r2 = 3.0 * v
        assert (r2.x, r2.y, r2.z) == (3.0, 6.0, 9.0)

    def test_negate(self):
        v = XYZ(1.0, -2.0, 3.0)
        n = -v
        assert (n.x, n.y, n.z) == (-1.0, 2.0, -3.0)


# ── Poly1D ───────────────────────────────────────────────────────────


class TestPoly1D:
    """Tests for 1D polynomial evaluation and differentiation."""

    def test_constant(self):
        p = Poly1D(coefs=np.array([5.0]))
        assert p(0.0) == pytest.approx(5.0)
        assert p(100.0) == pytest.approx(5.0)

    def test_linear(self):
        # P(x) = 2 + 3x
        p = Poly1D(coefs=np.array([2.0, 3.0]))
        assert p(0.0) == pytest.approx(2.0)
        assert p(1.0) == pytest.approx(5.0)
        assert p(-1.0) == pytest.approx(-1.0)

    def test_quadratic(self):
        # P(x) = 1 + 0x + 2x^2
        p = Poly1D(coefs=np.array([1.0, 0.0, 2.0]))
        assert p(0.0) == pytest.approx(1.0)
        assert p(3.0) == pytest.approx(19.0)

    def test_array_eval(self):
        p = Poly1D(coefs=np.array([1.0, 2.0]))
        x = np.array([0.0, 1.0, 2.0])
        result = p(x)
        np.testing.assert_allclose(result, [1.0, 3.0, 5.0])

    def test_derivative_linear(self):
        # P(x) = 2 + 3x  →  P'(x) = 3
        p = Poly1D(coefs=np.array([2.0, 3.0]))
        dp = p.derivative()
        assert dp(0.0) == pytest.approx(3.0)
        assert dp(99.0) == pytest.approx(3.0)

    def test_derivative_quadratic(self):
        # P(x) = 1 + 2x + 3x^2  →  P'(x) = 2 + 6x
        p = Poly1D(coefs=np.array([1.0, 2.0, 3.0]))
        dp = p.derivative()
        assert dp(0.0) == pytest.approx(2.0)
        assert dp(1.0) == pytest.approx(8.0)

    def test_second_derivative(self):
        # P(x) = 1 + 2x + 3x^2  →  P''(x) = 6
        p = Poly1D(coefs=np.array([1.0, 2.0, 3.0]))
        ddp = p.derivative(2)
        assert ddp(0.0) == pytest.approx(6.0)

    def test_derivative_beyond_order(self):
        # P(x) = 5  →  P'(x) = 0
        p = Poly1D(coefs=np.array([5.0]))
        dp = p.derivative()
        assert dp(99.0) == pytest.approx(0.0)

    def test_derivative_eval_shortcut(self):
        p = Poly1D(coefs=np.array([1.0, 2.0, 3.0]))
        assert p.derivative_eval(1.0) == pytest.approx(8.0)

    def test_order_property(self):
        assert Poly1D(coefs=np.array([1.0])).order == 0
        assert Poly1D(coefs=np.array([1.0, 2.0, 3.0])).order == 2
        assert Poly1D().order == -1

    def test_no_coefs_raises(self):
        p = Poly1D()
        with pytest.raises(ValueError, match="not set"):
            p(1.0)

    def test_derivative_no_coefs_raises(self):
        with pytest.raises(ValueError, match="not set"):
            Poly1D().derivative()


# ── Poly2D ───────────────────────────────────────────────────────────


class TestPoly2D:
    """Tests for 2D polynomial evaluation and differentiation."""

    def test_constant(self):
        p = Poly2D(coefs=np.array([[7.0]]))
        assert p(0.0, 0.0) == pytest.approx(7.0)
        assert p(5.0, 3.0) == pytest.approx(7.0)

    def test_bilinear(self):
        # P(x, y) = 1 + 2y + 3x + 4xy
        # coefs[i, j] = coef of x^i * y^j
        c = np.array([[1.0, 2.0],
                       [3.0, 4.0]])
        p = Poly2D(coefs=c)
        # P(0, 0) = 1
        assert p(0.0, 0.0) == pytest.approx(1.0)
        # P(1, 0) = 1 + 3 = 4
        assert p(1.0, 0.0) == pytest.approx(4.0)
        # P(0, 1) = 1 + 2 = 3
        assert p(0.0, 1.0) == pytest.approx(3.0)
        # P(1, 1) = 1 + 2 + 3 + 4 = 10
        assert p(1.0, 1.0) == pytest.approx(10.0)

    def test_quadratic_in_x(self):
        # P(x, y) = x^2  →  coefs = [[0], [0], [1]]
        c = np.array([[0.0], [0.0], [1.0]])
        p = Poly2D(coefs=c)
        assert p(3.0, 0.0) == pytest.approx(9.0)
        assert p(3.0, 99.0) == pytest.approx(9.0)

    def test_array_eval(self):
        # P(x, y) = x + y
        c = np.array([[0.0, 1.0],
                       [1.0, 0.0]])
        p = Poly2D(coefs=c)
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([10.0, 20.0, 30.0])
        np.testing.assert_allclose(p(x, y), [11.0, 22.0, 33.0])

    def test_derivative_dx(self):
        # P(x, y) = 1 + 2y + 3x + 4xy
        # dP/dx = 3 + 4y
        c = np.array([[1.0, 2.0], [3.0, 4.0]])
        p = Poly2D(coefs=c)
        dp_dx = p.derivative(dx=1)
        assert dp_dx(0.0, 0.0) == pytest.approx(3.0)
        assert dp_dx(0.0, 1.0) == pytest.approx(7.0)

    def test_derivative_dy(self):
        # P(x, y) = 1 + 2y + 3x + 4xy
        # dP/dy = 2 + 4x
        c = np.array([[1.0, 2.0], [3.0, 4.0]])
        p = Poly2D(coefs=c)
        dp_dy = p.derivative(dy=1)
        assert dp_dy(0.0, 0.0) == pytest.approx(2.0)
        assert dp_dy(1.0, 0.0) == pytest.approx(6.0)

    def test_mixed_derivative(self):
        # P(x, y) = 1 + 2y + 3x + 4xy
        # d^2P/dxdy = 4
        c = np.array([[1.0, 2.0], [3.0, 4.0]])
        p = Poly2D(coefs=c)
        d2 = p.derivative(dx=1, dy=1)
        assert d2(0.0, 0.0) == pytest.approx(4.0)

    def test_derivative_eval_shortcut(self):
        c = np.array([[1.0, 2.0], [3.0, 4.0]])
        p = Poly2D(coefs=c)
        assert p.derivative_eval(1.0, 1.0, dx=1) == pytest.approx(7.0)

    def test_order_property(self):
        c = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        p = Poly2D(coefs=c)
        assert p.order == (2, 1)

    def test_no_coefs_raises(self):
        with pytest.raises(ValueError, match="not set"):
            Poly2D()(1.0, 2.0)

    def test_timecoapoly_pattern(self):
        """Verify Poly2D works for the SICD TimeCOAPoly use case.

        TimeCOAPoly maps (row_transform, col_transform) → COA time.
        Typical 1st-order: t_COA = c00 + c10*x + c01*y + c11*x*y.
        """
        # Simulated TimeCOAPoly: t_COA ≈ 2.5 + 0.001*row - 0.0005*col
        c = np.array([[2.5, -0.0005],
                       [0.001, 0.0]])
        tcoa = Poly2D(coefs=c)
        # At SCP (row=0, col=0): t_COA = 2.5
        assert tcoa(0.0, 0.0) == pytest.approx(2.5)
        # At row=100, col=200: t_COA = 2.5 + 0.1 - 0.1 = 2.5
        assert tcoa(100.0, 200.0) == pytest.approx(2.5)


# ── XYZPoly ──────────────────────────────────────────────────────────


class TestXYZPoly:
    """Tests for XYZPoly position polynomial."""

    @pytest.fixture
    def arp_poly(self):
        """Simulated ARP trajectory: constant velocity along X."""
        # X(t) = 1e6 + 100*t
        # Y(t) = 2e6 + 50*t
        # Z(t) = 7e6
        return XYZPoly(
            x=Poly1D(coefs=np.array([1e6, 100.0])),
            y=Poly1D(coefs=np.array([2e6, 50.0])),
            z=Poly1D(coefs=np.array([7e6])),
        )

    def test_eval_scalar(self, arp_poly):
        pos = arp_poly(0.0)
        assert pos.shape == (3,)
        np.testing.assert_allclose(pos, [1e6, 2e6, 7e6])

    def test_eval_scalar_nonzero_time(self, arp_poly):
        pos = arp_poly(10.0)
        np.testing.assert_allclose(pos, [1e6 + 1000.0, 2e6 + 500.0, 7e6])

    def test_eval_array(self, arp_poly):
        t = np.array([0.0, 1.0, 2.0])
        pos = arp_poly(t)
        assert pos.shape == (3, 3)
        np.testing.assert_allclose(pos[0], [1e6, 2e6, 7e6])
        np.testing.assert_allclose(pos[2], [1e6 + 200.0, 2e6 + 100.0, 7e6])

    def test_derivative_velocity(self, arp_poly):
        vel_poly = arp_poly.derivative(1)
        vel = vel_poly(0.0)
        np.testing.assert_allclose(vel, [100.0, 50.0, 0.0])

    def test_derivative_acceleration(self, arp_poly):
        acc_poly = arp_poly.derivative(2)
        acc = acc_poly(0.0)
        np.testing.assert_allclose(acc, [0.0, 0.0, 0.0])

    def test_derivative_eval_shortcut(self, arp_poly):
        vel = arp_poly.derivative_eval(5.0, n=1)
        np.testing.assert_allclose(vel, [100.0, 50.0, 0.0])

    def test_missing_component_raises(self):
        p = XYZPoly(x=Poly1D(coefs=np.array([1.0])))
        with pytest.raises(ValueError, match="must be set"):
            p(0.0)

    def test_cubic_trajectory(self):
        """Verify cubic ARPPoly (typical real-world order)."""
        # X(t) = 1e6 + 100*t + 0.01*t^2 + 1e-5*t^3
        x_poly = Poly1D(coefs=np.array([1e6, 100.0, 0.01, 1e-5]))
        y_poly = Poly1D(coefs=np.array([2e6, -50.0, 0.005]))
        z_poly = Poly1D(coefs=np.array([7e6, 0.0, -0.002]))
        arp = XYZPoly(x=x_poly, y=y_poly, z=z_poly)

        # Position at t=0
        np.testing.assert_allclose(arp(0.0), [1e6, 2e6, 7e6])

        # Velocity at t=0: dX/dt = 100, dY/dt = -50, dZ/dt = 0
        vel = arp.derivative_eval(0.0, n=1)
        np.testing.assert_allclose(vel, [100.0, -50.0, 0.0])

        # Velocity at t=10: dX/dt = 100 + 0.02*10 + 3e-5*100
        #                         = 100 + 0.2 + 0.003 = 100.203
        vel10 = arp.derivative_eval(10.0, n=1)
        assert vel10[0] == pytest.approx(100.203)
