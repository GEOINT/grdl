# -*- coding: utf-8 -*-
"""Dual-pol scattering power components (decomposition/factorization modes).

References
----------
Verma, A., Bhattacharya, A., Dey, S., and Marino, A. (2024),
"Enhanced Target Characterization with Dual-Pol Sentinel-1 SAR Data,"
in IGARSS 2024 - 2024 IEEE International Geoscience and Remote
Sensing Symposium, pp.11461-11464.
doi:10.1109/IGARSS53475.2024.10642506.

Bhogapurapu, N., Dey, S., Bhattacharya, A., Mandal, D.,
Lopez-Sanchez, J.M., McNairn, H., Lopez-Martinez, C., and Rao, Y.S.
(2021), "Dual-polarimetric descriptors from Sentinel-1 GRD SAR data
for crop growth assessment," ISPRS Journal of Photogrammetry and
Remote Sensing, 178, pp.20-35. doi:10.1016/j.isprsjprs.2021.05.013.

Dey, S., Bhattacharya, A., Ratha, D., Mandal, D., and Frery, A.C.
(2020), "Target characterization and scattering power decomposition
for full and compact polarimetric SAR data," IEEE Geoscience and
Remote Sensing Letters, 18(6), pp.1048-1052.

Notes
-----
The GRDL implementation is an independently structured implementation
validated against open-form equations and numerical cross-checks.
"""

from typing import Annotated, Dict, Tuple, TYPE_CHECKING

import numpy as np

from grdl.image_processing.decomposition.dual_pol_base import DualPolDecompositionBase
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.image_processing.params import Range, Desc
from grdl.vocabulary import ImageModality

if TYPE_CHECKING:
    from grdl.IO.models.base import ImageMetadata


@processor_version('1.0.0')
@processor_tags(modalities=[ImageModality.SAR])
class ScatteringPowerDP(DualPolDecompositionBase):
    """Dual-pol scattering powers from DpRBI/DpRSI-derived characteristics.

    method=1: decomposition-based components (alpha, Pd, Ps, Pu)
    method=2: factorization-based components (Pd, Ps, Pr)
    """

    window_size: Annotated[int, Range(min=1, max=31),
                           Desc('Boxcar averaging window size')] = 7
    method: Annotated[int, Range(min=1, max=2),
                      Desc('1=decomposition, 2=factorization')] = 1
    nesz_db: float = -16.0
    percentile_low: float = 2.0
    percentile_high: float = 98.0

    @property
    def component_names(self) -> Tuple[str, ...]:
        if int(self.method) == 1:
            return ('alpha', 'double_bounce', 'surface', 'unpolarized')
        return ('double_bounce', 'surface', 'residual')

    def decompose_dual(self, s_co: np.ndarray, s_cross: np.ndarray) -> Dict[str, np.ndarray]:
        self._validate_dual_inputs(s_co, s_cross)
        c11, c12, c22 = self._compute_c2(s_co, s_cross, self.window_size)

        c11_db = 10.0 * np.log10(np.maximum(np.real(c11), np.finfo(np.float64).tiny))
        s0, s1, s2, s3, s1_norm, s2_norm, s3_norm = self._normalized_stokes(
            c11,
            c12,
            c22,
            percentile_low=self.percentile_low,
            percentile_high=self.percentile_high,
        )
        s0_abs = np.abs(s0)

        tpp = np.sqrt(s1 ** 2 + s2 ** 2 + s3 ** 2)
        l1 = (s0 + tpp) / 2.0
        l2 = (s0 - tpp) / 2.0
        lsum = l1 + l2
        eps = np.finfo(np.float64).tiny

        with np.errstate(divide='ignore', invalid='ignore'):
            p1 = l1 / np.where(np.abs(lsum) > eps, lsum, eps)
            p2 = l2 / np.where(np.abs(lsum) > eps, lsum, eps)
            ent = -(p1 * np.log2(p1) + p2 * np.log2(p2))
            dop = (l1 - l2) / np.where(np.abs(lsum) > eps, lsum, eps)
            beta = l1 / np.where(np.abs(lsum) > eps, lsum, eps)

        ent = np.real(np.nan_to_num(ent, nan=0.0, posinf=0.0, neginf=0.0))
        dop = np.real(np.nan_to_num(dop, nan=0.0, posinf=0.0, neginf=0.0))
        beta = np.real(np.nan_to_num(beta, nan=0.0, posinf=0.0, neginf=0.0))

        dprbi = np.sqrt(s1_norm ** 2 + s2_norm ** 2 + s3_norm ** 2) / np.sqrt(3.0)
        root_term = np.sqrt(np.clip(1.0 - s1_norm ** 2, 0.0, 1.0))
        dprsi_valid = (1.0 - ent) * root_term
        dprsi_noise = root_term
        dprsi = np.where(c11_db > self.nesz_db, dprsi_valid, dprsi_noise)

        if int(self.method) == 1:
            alpha1 = np.degrees(np.arctan2(dprbi, 1.0 - dprbi))
            alpha2 = np.degrees(np.arctan2(1.0 - dprsi, dprsi))
            alpha_dp = (alpha1 + alpha2) / 2.0

            alpha_rad = np.radians(2.0 * alpha_dp)
            cos_a = np.cos(alpha_rad)

            pu_valid = (1.0 - dop) * s0_abs
            pd_valid = 0.5 * dop * s0_abs * (1.0 - cos_a)
            ps_valid = 0.5 * dop * s0_abs * (1.0 + cos_a)

            pu_noise = (1.0 - beta) * s0_abs
            pd_noise = 0.5 * beta * s0_abs * (1.0 - cos_a)
            ps_noise = 0.5 * beta * s0_abs * (1.0 + cos_a)

            pu = np.where(c11_db > self.nesz_db, pu_valid, pu_noise)
            pd = np.where(c11_db > self.nesz_db, pd_valid, pd_noise)
            ps = np.where(c11_db > self.nesz_db, ps_valid, ps_noise)

            return {
                'alpha': np.real(alpha_dp),
                'double_bounce': np.real(pd),
                'surface': np.real(ps),
                'unpolarized': np.real(pu),
            }

        dprbi_flat = dprbi.ravel()
        dprsi_flat = dprsi.ravel()

        indices_vec = np.stack([dprsi_flat, dprbi_flat], axis=1)
        dominant = np.max(indices_vec, axis=1)
        subordinate = np.min(indices_vec, axis=1)

        y1 = dominant
        y2 = (1.0 - dominant) * subordinate
        residue = 1.0 - (y1 + y2)

        dprsi_dom = dprsi_flat > dprbi_flat
        dprbi_dom = dprsi_flat < dprbi_flat

        ps = np.zeros_like(dprbi_flat)
        ps[dprsi_dom] = y1[dprsi_dom]
        ps[dprbi_dom] = y2[dprbi_dom]

        pd = np.zeros_like(dprbi_flat)
        pd[dprbi_dom] = y1[dprbi_dom]
        pd[dprsi_dom] = y2[dprsi_dom]

        ps = ps.reshape(dprbi.shape) * s0_abs
        pd = pd.reshape(dprbi.shape) * s0_abs
        pr = residue.reshape(dprbi.shape) * s0_abs

        return {
            'double_bounce': np.real(pd),
            'surface': np.real(ps),
            'residual': np.real(pr),
        }

    def to_rgb(
        self,
        components: Dict[str, np.ndarray],
        representation: str = 'db',
        percentile_low: float = 2.0,
        percentile_high: float = 98.0,
    ) -> Tuple[np.ndarray, 'ImageMetadata']:
        from grdl.IO.models.base import ImageMetadata, ChannelMetadata

        if int(self.method) == 1:
            r = self._percentile_stretch(
                self._apply_power_representation(components['double_bounce'], representation),
                percentile_low,
                percentile_high,
            )
            g = self._percentile_stretch(
                self._apply_power_representation(components['unpolarized'], representation),
                percentile_low,
                percentile_high,
            )
            b = self._percentile_stretch(
                self._apply_power_representation(components['surface'], representation),
                percentile_low,
                percentile_high,
            )
            ch = [
                ChannelMetadata(index=0, name='double_bounce', role='rgb_red'),
                ChannelMetadata(index=1, name='unpolarized', role='rgb_green'),
                ChannelMetadata(index=2, name='surface', role='rgb_blue'),
            ]
            fmt = 'PowersDP1_RGB'
        else:
            r = self._percentile_stretch(
                self._apply_power_representation(components['double_bounce'], representation),
                percentile_low,
                percentile_high,
            )
            g = self._percentile_stretch(
                self._apply_power_representation(components['residual'], representation),
                percentile_low,
                percentile_high,
            )
            b = self._percentile_stretch(
                self._apply_power_representation(components['surface'], representation),
                percentile_low,
                percentile_high,
            )
            ch = [
                ChannelMetadata(index=0, name='double_bounce', role='rgb_red'),
                ChannelMetadata(index=1, name='residual', role='rgb_green'),
                ChannelMetadata(index=2, name='surface', role='rgb_blue'),
            ]
            fmt = 'PowersDP2_RGB'

        rgb = np.stack([r, g, b], axis=0)
        meta = ImageMetadata(
            format=fmt,
            rows=r.shape[0],
            cols=r.shape[1],
            bands=3,
            dtype='float32',
            axis_order='CYX',
            channel_metadata=ch,
        )
        return rgb, meta
