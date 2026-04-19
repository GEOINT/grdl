# -*- coding: utf-8 -*-
"""
CRSD Metadata - Typed metadata for Compensated Radar Signal Data.

Per NGA.STND.0080-1 v1.0 (2025-02-25). Covers the fields required to
recover per-channel Doppler/timing behaviour: receive reference point,
reference frequency, and the 2-D DwellTime / CODTime polynomials that
express aperture-time vs image coordinate.  Lightweight -- the full
CRSD XML tree is far larger than this model, but these fields are the
minimum required for downstream native projection / image formation.

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
2026-04-17

Modified
--------
2026-04-17
"""

# Standard library
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Third-party
import numpy as np

# GRDL internal
from grdl.IO.models.base import ImageMetadata
from grdl.IO.models.common import XYZ, Poly2D


@dataclass
class CRSDDwellPolynomialSet:
    """A CRSD top-level DwellPolynomials entry.

    Each identifier in ``CRSDsar/DwellPolynomials/CODTime`` or
    ``.../DwellTime`` names a 2-D polynomial in (IAX, IAY) whose value
    is in seconds.  Channels reference one by ID via
    ``Channel/Parameters/SARImage/DwellTimes/Polynomials/{CODId,DwellId}``.

    Parameters
    ----------
    identifier : str
        The ID string used to cross-reference channel parameters.
    poly : Poly2D
        The 2-D polynomial P(iax, iay) producing seconds.
    """

    identifier: str
    poly: Poly2D


@dataclass
class CRSDChannelParameters:
    """Per-channel CRSD parameters needed for Doppler/time recovery.

    Parameters
    ----------
    channel_id : str, optional
        Channel identifier (``CRSDsar/Channel/Parameters/Identifier``).
    num_vectors : int, optional
        Number of receive vectors / pulses in this channel.
    num_samples : int, optional
        Number of signal samples per vector.
    f0_ref : float, optional
        Scalar reference frequency ``f_0(v_CH_REF)``, Hz
        (``Channel/Parameters/F0Ref``).
    ref_freq_fixed : bool, optional
        Whether the per-vector RefFreq is fixed at ``f0_ref``.
    rcv_ref_point_ecf : XYZ, optional
        Receive reference point in ECEF meters
        (``Channel/Parameters/RcvRefPoint/ECF``).
    rcv_ref_point_iac : Optional[tuple], optional
        Receive reference point in image-area coords (IAX, IAY) meters
        (``Channel/Parameters/RcvRefPoint/IAC``).
    cod_poly_id : str, optional
        Cross-reference to the CODTime entry in ``DwellPolynomials``
        (``Channel/Parameters/SARImage/DwellTimes/Polynomials/CODId``).
    dwell_poly_id : str, optional
        Cross-reference to the DwellTime entry in ``DwellPolynomials``.
    cod_poly : Poly2D, optional
        Resolved center-of-dwell time polynomial seconds(iax, iay).
    dwell_poly : Poly2D, optional
        Resolved dwell time polynomial seconds(iax, iay).
    """

    channel_id: Optional[str] = None
    num_vectors: Optional[int] = None
    num_samples: Optional[int] = None
    f0_ref: Optional[float] = None
    ref_freq_fixed: Optional[bool] = None
    rcv_ref_point_ecf: Optional[XYZ] = None
    rcv_ref_point_iac: Optional[tuple] = None
    cod_poly_id: Optional[str] = None
    dwell_poly_id: Optional[str] = None
    cod_poly: Optional[Poly2D] = None
    dwell_poly: Optional[Poly2D] = None


@dataclass
class CRSDReferenceGeometry:
    """CRSD scalar reference geometry at the scene reference point.

    Parameters
    ----------
    ref_point_ecf : XYZ, optional
        Scene reference point, ECEF meters
        (``ReferenceGeometry/RefPoint/ECF``).
    ref_point_iac : Optional[tuple], optional
        Scene reference point in image-area coords (IAX, IAY) meters.
    cod_time : float, optional
        Reference center-of-dwell time, seconds
        (``ReferenceGeometry/SARImage/CODTime``).
    dwell_time : float, optional
        Reference dwell time, seconds
        (``ReferenceGeometry/SARImage/DwellTime``).
    """

    ref_point_ecf: Optional[XYZ] = None
    ref_point_iac: Optional[tuple] = None
    cod_time: Optional[float] = None
    dwell_time: Optional[float] = None


@dataclass
class CRSDMetadata(ImageMetadata):
    """Typed metadata for CRSD products.

    Extends the generic ``ImageMetadata`` with the per-channel
    parameters and DwellPolynomials cross-references required to
    reconstruct per-pulse Doppler timing.  Lightweight by design -- the
    full CRSD XML tree is parsed lazily by the reader; this class only
    surfaces the fields that downstream geolocation / image formation
    needs.

    Parameters
    ----------
    product_type : str, optional
        ``'CRSDsar'``, ``'CRSDtx'``, or ``'CRSDrcv'``.
    channels : Dict[str, CRSDChannelParameters], optional
        Per-channel parameters, keyed by channel identifier.
    cod_polys : Dict[str, CRSDDwellPolynomialSet], optional
        Top-level CODTime polynomials keyed by identifier.
    dwell_polys : Dict[str, CRSDDwellPolynomialSet], optional
        Top-level DwellTime polynomials keyed by identifier.
    reference_geometry : CRSDReferenceGeometry, optional
        Scalar reference geometry (scene-level).
    fx_rates : Dict[str, np.ndarray], optional
        Per-channel ``FxRate`` arrays from the PPP block (CRSDsar /
        CRSDtx only).  Shape ``(num_pulses,)``, units Hz/s.
    ref_freqs : Dict[str, np.ndarray], optional
        Per-channel ``RefFreq`` arrays from the PVP block.  Shape
        ``(num_vectors,)``, units Hz.
    """

    product_type: Optional[str] = None
    channels: Dict[str, CRSDChannelParameters] = field(default_factory=dict)
    cod_polys: Dict[str, CRSDDwellPolynomialSet] = field(default_factory=dict)
    dwell_polys: Dict[str, CRSDDwellPolynomialSet] = field(default_factory=dict)
    reference_geometry: Optional[CRSDReferenceGeometry] = None
    fx_rates: Dict[str, np.ndarray] = field(default_factory=dict)
    ref_freqs: Dict[str, np.ndarray] = field(default_factory=dict)
