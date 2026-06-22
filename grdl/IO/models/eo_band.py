# -*- coding: utf-8 -*-
"""
Multispectral band TRE metadata -- BANDSB and BANDSA dataclasses.

Typed metadata models for the multispectral / hyperspectral band
parameter Tagged Record Extensions defined in STDI-0002:

* ``BANDSB`` -- Band Set B (spectral band parameters with existence
  mask; per-band center wavelength, FWHM, band bounds).
* ``BANDSA`` -- Band Set A (airborne band parameters; per-band peak
  response, band bounds, bandwidth, calibration values).

Wavelength-bearing fields on :class:`BANDSBBand` are normalized to
micrometers when the TRE's ``WAVE_LENGTH_UNIT`` permits conversion
(``U`` = micrometers verbatim, ``W`` = wavenumber in cm^-1 converted
via ``lambda_um = 1e4 / nu``).

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
2026-06-09

Modified
--------
2026-06-09
"""

# Standard library
from dataclasses import dataclass, field
from typing import List, Optional


# ===================================================================
# BANDSB -- Band Set B (STDI-0002, spectral band parameters)
# ===================================================================


@dataclass
class BANDSBBand:
    """Per-band spectral parameters from a BANDSB TRE.

    All fields are optional -- the BANDSB existence mask controls
    which per-band parameters are present in a given file.  Fields
    absent from the TRE are ``None``.

    Parameters
    ----------
    band_id : str, optional
        Band identifier string (``BANDIDn``).
    bad_band : int, optional
        Bad-band flag (``BAD_BANDn``); 1 = usable, 0 = bad.
    niirs : float, optional
        Per-band NIIRS rating (``NIIRSn``).
    focal_length : int, optional
        Per-band focal length (``FOCAL_LENn``), centimeters.
    center_wavelength_um : float, optional
        Band center response wavelength (``CWAVEn``), micrometers.
    fwhm_um : float, optional
        Full width at half maximum (``FWHMn``), micrometers.
    nominal_wavelength_um : float, optional
        Nominal principal wavelength (``NOM_WAVEn``), micrometers.
    lower_bound_um : float, optional
        Lower spectral bound (``LBOUNDn``), micrometers.
    upper_bound_um : float, optional
        Upper spectral bound (``UBOUNDn``), micrometers.
    """

    band_id: Optional[str] = None
    bad_band: Optional[int] = None
    niirs: Optional[float] = None
    focal_length: Optional[int] = None
    center_wavelength_um: Optional[float] = None
    fwhm_um: Optional[float] = None
    nominal_wavelength_um: Optional[float] = None
    lower_bound_um: Optional[float] = None
    upper_bound_um: Optional[float] = None


@dataclass
class BANDSBMetadata:
    """BANDSB TRE -- spectral band set metadata.

    Parameters
    ----------
    band_count : int
        Number of bands described (``COUNT``).
    radiometric_quantity : str, optional
        Data representation, e.g. ``RADIANCE`` or ``REFLECTANCE``
        (``RADIOMETRIC_QUANTITY``).
    radiometric_quantity_unit : str, optional
        Unit code for the radiometric quantity
        (``RADIOMETRIC_QUANTITY_UNIT``).
    wave_length_unit : str, optional
        Wavelength unit code from the TRE (``WAVE_LENGTH_UNIT``):
        ``U`` = micrometers, ``W`` = wavenumber (cm^-1).  Per-band
        wavelength fields are already normalized to micrometers.
    bands : list of BANDSBBand
        Per-band spectral parameters, in band order.
    """

    band_count: int = 0
    radiometric_quantity: Optional[str] = None
    radiometric_quantity_unit: Optional[str] = None
    wave_length_unit: Optional[str] = None
    bands: List[BANDSBBand] = field(default_factory=list)

    @property
    def band_names(self) -> List[str]:
        """Band identifier per band, with positional fallback.

        Returns
        -------
        list of str
            ``BANDIDn`` when present, otherwise ``'Band <n>'``
            (1-based).
        """
        names: List[str] = []
        for i, band in enumerate(self.bands):
            if band.band_id:
                names.append(band.band_id)
            else:
                names.append(f'Band {i + 1}')
        return names

    @property
    def wavelengths_um(self) -> List[float]:
        """Center wavelength per band in micrometers.

        Returns
        -------
        list of float
            ``CWAVEn`` in micrometers; ``nan`` for bands without a
            center wavelength.
        """
        return [
            band.center_wavelength_um
            if band.center_wavelength_um is not None else float('nan')
            for band in self.bands
        ]


# ===================================================================
# BANDSA -- Band Set A (STDI-0002, airborne band parameters)
# ===================================================================


@dataclass
class BANDSABand:
    """Per-band parameters from a BANDSA TRE.

    Blank (all-space) fields in the fixed-width layout are ``None``.

    Parameters
    ----------
    peak_response : float, optional
        Wavelength of band peak response (``BANDPEAK``), micrometers.
    lower_bound : float, optional
        Lower band bound at 50% response (``BANDLBOUND``),
        micrometers.
    upper_bound : float, optional
        Upper band bound at 50% response (``BANDUBOUND``),
        micrometers.
    bandwidth : float, optional
        Band width at 50% response (``BANDWIDTH``), micrometers.
    cal_dark_value : float, optional
        Calibration dark count value (``BANDCALDRK``).
    cal_increment : float, optional
        Calibration increment (``BANDCALINC``).
    response : float, optional
        Band predicted response (``BANDRESP``).
    asd : float, optional
        Band atmospheric scattering data (``BANDASD``).
    gsd : float, optional
        Band ground sample distance (``BANDGSD``).
    """

    peak_response: Optional[float] = None
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    bandwidth: Optional[float] = None
    cal_dark_value: Optional[float] = None
    cal_increment: Optional[float] = None
    response: Optional[float] = None
    asd: Optional[float] = None
    gsd: Optional[float] = None


@dataclass
class BANDSAMetadata:
    """BANDSA TRE -- airborne band set metadata.

    Parameters
    ----------
    row_gsd : float, optional
        Row ground sample distance (``ROW_SPACING``).
    row_gsd_units : str, optional
        Row GSD unit code (``ROW_SPACING_UNITS``).
    col_gsd : float, optional
        Column ground sample distance (``COL_SPACING``).
    col_gsd_units : str, optional
        Column GSD unit code (``COL_SPACING_UNITS``).
    focal_length : float, optional
        Sensor focal length (``FOCAL_LENGTH``), centimeters.
    band_count : int
        Number of bands described (``BANDCOUNT``).
    bands : list of BANDSABand
        Per-band parameters, in band order.
    """

    row_gsd: Optional[float] = None
    row_gsd_units: Optional[str] = None
    col_gsd: Optional[float] = None
    col_gsd_units: Optional[str] = None
    focal_length: Optional[float] = None
    band_count: int = 0
    bands: List[BANDSABand] = field(default_factory=list)
