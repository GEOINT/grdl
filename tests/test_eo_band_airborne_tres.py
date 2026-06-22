# -*- coding: utf-8 -*-
"""
Tests for BANDSB/BANDSA band TREs and SENSRB/MENSRB/MENSRA/ACFTB
airborne TRE parsers.

Exercises both entry points per TRE: the xml:TRE Element path
(synthetic ``<tre>`` documents) and the raw fixed-width CEDATA
string path, plus malformed-input -> None behavior.

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
import math
import struct
from typing import List, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET

# GRDL internal
from grdl.IO.eo._tre_airborne import (
    parse_acftb,
    parse_acftb_cedata,
    parse_mensra,
    parse_mensra_cedata,
    parse_mensrb,
    parse_mensrb_cedata,
    parse_sensrb,
    parse_sensrb_cedata,
)
from grdl.IO.eo._tre_band import (
    parse_bandsa,
    parse_bandsa_cedata,
    parse_bandsb,
    parse_bandsb_cedata,
)


# ===================================================================
# Helpers
# ===================================================================


def make_tre(
    name: str,
    fields: Sequence[Tuple[str, str]],
    groups: Optional[List[Sequence[Tuple[str, str]]]] = None,
) -> ET.Element:
    """Build a synthetic ``<tre>`` element in GDAL xml:TRE shape."""
    parts = [f'<tre name="{name}">']
    for key, value in fields:
        parts.append(f'<field name="{key}" value="{value}"/>')
    if groups is not None:
        parts.append(f'<repeated name="GROUPS" number="{len(groups)}">')
        for i, group in enumerate(groups):
            parts.append(f'<group index="{i}">')
            for key, value in group:
                parts.append(f'<field name="{key}" value="{value}"/>')
            parts.append('</group>')
        parts.append('</repeated>')
    parts.append('</tre>')
    return ET.fromstring(''.join(parts))


def _f32(value: float) -> str:
    """4-byte big-endian IEEE-754 float as a latin-1 string."""
    return struct.pack('>f', value).decode('latin-1')


def _mask(value: int) -> str:
    """4-byte big-endian existence mask as a latin-1 string."""
    return value.to_bytes(4, 'big').decode('latin-1')


# ===================================================================
# BANDSB -- xml path
# ===================================================================


class TestBANDSBXml:

    def _node(self) -> ET.Element:
        return make_tre(
            'BANDSB',
            fields=[
                ('COUNT', '00003'),
                ('RADIOMETRIC_QUANTITY', 'REFLECTANCE'),
                ('RADIOMETRIC_QUANTITY_UNIT', 'F'),
                ('WAVE_LENGTH_UNIT', 'U'),
            ],
            groups=[
                [('BANDID', 'Blue'), ('CWAVE', '0.48500'),
                 ('FWHM', '0.06500'), ('LBOUND', '0.45000'),
                 ('UBOUND', '0.52000')],
                [('BANDID', ''), ('CWAVE', ''),
                 ('FWHM', ''), ('LBOUND', ''), ('UBOUND', '')],
                [('BANDID', 'NIR'), ('CWAVE', '0.86000'),
                 ('FWHM', '0.04000'), ('LBOUND', '0.84000'),
                 ('UBOUND', '0.88000')],
            ],
        )

    def test_expected_path(self):
        meta = parse_bandsb(self._node())
        assert meta is not None
        assert meta.band_count == 3
        assert meta.radiometric_quantity == 'REFLECTANCE'
        assert meta.radiometric_quantity_unit == 'F'
        assert meta.wave_length_unit == 'U'
        assert len(meta.bands) == 3
        assert meta.bands[0].band_id == 'Blue'
        assert meta.bands[0].center_wavelength_um == 0.485
        assert meta.bands[0].fwhm_um == 0.065
        assert meta.bands[0].lower_bound_um == 0.45
        assert meta.bands[0].upper_bound_um == 0.52
        assert meta.bands[1].center_wavelength_um is None

    def test_band_names_convenience(self):
        meta = parse_bandsb(self._node())
        assert meta.band_names == ['Blue', 'Band 2', 'NIR']

    def test_wavelengths_um_convenience(self):
        meta = parse_bandsb(self._node())
        wl = meta.wavelengths_um
        assert len(wl) == 3
        assert wl[0] == 0.485
        assert math.isnan(wl[1])
        assert wl[2] == 0.86

    def test_wavenumber_unit_conversion(self):
        node = make_tre(
            'BANDSB',
            fields=[('COUNT', '00001'), ('WAVE_LENGTH_UNIT', 'W')],
            groups=[[('CWAVE', '2000.00'), ('FWHM', '10.0000')]],
        )
        meta = parse_bandsb(node)
        assert meta is not None
        assert math.isclose(meta.bands[0].center_wavelength_um, 5.0)
        # d_lambda = 1e4 * d_nu / nu^2 = 1e4 * 10 / 2000^2
        assert math.isclose(meta.bands[0].fwhm_um, 0.025)

    def test_missing_count_returns_none(self):
        node = make_tre('BANDSB', fields=[
            ('RADIOMETRIC_QUANTITY', 'RADIANCE')])
        assert parse_bandsb(node) is None

    def test_wrong_tre_name_returns_none(self):
        node = make_tre('BANDSA', fields=[('COUNT', '00001')])
        assert parse_bandsb(node) is None


# ===================================================================
# BANDSB -- CEDATA path
# ===================================================================


def _bandsb_cedata_prefix(count: int, mask: int) -> str:
    return (
        f'{count:05d}'
        + 'REFLECTANCE'.ljust(24)
        + 'F'
        + _f32(1.0) + _f32(0.0)
        + '0001.25' + 'M' + '0001.25' + 'M'
        + '0002.50' + 'M' + '0002.50' + 'M'
        + ' ' * 48
        + _mask(mask)
    )


class TestBANDSBCedata:

    # BANDID (28) | WAVE_LENGTH_UNIT+CWAVE (24) | FWHM (23) |
    # LBOUND/UBOUND (19)
    MASK = (1 << 28) | (1 << 24) | (1 << 23) | (1 << 19)

    def _cedata(self) -> str:
        bands = (
            'Blue'.ljust(50) + '0.48500' + '0.06500'
            + '0.45000' + '0.52000'
            + 'NIR'.ljust(50) + '0.86000' + '0.04000'
            + '0.84000' + '0.88000'
        )
        return _bandsb_cedata_prefix(2, self.MASK) + 'U' + bands

    def test_expected_path(self):
        meta = parse_bandsb_cedata(self._cedata())
        assert meta is not None
        assert meta.band_count == 2
        assert meta.radiometric_quantity == 'REFLECTANCE'
        assert meta.wave_length_unit == 'U'
        assert meta.bands[0].band_id == 'Blue'
        assert meta.bands[0].center_wavelength_um == 0.485
        assert meta.bands[0].fwhm_um == 0.065
        assert meta.bands[1].lower_bound_um == 0.84
        assert meta.band_names == ['Blue', 'NIR']
        assert meta.wavelengths_um == [0.485, 0.86]

    def test_unsupported_mask_bit_returns_none(self):
        # Bit 18 selects per-band binary scale/additive factors --
        # unsupported in the text CEDATA path.
        cedata = _bandsb_cedata_prefix(1, self.MASK | (1 << 18))
        assert parse_bandsb_cedata(cedata) is None

    def test_undecodable_mask_returns_none(self):
        # Non-latin-1 character where the binary mask should be:
        # the xml path is authoritative for binary-mask files.
        bad = _bandsb_cedata_prefix(1, self.MASK)
        bad = bad[:-4] + '…xyz'
        assert parse_bandsb_cedata(bad) is None

    def test_truncated_returns_none(self):
        assert parse_bandsb_cedata(self._cedata()[:80]) is None

    def test_garbage_returns_none(self):
        assert parse_bandsb_cedata('not a bandsb tre') is None


# ===================================================================
# BANDSA -- xml path
# ===================================================================


class TestBANDSAXml:

    def _node(self) -> ET.Element:
        return make_tre(
            'BANDSA',
            fields=[
                ('ROW_SPACING', '0000.60'),
                ('ROW_SPACING_UNITS', 'm'),
                ('COL_SPACING', '0000.75'),
                ('COL_SPACING_UNITS', 'm'),
                ('FOCAL_LENGTH', '000150'),
                ('BANDCOUNT', '0002'),
            ],
            groups=[
                [('BANDPEAK', '0.550'), ('BANDLBOUND', '0.450'),
                 ('BANDUBOUND', '0.690'), ('BANDWIDTH', '0.240'),
                 ('BANDCALDRK', '000100'), ('BANDCALINC', '01.50'),
                 ('BANDRESP', '00.95'), ('BANDASD', '00.10'),
                 ('BANDGSD', '00.60')],
                [('BANDPEAK', '0.860'), ('BANDLBOUND', '0.770'),
                 ('BANDUBOUND', '0.900'), ('BANDWIDTH', '0.130'),
                 ('BANDCALDRK', ''), ('BANDCALINC', ''),
                 ('BANDRESP', ''), ('BANDASD', ''), ('BANDGSD', '')],
            ],
        )

    def test_expected_path(self):
        meta = parse_bandsa(self._node())
        assert meta is not None
        assert meta.row_gsd == 0.6
        assert meta.row_gsd_units == 'm'
        assert meta.col_gsd == 0.75
        assert meta.focal_length == 150.0
        assert meta.band_count == 2
        assert meta.bands[0].peak_response == 0.55
        assert meta.bands[0].lower_bound == 0.45
        assert meta.bands[0].upper_bound == 0.69
        assert meta.bands[0].bandwidth == 0.24
        assert meta.bands[0].cal_dark_value == 100.0
        assert meta.bands[1].peak_response == 0.86
        assert meta.bands[1].cal_dark_value is None

    def test_missing_bandcount_returns_none(self):
        node = make_tre('BANDSA', fields=[('ROW_SPACING', '0000.60')])
        assert parse_bandsa(node) is None

    def test_wrong_tre_name_returns_none(self):
        node = make_tre('BANDSB', fields=[('BANDCOUNT', '0001')])
        assert parse_bandsa(node) is None


# ===================================================================
# BANDSA -- CEDATA path
# ===================================================================


class TestBANDSACedata:

    def _cedata(self) -> str:
        band1 = ('0.550' + '0.450' + '0.690' + '0.240'
                 + '000100' + '01.50' + '00.95' + '00.10' + '00.60')
        band2 = ('0.860' + '0.770' + '0.900' + '0.130'
                 + ' ' * 6 + ' ' * 5 + ' ' * 5 + ' ' * 5 + ' ' * 5)
        head = ('0000.60' + 'm' + '0000.75' + 'm' + '000150' + '0002')
        assert len(band1) == 46 and len(band2) == 46
        return head + band1 + band2

    def test_expected_path(self):
        meta = parse_bandsa_cedata(self._cedata())
        assert meta is not None
        assert meta.row_gsd == 0.6
        assert meta.col_gsd == 0.75
        assert meta.col_gsd_units == 'm'
        assert meta.focal_length == 150.0
        assert meta.band_count == 2
        assert meta.bands[0].gsd == 0.6
        assert meta.bands[0].cal_increment == 1.5
        assert meta.bands[1].bandwidth == 0.13
        assert meta.bands[1].cal_dark_value is None
        assert meta.bands[1].gsd is None

    def test_truncated_returns_none(self):
        assert parse_bandsa_cedata(self._cedata()[:40]) is None

    def test_garbage_returns_none(self):
        assert parse_bandsa_cedata('XXXXXXXXXXXXXXXXXXXXXXXXXX') is None


# ===================================================================
# SENSRB -- xml path
# ===================================================================


class TestSENSRBXml:

    def _node(self) -> ET.Element:
        return make_tre('SENSRB', fields=[
            ('GENERAL_DATA', 'Y'),
            ('SENSOR', 'EO-SENSOR-ALPHA'),
            ('SENSOR_URI', ''),
            ('PLATFORM', 'TEST-PLATFORM'),
            ('OPERATION_DOMAIN', 'Airborne'),
            ('CONTENT_LEVEL', '4'),
            ('GEODETIC_SYSTEM', 'WGS84'),
            ('GEODETIC_TYPE', 'G'),
            ('ELEVATION_DATUM', 'HAE'),
            ('START_DATE', '20260609'),
            ('START_TIME', '12.5'),
            ('REFERENCE_TIME', '0.0'),
            ('REFERENCE_ROW', '512'),
            ('REFERENCE_COLUMN', '640'),
            ('LATITUDE_OR_X', '34.123'),
            ('LONGITUDE_OR_Y', '-117.25'),
            ('ALTITUDE_OR_Z', '8500.0'),
            ('SENSOR_X_OFFSET', '0'),
            ('SENSOR_Y_OFFSET', '0'),
            ('SENSOR_Z_OFFSET', '0'),
            ('ATTITUDE_EULER_ANGLES', 'Y'),
            ('SENSOR_ANGLE_MODEL', '1'),
            ('SENSOR_ANGLE_1', '45.0'),
            ('SENSOR_ANGLE_2', '-5.0'),
            ('SENSOR_ANGLE_3', '0.0'),
            ('PLATFORM_RELATIVE', 'Y'),
            ('PLATFORM_HEADING', '90.0'),
            ('PLATFORM_PITCH', '1.5'),
            ('PLATFORM_ROLL', '-2.25'),
            # Uncovered module field -- must be retained in raw
            ('DETECTION', 'PANCHROMATIC'),
        ])

    def test_expected_path(self):
        meta = parse_sensrb(self._node())
        assert meta is not None
        assert meta.sensor == 'EO-SENSOR-ALPHA'
        assert meta.platform == 'TEST-PLATFORM'
        assert meta.operation_domain == 'Airborne'
        assert meta.content_level == 4
        assert meta.geodetic_system == 'WGS84'
        assert meta.start_time == 12.5
        assert meta.latitude_or_x == 34.123
        assert meta.longitude_or_y == -117.25
        assert meta.altitude_or_z == 8500.0
        assert meta.sensor_angle_model == 1
        assert meta.sensor_angle_1 == 45.0
        assert meta.platform_heading == 90.0
        assert meta.platform_roll == -2.25

    def test_uncovered_module_retained_in_raw(self):
        meta = parse_sensrb(self._node())
        assert meta.raw['DETECTION'] == 'PANCHROMATIC'
        assert meta.raw['SENSOR'] == 'EO-SENSOR-ALPHA'

    def test_wrong_tre_name_returns_none(self):
        node = make_tre('MENSRB', fields=[('SENSOR', 'X')])
        assert parse_sensrb(node) is None


# ===================================================================
# SENSRB -- CEDATA path
# ===================================================================


def _sensrb_cedata() -> str:
    general = (
        'EO-SENSOR-ALPHA'.ljust(25)
        + ' ' * 32
        + 'TEST-PLATFORM'.ljust(25)
        + ' ' * 32
        + 'Airborne'.ljust(10)
        + '4'
        + 'WGS84' + 'G' + 'HAE' + 'SI' + 'DEG'
        + '20260609' + '00000012.00000'
        + '20260609' + '00000099.00000'
        + '00' + '20260609' + '0000000000'
    )
    ref = '000000000.00' + '00000512' + '00000640'
    position = (
        '34.12345678' + '-117.1234567' + '00008500.00'
        + '00000000' + '00000000' + '00000000'
    )
    euler = (
        '1' + '000045.000' + '-0005.000' + '000000.000'
        + 'Y' + '00090.000' + '00001.500' + '-00002.250'
    )
    tail = 'N' + 'N' + 'N' + '00' + '00' + '00' + '000' + '000'
    return 'Y' + general + 'N' + 'N' + 'N' + ref + position \
        + 'Y' + euler + tail


class TestSENSRBCedata:

    def test_expected_path(self):
        meta = parse_sensrb_cedata(_sensrb_cedata())
        assert meta is not None
        assert meta.sensor == 'EO-SENSOR-ALPHA'
        assert meta.platform == 'TEST-PLATFORM'
        assert meta.operation_domain == 'Airborne'
        assert meta.content_level == 4
        assert meta.geodetic_system == 'WGS84'
        assert meta.geodetic_type == 'G'
        assert meta.elevation_datum == 'HAE'
        assert meta.length_unit == 'SI'
        assert meta.angular_unit == 'DEG'
        assert meta.start_date == '20260609'
        assert meta.start_time == 12.0
        assert meta.end_time == 99.0
        assert meta.reference_time == 0.0
        assert meta.reference_row == 512.0
        assert meta.reference_column == 640.0
        assert meta.latitude_or_x == 34.12345678
        assert meta.longitude_or_y == -117.1234567
        assert meta.altitude_or_z == 8500.0
        assert meta.sensor_angle_model == 1
        assert meta.sensor_angle_1 == 45.0
        assert meta.sensor_angle_2 == -5.0
        assert meta.platform_relative == 'Y'
        assert meta.platform_heading == 90.0
        assert meta.platform_pitch == 1.5
        assert meta.platform_roll == -2.25

    def test_uncovered_tail_retained_in_raw(self):
        meta = parse_sensrb_cedata(_sensrb_cedata())
        assert '_unparsed' in meta.raw
        assert meta.raw['_unparsed'].startswith('N')

    def test_truncated_returns_none(self):
        assert parse_sensrb_cedata('Y' + 'X' * 10) is None

    def test_general_data_skipped_when_flag_n(self):
        cedata = _sensrb_cedata()
        # Replace the leading 'Y' + 203-char general module with 'N'.
        meta = parse_sensrb_cedata('N' + cedata[204:])
        assert meta is not None
        assert meta.sensor is None
        assert meta.latitude_or_x == 34.12345678


# ===================================================================
# MENSRB
# ===================================================================


def _mensrb_cedata() -> str:
    cedata = (
        '331025.12N0117025.12W'.ljust(25)   # ACFT_LOC
        + '0025.0'                          # ACFT_LOC_ACCY
        + '012500'                          # ACFT_ALT
        + '331020.00N0117030.00W'.ljust(25)  # RP_LOC
        + '0010.0'                          # RP_LOC_ACCY
        + '001250'                          # RP_ELV
        + '0000.00' + '0000.00'             # OF_PC_R, OF_PC_A
        + '0.86603'                         # COSGRZ
        + '0035000'                         # RGCRP
        + 'R'                               # RLMAP
        + '00512' + '00640'                 # RP_ROW, RP_COL
        + '0.50000000' + '-0.5000000' + '0.70710678'   # C_R_*
        + '0.5000000' + '-0.500000' + '0.0000000'      # C_AZ_*
        + '0.0000000' + '0.0000000' + '1.0000000'      # C_AL_*
        + '004' + '00016'                   # TOTAL_TILES_COLS/ROWS
    )
    return cedata


class TestMENSRB:

    def test_cedata_expected_path(self):
        cedata = _mensrb_cedata()
        assert len(cedata) == 205
        meta = parse_mensrb_cedata(cedata)
        assert meta is not None
        assert meta.acft_loc == '331025.12N0117025.12W'
        assert meta.acft_loc_accy == 25.0
        assert meta.acft_alt == 12500.0
        assert meta.rp_loc == '331020.00N0117030.00W'
        assert meta.rp_elv == 1250.0
        assert meta.cos_graze == 0.86603
        assert meta.range_crp == 35000.0
        assert meta.rl_map == 'R'
        assert meta.rp_row == 512
        assert meta.rp_col == 640
        assert meta.c_r_nc == 0.5
        assert meta.c_r_ec == -0.5
        assert meta.c_r_dc == 0.70710678
        assert meta.c_az_ec == -0.5
        assert meta.c_al_dc == 1.0
        assert meta.total_tiles_cols == 4
        assert meta.total_tiles_rows == 16

    def test_cedata_truncated_returns_none(self):
        assert parse_mensrb_cedata(_mensrb_cedata()[:100]) is None

    def test_xml_expected_path(self):
        node = make_tre('MENSRB', fields=[
            ('ACFT_LOC', '331025.12N0117025.12W'),
            ('ACFT_LOC_ACCY', '0025.0'),
            ('ACFT_ALT', '012500'),
            ('RP_LOC', '331020.00N0117030.00W'),
            ('RP_ELV', '001250'),
            ('COSGRZ', '0.86603'),
            ('RGCRP', '0035000'),
            ('RLMAP', 'R'),
            ('RP_ROW', '00512'),
            ('RP_COL', '00640'),
            ('C_R_NC', '0.50000000'),
            ('TOTAL_TILES_ROWS', '00016'),
        ])
        meta = parse_mensrb(node)
        assert meta is not None
        assert meta.acft_alt == 12500.0
        assert meta.cos_graze == 0.86603
        assert meta.rp_row == 512
        assert meta.c_r_nc == 0.5
        assert meta.total_tiles_rows == 16

    def test_xml_wrong_name_returns_none(self):
        node = make_tre('MENSRA', fields=[('ACFT_ALT', '012500')])
        assert parse_mensrb(node) is None


# ===================================================================
# MENSRA
# ===================================================================


def _mensra_174_cedata() -> str:
    cosines = ('0.5000000' + '-0.500000' + '0.7071068'
               + '0.5000000' + '0.5000000' + '0.0000000'
               + '0.0000000' + '0.0000000' + '1.0000000')
    return (
        '331025.12N0117025.12W'   # ACFT_LOC (21)
        + '012500'                # ACFT_ALT
        + '331020.00N0117030.00W'  # CCRP_LOC (21)
        + '001250'                # CCRP_ALT
        + '0000.00' + '0000.00'   # OF_PC_R, OF_PC_A
        + '0.86603'               # COSGRZ
        + '0035000'               # RGCCRP
        + 'L'                     # RLMAP
        + '00512' + '00640'       # CCRP_ROW, CCRP_COL
        + cosines
    )


class TestMENSRA:

    def test_cedata_174_expected_path(self):
        cedata = _mensra_174_cedata()
        assert len(cedata) == 174
        meta = parse_mensra_cedata(cedata)
        assert meta is not None
        assert meta.acft_loc == '331025.12N0117025.12W'
        assert meta.acft_alt == 12500.0
        assert meta.ccrp_loc == '331020.00N0117030.00W'
        assert meta.ccrp_alt == 1250.0
        assert meta.cos_graze == 0.86603
        assert meta.range_ccrp == 35000.0
        assert meta.rl_map == 'L'
        assert meta.ccrp_row == 512
        assert meta.ccrp_col == 640
        assert meta.c_r_ec == -0.5
        assert meta.c_al_dc == 1.0
        # 174-byte variant carries no tile counts
        assert meta.total_tiles_cols is None
        assert meta.total_tiles_rows is None

    def test_cedata_unknown_length_returns_none(self):
        assert parse_mensra_cedata('X' * 160) is None

    def test_xml_expected_path(self):
        node = make_tre('MENSRA', fields=[
            ('ACFT_LOC', '331025.12N0117025.12W'),
            ('ACFT_ALT', '012500'),
            ('CCRP_LOC', '331020.00N0117030.00W'),
            ('COSGZ', '0.86603'),   # 185-byte variant spelling
            ('RGCCRP', '0035000'),
            ('CCRP_ROW', '00512'),
        ])
        meta = parse_mensra(node)
        assert meta is not None
        assert meta.acft_alt == 12500.0
        assert meta.cos_graze == 0.86603
        assert meta.range_ccrp == 35000.0
        assert meta.ccrp_row == 512

    def test_xml_wrong_name_returns_none(self):
        node = make_tre('MENSRB', fields=[('ACFT_ALT', '012500')])
        assert parse_mensra(node) is None


# ===================================================================
# ACFTB
# ===================================================================


def _acftb_cedata() -> str:
    return (
        'MISSION-ALPHA'.ljust(20)           # AC_MSN_ID
        + 'N12345'.ljust(10)                # AC_TAIL_NO
        + '202606090830'                    # AC_TO
        + 'EO'.ljust(4)                     # SENSOR_ID_TYPE
        + 'SYERS'.ljust(6)                  # SENSOR_ID
        + '1'                               # SCENE_SOURCE
        + '000042'                          # SCNUM
        + '20260609'                        # PDATE
        + '000001'                          # IMHOSTNO
        + '00001'                           # IMREQID
        + '001'                             # MPLAN
        + '331025.12N0117025.12W'.ljust(25)  # ENTLOC
        + '0025.0'                          # LOC_ACCY
        + '001250'                          # ENTELV
        + 'f'                               # ELV_UNIT
        + '331030.00N0117020.00W'.ljust(25)  # EXITLOC
        + '001300'                          # EXITELV
        + '000.000'                         # TMAP
        + '0000.60' + 'm'                   # ROW_SPACING + units
        + '0000.75' + 'm'                   # COL_SPACING + units
        + '0150.0'                          # FOCAL_LENGTH
        + '123456'                          # SENSERIAL
        + '01.00.0'                         # ABSWVER
        + '20260101'                        # CAL_DATE
        + '0001'                            # PATCH_TOT
        + '000'                             # MTI_TOT
    )


class TestACFTB:

    def test_cedata_expected_path(self):
        cedata = _acftb_cedata()
        assert len(cedata) == 207
        meta = parse_acftb_cedata(cedata)
        assert meta is not None
        assert meta.mission_id == 'MISSION-ALPHA'
        assert meta.tail_number == 'N12345'
        assert meta.takeoff_datetime == '202606090830'
        assert meta.sensor_id_type == 'EO'
        assert meta.sensor_id == 'SYERS'
        assert meta.scene_source == '1'
        assert meta.entry_location == '331025.12N0117025.12W'
        assert meta.location_accuracy == 25.0
        assert meta.entry_elevation == 1250.0
        assert meta.elevation_unit == 'f'
        assert meta.row_gsd == 0.6
        assert meta.row_gsd_units == 'm'
        assert meta.col_gsd == 0.75
        assert meta.focal_length == 150.0
        assert meta.patch_total == 1
        assert meta.mti_total == 0

    def test_cedata_truncated_returns_none(self):
        assert parse_acftb_cedata(_acftb_cedata()[:120]) is None

    def test_xml_expected_path(self):
        node = make_tre('ACFTB', fields=[
            ('AC_MSN_ID', 'MISSION-ALPHA'),
            ('AC_TAIL_NO', 'N12345'),
            ('AC_TO', '202606090830'),
            ('SENSOR_ID_TYPE', 'EO'),
            ('SENSOR_ID', 'SYERS'),
            ('SCENE_SOURCE', '1'),
            ('ENTLOC', '331025.12N0117025.12W'),
            ('ENTELV', '001250'),
            ('ELV_UNIT', 'f'),
            ('ROW_SPACING', '0000.60'),
            ('ROW_SPACING_UNITS', 'm'),
            ('COL_SPACING', '0000.75'),
            ('COL_SPACING_UNITS', 'm'),
            ('PATCH_TOT', '0001'),
        ])
        meta = parse_acftb(node)
        assert meta is not None
        assert meta.mission_id == 'MISSION-ALPHA'
        assert meta.tail_number == 'N12345'
        assert meta.sensor_id == 'SYERS'
        assert meta.entry_elevation == 1250.0
        assert meta.row_gsd == 0.6
        assert meta.col_gsd_units == 'm'
        assert meta.patch_total == 1

    def test_xml_wrong_name_returns_none(self):
        node = make_tre('ACFTA', fields=[('AC_MSN_ID', 'X')])
        assert parse_acftb(node) is None
