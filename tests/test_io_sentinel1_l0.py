# -*- coding: utf-8 -*-
"""
Tests for Sentinel-1 Level 0 reader components (Stage 2 coverage).

All tests use synthetic SAFE directories — no real products required.
Covers ``safe_product``, ``annotation_parser``, ``binary_parser``,
and ``timing`` modules.

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
2026-04-16

Modified
--------
2026-04-16
"""

# Standard library
import struct
import textwrap
from datetime import datetime
from pathlib import Path

# Third-party
import numpy as np
import pytest

# GRDL
from grdl.IO.models.sentinel1_l0 import (
    S1L0AttitudeRecord,
    S1L0OrbitStateVector,
    Sentinel1L0Metadata,
    Sentinel1Mission,
    Sentinel1Mode,
)
from grdl.IO.sar.sentinel1_l0 import constants
from grdl.IO.sar.sentinel1_l0.annotation_parser import (
    AnnotationParser,
    merge_annotation_data,
    parse_all_annotations,
    parse_annotation_datetime,
    parse_annotation_file,
)
from grdl.IO.sar.sentinel1_l0.binary_parser import (
    ANNOT_RECORD_SIZE,
    BurstIndexRecord,
    INDEX_RECORD_SIZE,
    PacketAnnotationRecord,
    find_burst_for_time,
    get_burst_byte_offsets,
    parse_burst_index_file,
    parse_packet_annotation_file,
)
from grdl.IO.sar.sentinel1_l0.safe_product import (
    InvalidSAFEProductError,
    ManifestInfo,
    ProductIdentifier,
    SAFEProduct,
    parse_manifest,
    parse_product_name,
)
from grdl.IO.sar.sentinel1_l0.timing import (
    TimeComponents,
    TimingCalculator,
    determine_reference_time,
)


# ===================================================================
# Constants
# ===================================================================


def test_constants_physical_values():
    """Physical constants have expected values."""
    assert constants.SPEED_OF_LIGHT == 299_792_458.0
    assert abs(
        constants.SENTINEL1_CENTER_FREQUENCY_HZ
        - 144.0 * constants.SENTINEL1_F_REF_HZ
    ) < 1e-6


def test_constants_swath_mappings():
    """IW raw swath 10/11/12 maps to logical 1/2/3."""
    from grdl.IO.sar.sentinel1_l0.constants import (
        raw_swath_to_logical,
        raw_swath_to_name,
    )
    assert raw_swath_to_name(10, "IW") == "IW1"
    assert raw_swath_to_name(11, "IW") == "IW2"
    assert raw_swath_to_name(12, "IW") == "IW3"
    assert raw_swath_to_logical(10, "IW") == 1


def test_constants_polarization_expansion():
    """Dual-pol codes expand to channel pairs."""
    from grdl.IO.sar.sentinel1_l0.constants import (
        expand_polarization,
        polarization_to_txrx,
    )
    assert expand_polarization("DV") == ("VV", "VH")
    assert expand_polarization("DH") == ("HH", "HV")
    assert expand_polarization("VV") == ("VV",)
    assert polarization_to_txrx("VV") == ("V", "V")
    assert polarization_to_txrx("VH") == ("V", "H")
    with pytest.raises(ValueError):
        polarization_to_txrx("XX")


# ===================================================================
# Timing
# ===================================================================


def test_time_components_from_positive_seconds():
    tc = TimeComponents.from_seconds(3.25)
    assert tc.integer == 3
    assert tc.fractional == pytest.approx(0.25)
    assert tc.total_seconds == pytest.approx(3.25)


def test_time_components_from_negative_seconds():
    tc = TimeComponents.from_seconds(-1.25)
    # Fractional part must be in [0, 1).
    assert 0.0 <= tc.fractional < 1.0
    assert tc.total_seconds == pytest.approx(-1.25)


def test_timing_calculator_round_trip():
    t_ref = datetime(2023, 5, 1, 6, 0, 0)
    tc = TimingCalculator(t_ref=t_ref, prf_hz=1717.0)

    t = datetime(2023, 5, 1, 6, 0, 5, 500_000)
    rel = tc.to_relative_seconds(t)
    assert rel == pytest.approx(5.5)

    comps = tc.to_time_components(t)
    assert comps.integer == 5
    assert comps.fractional == pytest.approx(0.5)

    back = tc.from_relative_seconds(rel)
    assert back == t


def test_timing_calculator_pulse_sequence():
    tc = TimingCalculator(
        t_ref=datetime(2023, 5, 1, 0, 0, 0),
        prf_hz=1000.0,
    )
    start = datetime(2023, 5, 1, 0, 0, 1)
    times = tc.compute_pulse_times(start, num_pulses=4)
    assert times.shape == (4,)
    assert times[0] == pytest.approx(1.0)
    assert times[1] == pytest.approx(1.001)
    assert times[-1] == pytest.approx(1.003)


def test_determine_reference_time_strips_microseconds():
    t = datetime(2023, 5, 1, 6, 0, 1, 123_456)
    t_ref = determine_reference_time(t)
    assert t_ref == datetime(2023, 5, 1, 6, 0, 1, 0)


# ===================================================================
# Binary parsers
# ===================================================================


def _write_index_records(
    path: Path, records: list[dict],
) -> None:
    """Write a synthetic burst index file."""
    with open(path, "wb") as f:
        for rec in records:
            flags = (
                rec.get("swath_bits", 0) << 1
            ) | (1 if rec.get("variable_size") else 0)
            f.write(struct.pack(
                ">ddIIQI",
                rec["reference_time"],
                rec["duration"],
                rec["isp_size"],
                rec["start_packet"],
                rec["byte_offset"],
                flags,
            ))


def _write_annot_records(
    path: Path, records: list[dict],
) -> None:
    """Write a synthetic packet annotation file."""
    with open(path, "wb") as f:
        for rec in records:
            flag_byte = (
                (0x80 if rec.get("vcid_present") else 0x00)
                | ((rec.get("vcid", 0) & 0x3F) << 1)
                | (
                    (rec.get("channel", 0) >> 1) & 0x01
                )
            )
            chan_byte = (rec.get("channel", 0) & 0x01) << 7
            f.write(struct.pack(
                ">HIHHIHHHHBBBB",
                rec["sens_days"],
                rec["sens_ms"],
                rec["sens_us"],
                rec["dl_days"],
                rec["dl_ms"],
                rec["dl_us"],
                rec["packet_length"],
                rec["frames"],
                rec.get("missing", 0),
                1 if rec.get("crc_error") else 0,
                flag_byte,
                chan_byte,
                0,  # spare
            ))


def test_binary_index_roundtrip(tmp_path):
    index_path = tmp_path / "s1a-iw-raw-vv-index.dat"
    _write_index_records(index_path, [
        {
            "reference_time": 12345.6789,
            "duration": 0.001,
            "isp_size": 17440,
            "start_packet": 0,
            "byte_offset": 0,
            "swath_bits": 10,
            "variable_size": False,
        },
        {
            "reference_time": 12345.8123,
            "duration": 0.002,
            "isp_size": 17440,
            "start_packet": 100,
            "byte_offset": 17440 * 100,
            "swath_bits": 10,
            "variable_size": True,
        },
    ])

    records = parse_burst_index_file(index_path)
    assert len(records) == 2
    assert records[0].burst_index == 0
    assert records[0].reference_time == pytest.approx(12345.6789)
    assert records[0].variable_size_flag is False
    assert records[1].variable_size_flag is True
    assert records[1].byte_offset == 17440 * 100
    assert records[1].swath_number == 10


def test_binary_annot_roundtrip(tmp_path):
    annot_path = tmp_path / "s1a-iw-raw-vv-annot.dat"
    _write_annot_records(annot_path, [
        {
            "sens_days": 100,
            "sens_ms": 43200000,
            "sens_us": 500,
            "dl_days": 100,
            "dl_ms": 43200100,
            "dl_us": 0,
            "packet_length": 17440,
            "frames": 1,
            "missing": 0,
            "crc_error": False,
            "vcid_present": True,
            "vcid": 7,
            "channel": 1,
        },
    ])

    records = parse_packet_annotation_file(annot_path)
    assert len(records) == 1
    r = records[0]
    assert r.sensing_time_days == 100
    assert r.sensing_time_ms == 43200000
    assert r.sensing_time_us == 500
    assert r.packet_length == 17440
    assert r.vcid_present is True
    assert r.vcid == 7


def test_binary_get_burst_byte_offsets(tmp_path):
    index_path = tmp_path / "s1a-iw-raw-vv-index.dat"
    _write_index_records(index_path, [
        {
            "reference_time": 0.0,
            "duration": 0.001,
            "isp_size": 17440,
            "start_packet": 0,
            "byte_offset": 0,
        },
        {
            "reference_time": 1.0,
            "duration": 0.001,
            "isp_size": 17440,
            "start_packet": 100,
            "byte_offset": 99999,
        },
    ])
    assert get_burst_byte_offsets(index_path) == [0, 99999]


def test_binary_find_burst_for_time():
    records = [
        BurstIndexRecord(
            burst_index=0,
            reference_time=10.0,
            duration=1.0,
        ),
        BurstIndexRecord(
            burst_index=1,
            reference_time=15.0,
            duration=1.0,
        ),
    ]
    assert find_burst_for_time(records, 10.5).burst_index == 0
    assert find_burst_for_time(records, 15.5).burst_index == 1
    assert find_burst_for_time(records, 20.0) is None


def test_binary_record_sizes_match_spec():
    assert INDEX_RECORD_SIZE == 36
    assert ANNOT_RECORD_SIZE == 26


# ===================================================================
# Annotation datetime parsing
# ===================================================================


@pytest.mark.parametrize("s, expected", [
    (
        "2023-05-01T06:00:01.123456Z",
        datetime(2023, 5, 1, 6, 0, 1, 123_456),
    ),
    (
        "2023-05-01T06:00:01.123456",
        datetime(2023, 5, 1, 6, 0, 1, 123_456),
    ),
    (
        "2023-05-01T06:00:01",
        datetime(2023, 5, 1, 6, 0, 1, 0),
    ),
])
def test_annotation_datetime_formats(s, expected):
    assert parse_annotation_datetime(s) == expected


def test_annotation_datetime_rejects_empty():
    with pytest.raises(ValueError):
        parse_annotation_datetime("")


# ===================================================================
# Annotation XML parsing
# ===================================================================


def _annotation_xml() -> str:
    """Build a minimal annotation XML with orbit / attitude / radar /
    swath timing / geolocation / downlink blocks."""
    return textwrap.dedent("""\
        <?xml version="1.0"?>
        <product>
          <generalAnnotation>
            <productInformation>
              <radarFrequency>5405000000.0</radarFrequency>
              <rangeSamplingRate>64345238.0</rangeSamplingRate>
              <azimuthSteeringRate>1.59</azimuthSteeringRate>
              <polarisation>VV</polarisation>
            </productInformation>
            <orbitList count="2">
              <orbit>
                <time>2023-05-01T06:00:00.000000</time>
                <position>
                  <x>7000000.0</x><y>0.0</y><z>0.0</z>
                </position>
                <velocity>
                  <x>0.0</x><y>7500.0</y><z>0.0</z>
                </velocity>
              </orbit>
              <orbit>
                <time>2023-05-01T06:00:10.000000</time>
                <position>
                  <x>7000100.0</x><y>75000.0</y><z>0.0</z>
                </position>
                <velocity>
                  <x>0.0</x><y>7500.0</y><z>0.0</z>
                </velocity>
              </orbit>
            </orbitList>
            <attitudeList count="1">
              <attitude>
                <time>2023-05-01T06:00:00.000000</time>
                <roll>0.1</roll>
                <pitch>0.2</pitch>
                <yaw>0.3</yaw>
              </attitude>
            </attitudeList>
            <downlinkInformation>
              <downlinkValues>
                <swath>IW1</swath>
                <prf>1717.0</prf>
                <txPulseLength>5.2e-5</txPulseLength>
                <txPulseRampRate>7.79038e11</txPulseRampRate>
                <rank>9</rank>
                <swst>0.005</swst>
                <swl>25000</swl>
              </downlinkValues>
            </downlinkInformation>
          </generalAnnotation>
          <swathTiming>
            <swath>IW1</swath>
            <linesPerBurst>1500</linesPerBurst>
            <samplesPerBurst>20000</samplesPerBurst>
            <azimuthTimeInterval>0.002055</azimuthTimeInterval>
            <burstList count="2">
              <burst>
                <azimuthTime>2023-05-01T06:00:01.000000</azimuthTime>
                <sensingTime>2023-05-01T06:00:01.000000</sensingTime>
                <byteOffset>0</byteOffset>
                <azimuthAnxTime>1000.0</azimuthAnxTime>
                <firstValidSample>10 10 10</firstValidSample>
                <lastValidSample>19980 19990 19995</lastValidSample>
              </burst>
              <burst>
                <azimuthTime>2023-05-01T06:00:03.800000</azimuthTime>
                <sensingTime>2023-05-01T06:00:03.800000</sensingTime>
                <byteOffset>99999</byteOffset>
                <azimuthAnxTime>1002.8</azimuthAnxTime>
                <firstValidSample>10 10 10</firstValidSample>
                <lastValidSample>19980 19990 19995</lastValidSample>
              </burst>
            </burstList>
          </swathTiming>
          <geolocationGrid>
            <geolocationGridPointList>
              <geolocationGridPoint>
                <azimuthTime>2023-05-01T06:00:01.000000</azimuthTime>
                <slantRangeTime>5.0e-3</slantRangeTime>
                <latitude>30.0</latitude>
                <longitude>50.0</longitude>
                <height>0.0</height>
              </geolocationGridPoint>
              <geolocationGridPoint>
                <azimuthTime>2023-05-01T06:00:01.000000</azimuthTime>
                <slantRangeTime>5.5e-3</slantRangeTime>
                <latitude>30.0</latitude>
                <longitude>50.5</longitude>
                <height>0.0</height>
              </geolocationGridPoint>
            </geolocationGridPointList>
          </geolocationGrid>
        </product>
    """)


def test_annotation_parser_full(tmp_path):
    xml_path = (
        tmp_path / "s1a-iw-raw-vv-20230501t060001-annot.xml"
    )
    xml_path.write_text(_annotation_xml())

    parser = AnnotationParser()
    data = parser.parse(xml_path)

    assert data.polarization == "VV"
    assert len(data.orbit_state_vectors) == 2
    assert data.orbit_state_vectors[0].x == 7_000_000.0
    assert data.orbit_state_vectors[0].vy == 7500.0

    assert len(data.attitude_records) == 1
    assert data.attitude_records[0].roll == 0.1

    rp = data.radar_parameters
    assert rp.center_frequency_hz == 5_405_000_000.0
    assert rp.range_sampling_rate_hz == 64_345_238.0
    # PRF merged from downlink info.
    assert rp.pulse_repetition_frequency_hz == 1717.0

    assert len(data.downlink_info) == 1
    assert data.downlink_info[0].swath == "IW1"
    assert data.downlink_info[0].rank == 9

    assert len(data.swath_parameters) == 1
    sp = data.swath_parameters[0]
    assert sp.swath_id == "IW1"
    assert sp.num_bursts == 2
    assert sp.lines_per_burst == 1500
    assert sp.samples_per_burst == 20_000
    assert len(sp.bursts) == 2
    assert sp.bursts[0].byte_offset == 0
    assert sp.bursts[1].byte_offset == 99999
    assert sp.bursts[0].first_valid_sample == 10
    assert sp.bursts[0].last_valid_sample == 19995

    grid = data.geolocation_grid
    assert grid is not None
    assert len(grid.azimuth_times) == 2
    assert grid.latitudes.shape == (2,)


def test_annotation_parser_convenience(tmp_path):
    """``parse_annotation_file`` and ``parse_all_annotations``."""
    xml = _annotation_xml()
    vv_path = tmp_path / "s1a-iw-raw-vv-annot.xml"
    vh_path = tmp_path / "s1a-iw-raw-vh-annot.xml"
    vv_path.write_text(xml)
    vh_path.write_text(xml.replace(">VV<", ">VH<"))

    single = parse_annotation_file(vv_path)
    assert single.polarization == "VV"

    both = parse_all_annotations([vv_path, vh_path])
    assert set(both.keys()) == {"VV", "VH"}

    merged = merge_annotation_data(both)
    # Orbit is shared; taken from first entry.
    assert len(merged.orbit_state_vectors) == 2
    # Swath parameters de-duplicated by swath_id.
    assert len(merged.swath_parameters) == 1


# ===================================================================
# Product name parsing
# ===================================================================


def test_parse_product_name_dual_vv_vh():
    name = (
        "S1A_IW_RAW__0SDV_20230501T060001_20230501T060101_"
        "048123_05C7E2_F3B9.SAFE"
    )
    pid = parse_product_name(name)
    assert pid is not None
    assert pid.mission == "A"
    assert pid.mode == "IW"
    assert pid.polarization_code == "DV"
    assert pid.polarizations == ["VV", "VH"]
    assert pid.is_dual_pol
    assert pid.platform_name == "Sentinel-1A"
    assert pid.orbit_number == 48_123


def test_parse_product_name_rejects_invalid():
    assert parse_product_name("not-a-safe") is None


# ===================================================================
# Manifest XML parsing
# ===================================================================


_MANIFEST_XML = textwrap.dedent("""\
    <?xml version="1.0"?>
    <xfdu:XFDU
        xmlns:xfdu="urn:ccsds:schema:xfdu:1"
        xmlns:safe="http://www.esa.int/safe/sentinel-1.0"
        xmlns:s1="http://www.esa.int/safe/sentinel-1.0/sentinel-1"
        xmlns:s1sarl0=\
"http://www.esa.int/safe/sentinel-1.0/sentinel-1/sar/level-0"
        xmlns:gml="http://www.opengis.net/gml">
      <metadataSection>
        <metadataObject ID="platform">
          <metadataWrap>
            <xmlData>
              <safe:platform>
                <safe:nssdcIdentifier>2014-016A</safe:nssdcIdentifier>
                <safe:familyName>SENTINEL-1</safe:familyName>
                <safe:number>A</safe:number>
                <safe:instrument>
                  <safe:familyName>SENTINEL-1 C-SAR</safe:familyName>
                  <safe:abbreviation>C-SAR</safe:abbreviation>
                  <safe:extension>
                    <s1sarl0:instrumentMode>
                      <s1sarl0:mode>IW</s1sarl0:mode>
                      <s1sarl0:swath>IW</s1sarl0:swath>
                    </s1sarl0:instrumentMode>
                  </safe:extension>
                </safe:instrument>
              </safe:platform>
            </xmlData>
          </metadataWrap>
        </metadataObject>
        <metadataObject ID="orbit">
          <metadataWrap>
            <xmlData>
              <safe:orbitReference>
                <safe:orbitNumber type="start">48123</safe:orbitNumber>
                <safe:orbitNumber type="stop">48123</safe:orbitNumber>
                <safe:relativeOrbitNumber type="start">51</safe:relativeOrbitNumber>
                <safe:relativeOrbitNumber type="stop">51</safe:relativeOrbitNumber>
                <safe:cycleNumber>100</safe:cycleNumber>
                <safe:phaseIdentifier>1</safe:phaseIdentifier>
                <safe:extension>
                  <s1:orbitProperties>
                    <s1:pass>ASCENDING</s1:pass>
                    <s1:ascendingNodeTime>\
2023-05-01T05:55:00.000000Z</s1:ascendingNodeTime>
                  </s1:orbitProperties>
                </safe:extension>
              </safe:orbitReference>
            </xmlData>
          </metadataWrap>
        </metadataObject>
        <metadataObject ID="processing">
          <metadataWrap>
            <xmlData>
              <safe:processing name="Level-0 Processing"
                               start="2023-05-01T06:10:00.000000Z"
                               stop="2023-05-01T06:12:00.000000Z"
                               site="ESRIN">
                <safe:facility country="Italy" name="ESRIN"
                               organisation="ESA" site="ESRIN"/>
                <safe:software name="ipf" version="3.61"/>
              </safe:processing>
            </xmlData>
          </metadataWrap>
        </metadataObject>
        <metadataObject ID="acq">
          <metadataWrap>
            <xmlData>
              <safe:acquisitionPeriod>
                <safe:startTime>2023-05-01T06:00:01.000000Z</safe:startTime>
                <safe:stopTime>2023-05-01T06:01:01.000000Z</safe:stopTime>
              </safe:acquisitionPeriod>
            </xmlData>
          </metadataWrap>
        </metadataObject>
        <metadataObject ID="gpi">
          <metadataWrap>
            <xmlData>
              <s1sarl0:standAloneProductInformation>
                <s1sarl0:productClass>S</s1sarl0:productClass>
                <s1sarl0:productClassDescription>\
SAR Standard L0 Product</s1sarl0:productClassDescription>
                <s1sarl0:productConsolidation>SLICE\
</s1sarl0:productConsolidation>
                <s1sarl0:transmitterReceiverPolarisation>VV\
</s1sarl0:transmitterReceiverPolarisation>
                <s1sarl0:transmitterReceiverPolarisation>VH\
</s1sarl0:transmitterReceiverPolarisation>
                <s1sarl0:instrumentConfigurationID>7\
</s1sarl0:instrumentConfigurationID>
                <s1sarl0:echoCompressionType>FDBAQ\
</s1sarl0:echoCompressionType>
                <s1sarl0:sliceProductFlag>true\
</s1sarl0:sliceProductFlag>
                <s1sarl0:sliceNumber>3</s1sarl0:sliceNumber>
                <s1sarl0:totalSlices>6</s1sarl0:totalSlices>
              </s1sarl0:standAloneProductInformation>
            </xmlData>
          </metadataWrap>
        </metadataObject>
      </metadataSection>
    </xfdu:XFDU>
""")


def test_parse_manifest_extracts_platform_and_orbit(tmp_path):
    manifest = tmp_path / "manifest.safe"
    manifest.write_text(_MANIFEST_XML)

    info = parse_manifest(manifest)

    assert info.platform_number == "A"
    assert info.instrument_abbreviation == "C-SAR"
    assert info.instrument_mode == "IW"
    assert info.orbit_number_start == 48_123
    assert info.relative_orbit == 51
    assert info.is_ascending
    assert info.processing_center == "ESRIN"
    assert info.software_name == "ipf"
    assert info.acquisition_period_start is not None
    assert info.product_class == "S"
    assert info.transmitter_receiver_pol == ["VV", "VH"]
    assert info.echo_compression_type == "FDBAQ"
    assert info.slice_number == 3


def test_parse_manifest_raises_on_invalid(tmp_path):
    manifest = tmp_path / "manifest.safe"
    manifest.write_text("<bad-xml>")
    with pytest.raises(InvalidSAFEProductError):
        parse_manifest(manifest)


# ===================================================================
# SAFEProduct end-to-end
# ===================================================================


def _make_fake_safe(
    tmp_path: Path,
    polarizations: list[str] = None,
) -> Path:
    """Build a minimal but valid SAFE directory structure."""
    polarizations = polarizations or ["VV", "VH"]
    safe_dir = tmp_path / (
        "S1A_IW_RAW__0SDV_20230501T060001_20230501T060101_"
        "048123_05C7E2_F3B9.SAFE"
    )
    safe_dir.mkdir()

    # manifest.safe
    (safe_dir / "manifest.safe").write_text(_MANIFEST_XML)

    meas_dir = safe_dir / "measurement"
    ann_dir = safe_dir / "annotation"
    meas_dir.mkdir()
    ann_dir.mkdir()

    for pol in polarizations:
        stem = (
            "s1a-iw-raw-{p}-20230501t060001"
            "-20230501t060101-048123-05c7e2-001"
        ).format(p=pol.lower())
        # Raw .dat (empty file — reader doesn't read here).
        (meas_dir / f"{stem}.dat").write_bytes(b"")
        # Index file with one record.
        _write_index_records(
            meas_dir / f"{stem}-index.dat",
            [{
                "reference_time": 0.0,
                "duration": 0.001,
                "isp_size": 17440,
                "start_packet": 0,
                "byte_offset": 0,
            }],
        )
        # Annotation file with one record.
        _write_annot_records(
            meas_dir / f"{stem}-annot.dat",
            [{
                "sens_days": 100,
                "sens_ms": 43200000,
                "sens_us": 0,
                "dl_days": 100,
                "dl_ms": 43200100,
                "dl_us": 0,
                "packet_length": 17440,
                "frames": 1,
            }],
        )
        # Annotation XML.
        (
            ann_dir / f"s1a-iw-raw-{pol.lower()}-annot.xml"
        ).write_text(_annotation_xml().replace(
            ">VV<", f">{pol}<"
        ))

    return safe_dir


def test_safeproduct_discovery(tmp_path):
    safe = _make_fake_safe(tmp_path)
    product = SAFEProduct(safe)

    assert product.product_info is not None
    assert product.product_info.mission == "A"
    assert product.polarizations == ["VV", "VH"]
    assert len(product.measurement_files) == 2
    assert len(product.annotation_files) == 2

    vv_meas = product.get_measurement_file("VV")
    assert vv_meas is not None
    assert vv_meas.has_index
    assert vv_meas.has_annotations
    assert vv_meas.swath == "IW"

    vv_annot = product.get_annotation_file("VV")
    assert vv_annot is not None
    assert "vv" in vv_annot.name.lower()


def test_safeproduct_validate_missing_manifest(tmp_path):
    bad_safe = tmp_path / (
        "S1A_IW_RAW__0SDV_20230501T060001_20230501T060101_"
        "048123_05C7E2_F3B9.SAFE"
    )
    bad_safe.mkdir()
    (bad_safe / "measurement").mkdir()
    with pytest.raises(
        InvalidSAFEProductError, match="Missing manifest",
    ):
        SAFEProduct(bad_safe)


def test_safeproduct_summary_is_readable(tmp_path):
    safe = _make_fake_safe(tmp_path, polarizations=["VV"])
    product = SAFEProduct(safe)
    summary = product.summary()
    assert "SAFEProduct" in summary
    assert "Sentinel-1A" in summary
    assert "VV" in summary


# ===================================================================
# Metadata container integration
# ===================================================================


def test_sentinel1_l0_metadata_inherits_image_metadata():
    from grdl.IO.models.base import ImageMetadata
    m = Sentinel1L0Metadata(
        format="Sentinel-1 L0",
        rows=0, cols=0, dtype="complex64",
        product_id="S1A_IW_RAW__0SDV_test",
        mission=Sentinel1Mission.S1A,
        mode=Sentinel1Mode.IW,
        polarizations=["VV", "VH"],
    )
    assert isinstance(m, ImageMetadata)
    assert m.polarization == "DV"
    assert m.get_reference_time() is None

    m.start_time = datetime(2023, 5, 1, 6, 0, 1, 500_000)
    assert m.get_reference_time() == datetime(2023, 5, 1, 6, 0, 1)


def test_s1l0_orbit_state_vector_geodetic():
    ov = S1L0OrbitStateVector(
        time=datetime(2024, 1, 1),
        x=6_378_137.0, y=0.0, z=0.0,
        vx=0.0, vy=7500.0, vz=0.0,
    )
    lat, lon, alt = ov.geodetic
    assert abs(lat) < 1e-6
    assert abs(lon) < 1e-6
    assert abs(alt) < 1e-3
    assert ov.speed == pytest.approx(7500.0)


def test_s1l0_attitude_rotation_matrix():
    a = S1L0AttitudeRecord(
        time=datetime(2024, 1, 1),
        roll=0.0, pitch=0.0, yaw=0.0,
    )
    R = a.to_rotation_matrix()
    # Zero Euler angles ⇒ identity.
    assert R.shape == (3, 3)
    import numpy as np
    assert np.allclose(R, np.eye(3), atol=1e-12)


# ===================================================================
# Stage 3 — Orbit interpolation & geometry
# ===================================================================


def _make_circular_orbit_vectors(
    n: int = 10,
    t0: datetime = datetime(2023, 5, 1, 6, 0, 0),
    radius: float = 7_071_000.0,
    speed: float = 7500.0,
) -> list:
    """Build N orbit state vectors along a circular equatorial
    orbit (equispaced 1 s apart)."""
    from datetime import timedelta
    vectors = []
    omega = speed / radius  # rad/s
    for i in range(n):
        t = t0 + timedelta(seconds=i)
        phi = omega * i
        vectors.append(S1L0OrbitStateVector(
            time=t,
            x=radius * np.cos(phi),
            y=radius * np.sin(phi),
            z=0.0,
            vx=-speed * np.sin(phi),
            vy=speed * np.cos(phi),
            vz=0.0,
        ))
    return vectors


def test_orbit_interpolator_interpolate_single():
    import numpy as np
    from grdl.IO.sar.sentinel1_l0.orbit import OrbitInterpolator
    vectors = _make_circular_orbit_vectors(n=10)
    t_ref = vectors[0].time
    interp = OrbitInterpolator(vectors, t_ref)

    # At a sample point, interpolation should equal the vector.
    pos, vel = interp.interpolate_single(5.0)
    assert pos == pytest.approx(
        [vectors[5].x, vectors[5].y, vectors[5].z],
        rel=1e-9,
    )
    assert vel == pytest.approx(
        [vectors[5].vx, vectors[5].vy, vectors[5].vz],
        rel=1e-9,
    )


def test_orbit_interpolator_vectorized():
    import numpy as np
    from grdl.IO.sar.sentinel1_l0.orbit import OrbitInterpolator
    vectors = _make_circular_orbit_vectors(n=10)
    interp = OrbitInterpolator(vectors, vectors[0].time)

    times = np.linspace(0.0, 9.0, 20)
    pos, vel = interp.interpolate(times)
    assert pos.shape == (20, 3)
    assert vel.shape == (20, 3)
    # Radius should stay close to the orbital radius.
    radii = np.linalg.norm(pos, axis=1)
    assert np.allclose(radii, 7_071_000.0, rtol=1e-6)


def test_orbit_interpolator_rejects_too_few_vectors():
    from grdl.IO.sar.sentinel1_l0.orbit import OrbitInterpolator
    vectors = _make_circular_orbit_vectors(n=3)
    with pytest.raises(ValueError, match="at least 4"):
        OrbitInterpolator(vectors, vectors[0].time)


def test_attitude_interpolator_linear():
    import numpy as np
    from datetime import timedelta
    from grdl.IO.sar.sentinel1_l0.orbit import (
        AttitudeInterpolator,
    )
    t0 = datetime(2023, 5, 1, 6, 0, 0)
    records = [
        S1L0AttitudeRecord(
            time=t0 + timedelta(seconds=i),
            roll=float(i),
            pitch=2.0 * i,
            yaw=3.0 * i,
        )
        for i in range(6)
    ]
    interp = AttitudeInterpolator(records, t0)
    times = np.array([0.0, 2.5, 5.0])
    roll, pitch, yaw = interp.interpolate(times)
    assert roll[0] == pytest.approx(0.0)
    assert roll[-1] == pytest.approx(5.0, rel=1e-9)
    # Cubic spline of linear data reproduces linearity exactly.
    assert roll[1] == pytest.approx(2.5, rel=1e-9)
    assert pitch[1] == pytest.approx(5.0, rel=1e-9)
    assert yaw[1] == pytest.approx(7.5, rel=1e-9)


def test_poe_parser_roundtrip(tmp_path):
    from grdl.IO.sar.sentinel1_l0.orbit import (
        POEParser,
        parse_poe_file,
    )
    # Build a minimal POE XML document with 3 OSVs.
    poe_xml = textwrap.dedent("""\
        <?xml version="1.0"?>
        <Earth_Explorer_File>
          <Earth_Explorer_Header>
            <Fixed_Header>
              <Mission>Sentinel-1A</Mission>
              <Validity_Period>
                <Validity_Start>UTC=2023-05-01T00:00:00\
</Validity_Start>
                <Validity_Stop>UTC=2023-05-02T00:00:00\
</Validity_Stop>
              </Validity_Period>
            </Fixed_Header>
          </Earth_Explorer_Header>
          <Data_Block>
            <List_of_OSVs>
              <OSV>
                <UTC>UTC=2023-05-01T06:00:00.000000</UTC>
                <X unit="m">7000000.0</X>
                <Y unit="m">0.0</Y>
                <Z unit="m">0.0</Z>
                <VX unit="m/s">0.0</VX>
                <VY unit="m/s">7500.0</VY>
                <VZ unit="m/s">0.0</VZ>
              </OSV>
              <OSV>
                <UTC>UTC=2023-05-01T06:00:10.000000</UTC>
                <X unit="m">7000100.0</X>
                <Y unit="m">75000.0</Y>
                <Z unit="m">0.0</Z>
                <VX unit="m/s">0.0</VX>
                <VY unit="m/s">7500.0</VY>
                <VZ unit="m/s">0.0</VZ>
              </OSV>
              <OSV>
                <UTC>UTC=2023-05-01T06:00:20.000000</UTC>
                <X unit="m">7000200.0</X>
                <Y unit="m">150000.0</Y>
                <Z unit="m">0.0</Z>
                <VX unit="m/s">0.0</VX>
                <VY unit="m/s">7500.0</VY>
                <VZ unit="m/s">0.0</VZ>
              </OSV>
            </List_of_OSVs>
          </Data_Block>
        </Earth_Explorer_File>
    """)
    poe_path = tmp_path / (
        "S1A_OPER_AUX_POEORB_OPOD_20230503T000000_"
        "V20230501T000000_20230502T000000.EOF"
    )
    poe_path.write_text(poe_xml)

    parser = POEParser(poe_path)
    vectors = parser.parse()

    assert len(vectors) == 3
    assert parser.info.mission == "Sentinel-1A"
    assert parser.info.num_vectors == 3
    assert parser.info.covers_time(
        datetime(2023, 5, 1, 12, 0, 0)
    )
    assert not parser.info.covers_time(
        datetime(2024, 1, 1)
    )
    assert vectors[0].y == 0.0
    assert vectors[1].vy == 7500.0

    # Convenience wrapper.
    assert len(parse_poe_file(poe_path)) == 3


def test_orbit_loader_prefers_poe():
    from grdl.IO.sar.sentinel1_l0.orbit import OrbitLoader
    loader = OrbitLoader()
    ann = _make_circular_orbit_vectors(n=5)
    # Directly populate the internal lists — simulates
    # ``load_annotation_vectors`` + in-memory POE.
    loader.load_annotation_vectors(ann)
    assert loader.has_annotation_orbit
    assert not loader.has_poe_orbit
    assert loader.get_vectors() == ann

    # Simulate POE data by populating the internal attribute.
    poe_vectors = _make_circular_orbit_vectors(n=8)
    loader._poe_vectors = poe_vectors
    assert loader.has_poe_orbit
    # prefer_poe default: returns POE.
    assert loader.get_vectors() == poe_vectors
    # Explicit opt-out.
    assert loader.get_vectors(prefer_poe=False) == ann


def test_geometry_state_vector_basics():
    import numpy as np
    from grdl.IO.sar.sentinel1_l0.geometry import StateVector
    sv = StateVector(
        position=np.array([6_378_137.0, 0.0, 0.0]),
        velocity=np.array([0.0, 7500.0, 0.0]),
        time_offset=1.5,
    )
    assert sv.x == pytest.approx(6_378_137.0)
    assert sv.speed == pytest.approx(7500.0)
    assert sv.altitude == pytest.approx(0.0)
    assert sv.time_offset == 1.5


def test_geometry_acf_vectors_normalized():
    import numpy as np
    from grdl.IO.sar.sentinel1_l0.geometry import ACFVectors
    acf = ACFVectors(
        acx=np.array([2.0, 0.0, 0.0]),  # not unit
        acy=np.array([0.0, 3.0, 0.0]),  # not unit
    )
    assert np.linalg.norm(acf.acx) == pytest.approx(1.0)
    assert np.linalg.norm(acf.acy) == pytest.approx(1.0)


def test_geometry_compute_acf_vectors_batch_right_looking():
    import numpy as np
    from grdl.IO.sar.sentinel1_l0.geometry import (
        compute_acf_vectors_batch,
    )
    # Platform on +X axis moving along +Y.
    positions = np.array([[7_071_000.0, 0.0, 0.0]])
    velocities = np.array([[0.0, 7500.0, 0.0]])
    acx, acy = compute_acf_vectors_batch(
        positions, velocities,
        incidence_deg=30.0, look_side="right",
    )
    # ACX points along-track (velocity direction) → +Y.
    assert acx[0] == pytest.approx([0.0, 1.0, 0.0], abs=1e-9)
    # ACY should be a unit vector.
    assert np.linalg.norm(acy[0]) == pytest.approx(
        1.0, rel=1e-9,
    )


def test_geometry_swath_incidence_lookup():
    from grdl.IO.sar.sentinel1_l0.geometry import (
        get_mode_mid_incidence_deg,
        get_swath_incidence_deg,
    )
    assert get_swath_incidence_deg("IW1") == 33.0
    assert get_swath_incidence_deg("IW2") == 38.0
    assert get_swath_incidence_deg("EW3") == 37.0
    # Unknown → fallback.
    assert get_swath_incidence_deg("XX") == 30.0
    # Mode lookups.
    assert get_mode_mid_incidence_deg("IW") == 38.0
    assert get_mode_mid_incidence_deg("SM") == 35.0


def test_geometry_slant_range_and_incidence():
    from grdl.IO.sar.sentinel1_l0.geometry import (
        compute_incidence_from_timing,
        compute_slant_range,
    )
    # Round-trip time of 7 ms → slant range ~1048 km.
    sr = compute_slant_range(
        rank=0, pri_seconds=0.0, swst_seconds=7e-3,
    )
    assert sr == pytest.approx(
        0.5 * 299_792_458.0 * 7e-3, rel=1e-9,
    )

    # Any non-degenerate timing/altitude should produce a
    # physical incidence angle in (0°, 90°).  The exact value
    # depends on all four parameters — this test just verifies
    # the spherical-Earth math stays in range.
    inc = compute_incidence_from_timing(
        rank=8,
        pri_seconds=1.0 / 1717.0,
        swst_seconds=0.005,
        satellite_altitude=693_000.0,
    )
    assert 0.0 < inc < 90.0


def test_geometry_local_enu_basis_equator():
    import numpy as np
    from grdl.IO.sar.sentinel1_l0.geometry import (
        compute_local_enu_basis,
    )
    east, north, up = compute_local_enu_basis(0.0, 0.0)
    # At (0°,0°): east = +Y, north = +Z, up = +X.
    assert east == pytest.approx([0.0, 1.0, 0.0], abs=1e-12)
    assert north == pytest.approx([0.0, 0.0, 1.0], abs=1e-12)
    assert up == pytest.approx([1.0, 0.0, 0.0], abs=1e-12)


def test_geometry_calculator_end_to_end():
    import numpy as np
    from grdl.IO.sar.sentinel1_l0.geometry import (
        GeometryCalculator,
    )
    vectors = _make_circular_orbit_vectors(n=10)
    geom = GeometryCalculator(
        vectors, reference_time=vectors[0].time,
    )
    assert not geom.has_attitude
    assert geom.valid_start == 0.0
    assert geom.valid_end == 9.0

    times = np.array([0.0, 4.5, 9.0])
    positions, velocities, acx, acy = (
        geom.get_state_vectors_at_times(
            times, look_side="right",
        )
    )
    assert positions.shape == (3, 3)
    assert velocities.shape == (3, 3)
    assert acx.shape == (3, 3)
    assert acy.shape == (3, 3)
    # ACY rows should be unit vectors.
    assert np.allclose(
        np.linalg.norm(acy, axis=1), 1.0, atol=1e-9,
    )


# ===================================================================
# Stage 4 — Decoder and burst reader
# ===================================================================


def test_decoder_module_imports_without_crash():
    """The decoder module must load even if the optional
    ``sentinel1decoder`` package is absent."""
    from grdl.IO.sar.sentinel1_l0 import decoder
    # Either available or not — both are valid states.
    assert isinstance(decoder.check_decoder_available(), bool)
    # Module-level sentinel is exposed.
    assert hasattr(decoder, "_HAS_S1_DECODER")


def test_decoder_require_raises_dependency_error(monkeypatch):
    """When ``sentinel1decoder`` is unavailable, calling
    ``require_decoder`` raises :class:`DependencyError` with an
    install hint."""
    from grdl.exceptions import DependencyError
    from grdl.IO.sar.sentinel1_l0 import decoder

    monkeypatch.setattr(decoder, "_HAS_S1_DECODER", False)
    with pytest.raises(
        DependencyError, match=r"grdl\[s1_l0\]"
    ):
        decoder.require_decoder()


def test_decoder_construction_raises_without_package(
    monkeypatch, tmp_path,
):
    """Constructing a ``Sentinel1Decoder`` without the optional
    package raises :class:`DependencyError` before touching the
    filesystem."""
    from grdl.exceptions import DependencyError
    from grdl.IO.sar.sentinel1_l0 import decoder

    monkeypatch.setattr(decoder, "_HAS_S1_DECODER", False)
    fake_file = tmp_path / "measurement.dat"
    fake_file.write_bytes(b"")

    with pytest.raises(DependencyError):
        decoder.Sentinel1Decoder(fake_file)


def test_decoder_missing_file(tmp_path):
    """``Sentinel1Decoder`` raises FileNotFoundError when the
    measurement file is missing (only relevant when the optional
    package is installed)."""
    from grdl.IO.sar.sentinel1_l0 import decoder

    if not decoder.check_decoder_available():
        pytest.skip("sentinel1decoder not installed")

    with pytest.raises(FileNotFoundError):
        decoder.Sentinel1Decoder(
            tmp_path / "does_not_exist.dat"
        )


def test_packet_metadata_dataclass():
    from grdl.IO.sar.sentinel1_l0.decoder import PacketMetadata
    pm = PacketMetadata(
        packet_index=5,
        coarse_time=1_700_000_000,
        fine_time=32_768,
        swath_number=11,
        polarization=2,
        num_quads=12_000,
    )
    assert pm.packet_index == 5
    assert pm.num_quads == 12_000
    assert pm.baq_mode == 0  # default


def test_burst_info_num_packets_and_start_time():
    from grdl.IO.sar.sentinel1_l0.burst_reader import BurstInfo

    bi = BurstInfo(
        burst_index=3,
        swath=10,
        polarization=0,
        start_packet=100,
        end_packet=250,
        num_lines=150,
        num_samples=20_000,
        reference_time=0.0,
    )
    assert bi.num_packets == 150
    # reference_time of 0 ⇒ no UTC conversion performed.
    assert bi.start_time is None

    # GPS epoch + 86400 s = Jan 7 1980 00:00:00 GPS,
    # which is Jan 7 1980 minus leap seconds in UTC.
    bi_t = BurstInfo(
        burst_index=0,
        swath=10,
        polarization=0,
        start_packet=0,
        end_packet=1,
        reference_time=86400.0,
    )
    start = bi_t.start_time
    assert start is not None
    # GPS start of day on 1980-01-07, UTC is earlier by
    # leap-seconds (18 as of 2025).
    from datetime import datetime, timedelta
    expected = (
        datetime(1980, 1, 7, 0, 0, 0)
        - timedelta(seconds=18)
    )
    assert start == expected


def test_swath_info_defaults():
    from grdl.IO.sar.sentinel1_l0.burst_reader import SwathInfo
    si = SwathInfo(swath=11, polarization=0)
    assert si.num_bursts == 0
    assert si.bursts == []
    assert si.start_packet == 0
    assert si.end_packet == 0


def test_burst_reader_construction_without_decoder(
    monkeypatch, tmp_path,
):
    """Without the optional decoder, ``BurstReader`` still
    constructs, just without a live decoder.  It can still
    return index-based bursts if an index file exists."""
    from grdl.IO.sar.sentinel1_l0 import decoder as dec_mod
    from grdl.IO.sar.sentinel1_l0.burst_reader import BurstReader

    # Force decoder-unavailable path for construction.
    monkeypatch.setattr(dec_mod, "_HAS_S1_DECODER", False)

    # Synthetic .dat + -index.dat pair.
    stem = "s1a-iw-raw-vv-test"
    meas = tmp_path / f"{stem}.dat"
    meas.write_bytes(b"")
    index = tmp_path / f"{stem}-index.dat"
    _write_index_records(index, [
        {
            "reference_time": 100.0,
            "duration": 0.001,
            "isp_size": 17440,
            "start_packet": 0,
            "byte_offset": 0,
        },
        {
            "reference_time": 200.0,
            "duration": 0.001,
            "isp_size": 17440,
            "start_packet": 50,
            "byte_offset": 12345,
        },
    ])

    reader = BurstReader(meas)
    assert reader.has_index
    assert reader.num_packets == 0  # no decoder

    bursts = reader.get_burst_info()
    # Without decoder, we fall back to index-only bursts.
    assert len(bursts) == 2
    assert bursts[0].byte_offset == 0
    assert bursts[1].byte_offset == 12345

    # read_burst must raise without decoder.
    with pytest.raises(RuntimeError, match="Decoder"):
        reader.read_burst(0)


def test_burst_reader_respects_custom_thresholds(
    monkeypatch, tmp_path,
):
    """Constructor overrides for burst-gap threshold and line
    filter ratio are honored."""
    from grdl.IO.sar.sentinel1_l0 import decoder as dec_mod
    from grdl.IO.sar.sentinel1_l0.burst_reader import BurstReader

    monkeypatch.setattr(dec_mod, "_HAS_S1_DECODER", False)

    meas = tmp_path / "s1a-iw-raw-vv-test.dat"
    meas.write_bytes(b"")

    reader = BurstReader(
        meas,
        burst_gap_threshold_us=500_000,
        burst_line_filter_ratio=0.5,
    )
    assert reader._burst_gap_threshold_us == 500_000
    assert reader._burst_line_filter_ratio == 0.5


# ===================================================================
# Stage 5 — Sentinel1L0Reader (public API)
# ===================================================================


def _make_l0_safe_no_decoder(
    tmp_path: Path,
    polarizations: list = None,
) -> Path:
    """Build a minimal SAFE structure suitable for exercising the
    public reader's metadata path *without* the optional decoder
    being active."""
    polarizations = polarizations or ["VV", "VH"]
    safe_dir = tmp_path / (
        "S1A_IW_RAW__0SDV_20230501T060001_20230501T060101_"
        "048123_05C7E2_F3B9.SAFE"
    )
    safe_dir.mkdir()
    (safe_dir / "manifest.safe").write_text(_MANIFEST_XML)

    meas_dir = safe_dir / "measurement"
    ann_dir = safe_dir / "annotation"
    meas_dir.mkdir()
    ann_dir.mkdir()

    for pol in polarizations:
        stem = (
            f"s1a-iw-raw-{pol.lower()}-20230501t060001"
            "-20230501t060101-048123-05c7e2-001"
        )
        (meas_dir / f"{stem}.dat").write_bytes(b"")
        _write_index_records(
            meas_dir / f"{stem}-index.dat",
            [{
                "reference_time": 0.0,
                "duration": 0.001,
                "isp_size": 17440,
                "start_packet": 0,
                "byte_offset": 0,
            }],
        )
        (
            ann_dir / f"s1a-iw-raw-{pol.lower()}-annot.xml"
        ).write_text(_annotation_xml().replace(
            ">VV<", f">{pol}<"
        ))
    return safe_dir


def test_reader_is_imagereader_subclass():
    from grdl.IO.base import ImageReader
    from grdl.IO.sar.sentinel1_l0.reader import (
        Sentinel1L0Reader,
    )
    assert issubclass(Sentinel1L0Reader, ImageReader)


def test_reader_load_metadata(monkeypatch, tmp_path):
    """End-to-end metadata load on a synthetic SAFE (decoder
    disabled, so no burst-readers — metadata still populates
    from annotation XML and manifest)."""
    from grdl.IO.sar.sentinel1_l0 import decoder as dec_mod
    from grdl.IO.sar.sentinel1_l0.reader import (
        Sentinel1L0Reader,
    )

    monkeypatch.setattr(dec_mod, "_HAS_S1_DECODER", False)

    safe = _make_l0_safe_no_decoder(tmp_path)
    reader = Sentinel1L0Reader(safe)
    try:
        m = reader.metadata
        assert isinstance(m, Sentinel1L0Metadata)
        assert m.product_id.startswith("S1A_IW_RAW__0SDV_")
        assert m.mission == Sentinel1Mission.S1A
        assert m.mode == Sentinel1Mode.IW
        assert set(m.polarizations) == {"VV", "VH"}
        # Manifest gave sub-second precision.
        assert m.start_time == datetime(
            2023, 5, 1, 6, 0, 1,
        )
        # Orbit + attitude came from annotation XML.
        assert len(m.orbit_state_vectors) == 2
        assert len(m.attitude_records) == 1
        assert m.radar_parameters is not None
        assert (
            m.radar_parameters.pulse_repetition_frequency_hz
            == 1717.0
        )
        # Swath parameters merged from annotation.
        assert len(m.swath_parameters) == 1
        sp = m.swath_parameters[0]
        assert sp.swath_id == "IW1"
    finally:
        reader.close()


def test_reader_context_manager(monkeypatch, tmp_path):
    from grdl.IO.sar.sentinel1_l0 import decoder as dec_mod
    from grdl.IO.sar.sentinel1_l0.reader import (
        Sentinel1L0Reader,
    )

    monkeypatch.setattr(dec_mod, "_HAS_S1_DECODER", False)
    safe = _make_l0_safe_no_decoder(tmp_path)

    with Sentinel1L0Reader(safe) as reader:
        assert reader.product is not None
        # summary() returns non-empty string.
        assert "Sentinel1L0Reader" in reader.summary()


def test_reader_read_chip_without_decoder(
    monkeypatch, tmp_path,
):
    """Without ``sentinel1decoder``, ``read_chip`` raises
    :class:`DependencyError`."""
    from grdl.exceptions import DependencyError
    from grdl.IO.sar.sentinel1_l0 import decoder as dec_mod
    from grdl.IO.sar.sentinel1_l0.reader import (
        Sentinel1L0Reader,
    )

    monkeypatch.setattr(dec_mod, "_HAS_S1_DECODER", False)
    safe = _make_l0_safe_no_decoder(tmp_path)

    with Sentinel1L0Reader(safe) as reader:
        with pytest.raises(DependencyError):
            reader.read_chip(0, 10, 0, 10)


def test_reader_bursts_empty_without_decoder(
    monkeypatch, tmp_path,
):
    """Without decoder, burst readers are not initialized; the
    ``bursts`` property returns an empty list and
    ``read_burst`` errors."""
    from grdl.IO.sar.sentinel1_l0 import decoder as dec_mod
    from grdl.IO.sar.sentinel1_l0.reader import (
        Sentinel1L0Reader,
    )

    monkeypatch.setattr(dec_mod, "_HAS_S1_DECODER", False)
    safe = _make_l0_safe_no_decoder(tmp_path)

    with Sentinel1L0Reader(safe) as reader:
        assert reader.bursts == []
        with pytest.raises(RuntimeError):
            reader.read_burst(0)


def test_reader_set_current_burst(monkeypatch, tmp_path):
    from grdl.IO.sar.sentinel1_l0 import decoder as dec_mod
    from grdl.IO.sar.sentinel1_l0.reader import (
        Sentinel1L0Reader,
    )

    monkeypatch.setattr(dec_mod, "_HAS_S1_DECODER", False)
    safe = _make_l0_safe_no_decoder(tmp_path)

    with Sentinel1L0Reader(safe) as reader:
        reader.set_current_burst(5, polarization="vh")
        assert reader.current_burst_index == 5
        # Polarization is canonicalised to upper case.
        assert reader.current_polarization == "VH"


def test_reader_timing_and_geometry_none_without_radar(
    monkeypatch, tmp_path,
):
    """When enough orbit vectors and start_time are populated,
    :meth:`get_geometry_calculator` returns an instance (tested
    elsewhere with richer orbit data); with only 2 orbit
    vectors it returns ``None``."""
    from grdl.IO.sar.sentinel1_l0 import decoder as dec_mod
    from grdl.IO.sar.sentinel1_l0.reader import (
        Sentinel1L0Reader,
    )

    monkeypatch.setattr(dec_mod, "_HAS_S1_DECODER", False)
    safe = _make_l0_safe_no_decoder(tmp_path)

    with Sentinel1L0Reader(safe) as reader:
        # Annotation XML only has 2 orbit vectors — too few
        # for the 4-point cubic requirement.
        assert len(reader.orbit_state_vectors) == 2
        assert reader.get_geometry_calculator() is None
        # Timing calculator requires radar params + start_time
        # both present — both are set here, so it should build.
        tc = reader.get_timing_calculator()
        assert tc is not None
        assert tc.prf_hz == 1717.0


def test_reader_config_overrides(monkeypatch, tmp_path):
    from grdl.IO.sar.sentinel1_l0 import decoder as dec_mod
    from grdl.IO.sar.sentinel1_l0.reader import (
        ReaderConfig,
        Sentinel1L0Reader,
    )

    monkeypatch.setattr(dec_mod, "_HAS_S1_DECODER", False)
    safe = _make_l0_safe_no_decoder(tmp_path)

    cfg = ReaderConfig(
        parse_annotations=False, load_poe=False,
    )
    with Sentinel1L0Reader(safe, config=cfg) as reader:
        # Annotation parsing disabled — radar defaults kick in.
        assert reader.metadata.radar_parameters is not None
        assert len(reader.metadata.orbit_state_vectors) == 0
        assert len(reader.metadata.attitude_records) == 0


def test_reader_open_safe_product_factory(
    monkeypatch, tmp_path,
):
    from grdl.IO.sar.sentinel1_l0 import decoder as dec_mod
    from grdl.IO.sar.sentinel1_l0.reader import (
        Sentinel1L0Reader,
        open_safe_product,
    )

    monkeypatch.setattr(dec_mod, "_HAS_S1_DECODER", False)
    safe = _make_l0_safe_no_decoder(tmp_path)

    reader = open_safe_product(safe, load_poe=False)
    try:
        assert isinstance(reader, Sentinel1L0Reader)
        assert reader.config.load_poe is False
    finally:
        reader.close()


def test_reader_missing_safe_raises(tmp_path):
    from grdl.IO.sar.sentinel1_l0.reader import (
        Sentinel1L0Reader,
    )
    with pytest.raises(FileNotFoundError):
        Sentinel1L0Reader(tmp_path / "does_not_exist.SAFE")


def test_reader_invalid_safe_raises(monkeypatch, tmp_path):
    """A ``.SAFE`` directory without ``manifest.safe`` fails
    structural validation."""
    from grdl.IO.sar.sentinel1_l0 import decoder as dec_mod
    from grdl.IO.sar.sentinel1_l0.reader import (
        Sentinel1L0Reader,
    )
    from grdl.IO.sar.sentinel1_l0.safe_product import (
        InvalidSAFEProductError,
    )

    monkeypatch.setattr(dec_mod, "_HAS_S1_DECODER", False)
    bad = tmp_path / (
        "S1A_IW_RAW__0SDV_20230501T060001_20230501T060101_"
        "048123_05C7E2_F3B9.SAFE"
    )
    bad.mkdir()
    (bad / "measurement").mkdir()
    with pytest.raises(InvalidSAFEProductError):
        Sentinel1L0Reader(bad)
