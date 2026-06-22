# -*- coding: utf-8 -*-
"""
Tests for CPHD per-channel PVP byte-offset repair and ``read_pvp``.

Some CPHD writers emit inconsistent ``Data/Channel/PVPArrayByteOffset``
values (e.g. scaled by ``NumSamples``) so the signal reader lands on the
wrong bytes for channels after the first. ``CPHDReader`` repairs these
offsets in-place to a contiguous PVP block before any PVP read. These
tests exercise the repair on a synthetic multi-channel CPHD XML tree and
the input validation of the public ``read_pvp`` accessor -- neither path
touches the signal block.

Dependencies
------------
lxml
sarkit

Author
------
Duane Smalley
duane.d.smalley@gmail.com

License
-------
MIT License
See LICENSE file for full text.

Created
-------
2026-06-07

Modified
--------
2026-06-07
"""

# Standard library
from types import SimpleNamespace

# Third-party
import pytest

lxml_etree = pytest.importorskip("lxml.etree")
pytest.importorskip("sarkit")

# GRDL internal
from grdl.IO.sar.cphd import CPHDReader


CPHD_NS = "http://api.nsgreg.nga.mil/schema/cphd/1.1.0"

# NumBytesPVP drives get_pvp_dtype().itemsize, which is the per-vector
# byte stride used to recompute contiguous offsets.
_NUM_BYTES_PVP = 240


def _build_xml(offsets, num_vectors=(16, 16, 16)):
    """Build a minimal CPHD 1.1.0 XML tree with the given PVP offsets.

    Parameters
    ----------
    offsets : sequence of int
        ``PVPArrayByteOffset`` value to write for each channel.
    num_vectors : sequence of int
        ``NumVectors`` for each channel.

    Returns
    -------
    lxml.etree._Element
        Root ``<CPHD>`` element with ``Data`` and ``PVP`` sections.
    """
    channels = "".join(
        f"""
        <Channel>
          <Identifier>Ch{i + 1}</Identifier>
          <NumVectors>{nv}</NumVectors>
          <NumSamples>32</NumSamples>
          <SignalArrayByteOffset>0</SignalArrayByteOffset>
          <PVPArrayByteOffset>{off}</PVPArrayByteOffset>
        </Channel>"""
        for i, (off, nv) in enumerate(zip(offsets, num_vectors))
    )
    xml = f"""\
<CPHD xmlns="{CPHD_NS}">
  <Data>
    <SignalArrayFormat>CF8</SignalArrayFormat>
    <NumBytesPVP>{_NUM_BYTES_PVP}</NumBytesPVP>
    <NumCPHDChannels>{len(offsets)}</NumCPHDChannels>
    <SignalCompressionID>None</SignalCompressionID>{channels}
  </Data>
  <PVP>
    <TxTime><Offset>0</Offset><Size>1</Size><Format>F8</Format></TxTime>
    <TxPos><Offset>1</Offset><Size>3</Size><Format>X=F8;Y=F8;Z=F8;</Format></TxPos>
  </PVP>
</CPHD>
"""
    return lxml_etree.fromstring(xml.encode("utf-8"))


def _offsets(root):
    return [
        int(e.text)
        for e in root.findall("{*}Data/{*}Channel/{*}PVPArrayByteOffset")
    ]


def test_repair_rewrites_inconsistent_offsets():
    """Offsets scaled by NumSamples are rewritten to a contiguous block."""
    # Bug pattern: offset = cumulative_NumVectors * NumSamples (not * pvp_bytes)
    root = _build_xml(offsets=[0, 16 * 32, 32 * 32])
    reader = CPHDReader.__new__(CPHDReader)

    reader._repair_pvp_offsets_sarkit(root)

    # Contiguous block: cumulative_NumVectors * NumBytesPVP
    assert _offsets(root) == [0, 16 * _NUM_BYTES_PVP, 32 * _NUM_BYTES_PVP]


def test_repair_is_noop_when_offsets_already_contiguous():
    """Already-contiguous offsets are left untouched."""
    correct = [0, 16 * _NUM_BYTES_PVP, 32 * _NUM_BYTES_PVP]
    root = _build_xml(offsets=correct)
    reader = CPHDReader.__new__(CPHDReader)

    reader._repair_pvp_offsets_sarkit(root)

    assert _offsets(root) == correct


def test_repair_single_channel_offset_zero():
    """A single channel is repaired to offset 0 regardless of input."""
    root = _build_xml(offsets=[9999], num_vectors=(16,))
    reader = CPHDReader.__new__(CPHDReader)

    reader._repair_pvp_offsets_sarkit(root)

    assert _offsets(root) == [0]


def _reader_with_channels(*ids):
    reader = CPHDReader.__new__(CPHDReader)
    reader.metadata = SimpleNamespace(
        channels=[SimpleNamespace(identifier=i) for i in ids]
    )
    return reader


def test_read_pvp_unknown_identifier_raises():
    reader = _reader_with_channels("Ch1", "Ch2")
    with pytest.raises(ValueError, match="Unknown channel identifier"):
        reader.read_pvp("DoesNotExist")


def test_read_pvp_index_out_of_range_raises():
    reader = _reader_with_channels("Ch1", "Ch2")
    with pytest.raises(ValueError, match="out of range"):
        reader.read_pvp(5)
    with pytest.raises(ValueError, match="out of range"):
        reader.read_pvp(-1)
