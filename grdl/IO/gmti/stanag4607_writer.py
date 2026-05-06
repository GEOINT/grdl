# -*- coding: utf-8 -*-
"""
STANAG 4607 writer - serialize a STANAG4607Metadata to a GMTI file.

The writer takes an already-populated ``STANAG4607Metadata`` (a list of
``Packet`` dataclasses, each containing typed segments) and serializes
the wire format defined in ``grdl.IO.gmti._segments``. Packet sizes are
recomputed from the actual serialized segment lengths to keep the
``packet_size`` field consistent regardless of how the user populated
the dataclasses.

Edition handling: the writer stamps every packet header with the
edition supplied at construction (default Edition 4). Edition-specific
field-table differences are not applied — the v1 codec writes a fixed
field set per segment type. This matches the reader, so files written
by ``STANAG4607Writer`` round-trip cleanly through
``STANAG4607Reader``.

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
2026-04-29

Modified
--------
2026-04-29
"""

# Standard library
from pathlib import Path
from typing import List, Union

# GRDL internal
from grdl.exceptions import ValidationError
from grdl.IO.gmti import _segments as _seg
from grdl.IO.models.stanag4607 import STANAG4607Metadata


_EDITION_TO_VERSION_ID = {2: '20', 3: '30', 4: '40'}


class STANAG4607Writer:
    """Writer for STANAG 4607 (NATO GMTI) files.

    Parameters
    ----------
    filepath : str or Path
        Output file path. Created or overwritten on ``write()``.
    metadata : STANAG4607Metadata
        Fully populated metadata containing the packets and segments to
        serialize.
    edition : int
        STANAG 4607 edition stamped into the version field of every
        packet header. Must be one of 2, 3, or 4. Default 4.

    Raises
    ------
    ValidationError
        If ``edition`` is not in {2, 3, 4} or if ``metadata`` is not a
        ``STANAG4607Metadata``.

    Examples
    --------
    >>> writer = STANAG4607Writer('out.4607', metadata=meta)
    >>> writer.write()
    >>> writer.close()
    """

    def __init__(
        self,
        filepath: Union[str, Path],
        metadata: STANAG4607Metadata,
        edition: int = 4,
    ) -> None:
        if not isinstance(metadata, STANAG4607Metadata):
            raise ValidationError(
                f"metadata must be STANAG4607Metadata, got "
                f"{type(metadata).__name__}"
            )
        if edition not in _EDITION_TO_VERSION_ID:
            raise ValidationError(
                f"Unsupported edition {edition}; must be one of "
                f"{sorted(_EDITION_TO_VERSION_ID)}"
            )

        self.filepath = Path(filepath)
        self.metadata = metadata
        self.edition = edition

    # ------------------------------------------------------------------

    def _build_bytes(self) -> bytes:
        """Serialize all packets in ``self.metadata`` to wire bytes."""
        version_id = _EDITION_TO_VERSION_ID[self.edition]
        chunks: List[bytes] = []

        for packet in self.metadata.packets:
            # Serialize all segments first so we know the total size.
            segment_bytes = b''.join(
                _seg.serialize_segment(s) for s in packet.segments
            )
            packet_size = _seg.PACKET_HEADER_SIZE + len(segment_bytes)

            # Stamp the header with the chosen edition and the
            # recomputed packet size, leaving caller-supplied identity
            # fields (nationality, classification, ids, ...) intact.
            packet.header.version_id = version_id
            packet.header.packet_size = packet_size

            chunks.append(_seg.serialize_packet_header(packet.header))
            chunks.append(segment_bytes)

        return b''.join(chunks)

    def write(self) -> None:
        """Serialize and write all packets to ``self.filepath``."""
        data = self._build_bytes()
        with open(self.filepath, 'wb') as fh:
            fh.write(data)

    def close(self) -> None:
        """No-op — the writer does not hold persistent file handles."""
        return None

    def __enter__(self) -> 'STANAG4607Writer':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False
