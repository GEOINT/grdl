# -*- coding: utf-8 -*-
"""Unit tests for CSIProcessor execute metadata behavior."""

import numpy as np

from grdl.IO.models.base import ImageMetadata
from grdl.image_processing.sar.csi import CSIProcessor


class TestCSIExecute:

    def test_execute_sets_yxc_axis_and_rgb_channels(self):
        processor = CSIProcessor.__new__(CSIProcessor)
        processor.apply = (
            lambda source, **kwargs:
            np.zeros((source.shape[0], source.shape[1], 3), dtype=np.float32)
        )

        meta = ImageMetadata(
            format='test', rows=10, cols=12, dtype='complex64', bands=1,
        )
        source = np.ones((10, 12), dtype=np.complex64)

        result, out_meta = processor.execute(meta, source)

        assert result.shape == (10, 12, 3)
        assert out_meta.axis_order == 'YXC'
        assert out_meta.bands == 3
        assert out_meta.channel_metadata is not None
        assert [ch.name for ch in out_meta.channel_metadata] == ['R', 'G', 'B']
        assert [ch.role for ch in out_meta.channel_metadata] == ['display'] * 3
