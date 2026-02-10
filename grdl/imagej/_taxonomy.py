# -*- coding: utf-8 -*-
"""
ImageJ Component Taxonomy - Canonical category identifiers.

Maps ImageJ's menu structure to ``ProcessorCategory`` enum members used
in ``@processor_tags(category=...)`` decorators and GRDK discovery
filtering. Both GRDL and GRDK reference this module as the single source
of truth for the ImageJ component categorization.

Author
------
Steven Siebert
Jason Fritz

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-02-06

Modified
--------
2026-02-10
"""

from grdl.vocabulary import ProcessorCategory

# Convenience aliases matching subdirectory names under grdl/imagej/
FILTERS = ProcessorCategory.FILTERS
BACKGROUND = ProcessorCategory.BACKGROUND
BINARY = ProcessorCategory.BINARY
ENHANCE = ProcessorCategory.ENHANCE
EDGES = ProcessorCategory.EDGES
FFT = ProcessorCategory.FFT
FIND_MAXIMA = ProcessorCategory.FIND_MAXIMA
THRESHOLD = ProcessorCategory.THRESHOLD
SEGMENTATION = ProcessorCategory.SEGMENTATION
STACKS = ProcessorCategory.STACKS
MATH = ProcessorCategory.MATH
ANALYZE = ProcessorCategory.ANALYZE
NOISE = ProcessorCategory.NOISE

# Ordered tuple for iteration and UI display
ALL_CATEGORIES = tuple(ProcessorCategory)

# Human-readable labels mirroring ImageJ menu paths
CATEGORY_LABELS = {
    ProcessorCategory.FILTERS: 'Process > Filters',
    ProcessorCategory.BACKGROUND: 'Process > Subtract Background',
    ProcessorCategory.BINARY: 'Process > Binary',
    ProcessorCategory.ENHANCE: 'Process > Enhance Contrast',
    ProcessorCategory.EDGES: 'Process > Find Edges',
    ProcessorCategory.FFT: 'Process > FFT',
    ProcessorCategory.FIND_MAXIMA: 'Process > Find Maxima',
    ProcessorCategory.THRESHOLD: 'Image > Adjust > Threshold',
    ProcessorCategory.SEGMENTATION: 'Plugins > Segmentation',
    ProcessorCategory.STACKS: 'Image > Stacks',
    ProcessorCategory.MATH: 'Process > Image Calculator',
    ProcessorCategory.ANALYZE: 'Analyze > Analyze Particles',
    ProcessorCategory.NOISE: 'Plugins > Anisotropic Diffusion',
}
