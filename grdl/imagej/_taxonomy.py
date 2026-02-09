# -*- coding: utf-8 -*-
"""
ImageJ Component Taxonomy - Canonical category identifiers.

Maps ImageJ's menu structure to snake_case category strings used in
``@processor_tags(category=...)`` decorators and GRDK discovery filtering.
Both GRDL and GRDK reference this module as the single source of truth
for the ImageJ component categorization.

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
2026-02-09
"""

# Category constants — each matches its subdirectory name under grdl/imagej/
FILTERS = 'filters'
BACKGROUND = 'background'
BINARY = 'binary'
ENHANCE = 'enhance'
EDGES = 'edges'
FFT = 'fft'
FIND_MAXIMA = 'find_maxima'
THRESHOLD = 'threshold'
SEGMENTATION = 'segmentation'
STACKS = 'stacks'
MATH = 'math'
ANALYZE = 'analyze'
NOISE = 'noise'

# Ordered tuple for iteration and UI display
ALL_CATEGORIES = (
    FILTERS,
    BACKGROUND,
    BINARY,
    ENHANCE,
    EDGES,
    FFT,
    FIND_MAXIMA,
    THRESHOLD,
    SEGMENTATION,
    STACKS,
    MATH,
    ANALYZE,
    NOISE,
)

# Human-readable labels mirroring ImageJ menu paths
CATEGORY_LABELS = {
    FILTERS: 'Process > Filters',
    BACKGROUND: 'Process > Subtract Background',
    BINARY: 'Process > Binary',
    ENHANCE: 'Process > Enhance Contrast',
    EDGES: 'Process > Find Edges',
    FFT: 'Process > FFT',
    FIND_MAXIMA: 'Process > Find Maxima',
    THRESHOLD: 'Image > Adjust > Threshold',
    SEGMENTATION: 'Plugins > Segmentation',
    STACKS: 'Image > Stacks',
    MATH: 'Process > Image Calculator',
    ANALYZE: 'Analyze > Analyze Particles',
    NOISE: 'Plugins > Anisotropic Diffusion',
}
