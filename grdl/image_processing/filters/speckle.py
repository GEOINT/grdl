# -*- coding: utf-8 -*-
"""
Deprecated: this module has been renamed to ``standard_lee``.

All symbols are re-exported here for backward compatibility.  Import from
``grdl.image_processing.filters.standard_lee`` or the package-level
``grdl.image_processing.filters`` instead.
"""
from grdl.image_processing.filters.standard_lee import (  # noqa: F401
    ComplexLeeFilter,
    LeeFilter,
)
from grdl.image_processing.filters.sar_base import SARFilter as _SARFilter

# Module-level _estimate_enl kept for any code that imported it directly.
_estimate_enl = _SARFilter._estimate_enl
