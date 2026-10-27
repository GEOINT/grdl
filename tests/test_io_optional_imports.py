# -*- coding: utf-8 -*-
"""
Optional-dependency import guard tests for grdl.IO and grdl.geolocation.

Runs fresh interpreters with h5py, sarpy and sarkit made unimportable and
verifies that the package, the EO NITF reader and geolocation still import,
and that guarded classes raise a clear ImportError at construction.

Dependencies
------------
pytest

Author
------
Curtis Ruck
curtis.ruck@gmail.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-10-06

Modified
--------
2026-10-06
"""

# Standard library
import subprocess
import sys
import textwrap

# Third-party
import pytest

_BLOCKER = textwrap.dedent('''
    import importlib.abc
    import sys

    _BLOCKED = {'h5py', 'sarpy', 'sarkit'}

    class _Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name.split('.')[0] in _BLOCKED:
                raise ModuleNotFoundError(
                    "No module named %r (blocked)" % name, name=name)
            return None

    sys.meta_path.insert(0, _Blocker())
''')


def _run(body: str) -> subprocess.CompletedProcess:
    code = _BLOCKER + textwrap.dedent(body)
    return subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True, text=True, timeout=300,
    )


def test_imports_without_sar_and_hdf5_extras():
    result = _run('''
        import grdl.IO
        from grdl.IO.eo.nitf import EONITFReader
        import grdl.geolocation
        for name in ('h5py', 'sarpy', 'sarkit'):
            assert name not in sys.modules or sys.modules[name] is None
        print('OK')
    ''')
    assert result.returncode == 0, result.stderr
    assert 'OK' in result.stdout


def test_guarded_classes_raise_clear_import_error():
    result = _run('''
        import numpy as np
        from grdl.IO.sar import SIDDWriter
        from grdl.IO.sar.compact_pol_synthesis import write_compact_pol_c2_h5
        from grdl.IO.hdf5 import HDF5Reader

        z = np.zeros((2, 2), dtype=np.float32)
        calls = {
            'sarpy': lambda: SIDDWriter('out.nitf'),
            'h5py': lambda: write_compact_pol_c2_h5(
                'out.h5', z, z, z, z),
        }
        for dep, call in calls.items():
            try:
                call()
            except ImportError as exc:
                assert dep in str(exc), str(exc)
                assert 'pip install' in str(exc), str(exc)
            else:
                raise SystemExit('no ImportError for ' + dep)
        print('OK')
    ''')
    assert result.returncode == 0, result.stderr
    assert 'OK' in result.stdout
