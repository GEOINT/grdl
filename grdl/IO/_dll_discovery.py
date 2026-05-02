# -*- coding: utf-8 -*-
"""
DLL Discovery Utility - Automated library path resolution for grdl dependencies.

Focuses on resolving OpenJPEG (openjp2.dll) for glymur on Windows,
supporting various environment layouts (conda, venv, system).

Author
------
Claude Code (Anthropic)

Created
-------
2026-05-01
"""

import os
import sys
from pathlib import Path

def ensure_openjp2_loaded():
    """
    Ensures the OpenJPEG library (openjp2.dll) is discoverable on Windows.
    
    On Windows, this helper looks for openjp2.dll in common environment 
    locations like Library\\bin (conda) or bin (venv) and adds those
    directories to the os.environ["PATH"] to ensure glymur can find them.
    
    This function is a no-op on non-Windows platforms.
    """
    if sys.platform != "win32":
        return

    # Potential relative paths to DLLs from the Python executable
    # Conda: <env_root>/Library/bin
    # venv: <env_root>/bin (less common for DLLs but possible)
    # Generic: <env_root>/DLLs
    search_rel_paths = [
        "Library/bin",
        "bin",
        "DLLs",
        "../Library/bin",  # If executable is in <env>/Scripts
    ]

    executable_path = Path(sys.executable).parent
    
    added_paths = []
    for rel_path in search_rel_paths:
        dll_dir = (executable_path / rel_path).resolve()
        if dll_dir.exists() and dll_dir.is_dir():
            # Check if openjp2.dll or openjpeg.dll is actually here
            if list(dll_dir.glob("openjp*.dll")):
                added_paths.append(str(dll_dir))
    
    # Also check the environment's sys.prefix if it differs from executable path
    prefix_path = Path(sys.prefix)
    if prefix_path != executable_path:
        for rel_path in search_rel_paths:
            dll_dir = (prefix_path / rel_path).resolve()
            if dll_dir.exists() and dll_dir.is_dir():
                if list(dll_dir.glob("openjp*.dll")):
                    added_paths.append(str(dll_dir))
    
    if added_paths:
        # Add to PATH. Glymur (via ctypes/CFFI) often relies on PATH 
        # for discovering underlying C libraries on Windows.
        current_path = os.environ.get("PATH", "")
        for path in added_paths:
            if path not in current_path:
                os.environ["PATH"] = path + os.pathsep + os.environ["PATH"]
        return True
    
    return False
