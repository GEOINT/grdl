#!/bin/bash
set -e

NOTEBOOK_DIRS=(
    "/nfs_home/alcourt/grdx/grdl/notebooks/image_processing"
    "/nfs_home/alcourt/grdx/grdl/notebooks/ortho"
    "/nfs_home/alcourt/grdx/grdl/notebooks/shapes"
    "/nfs_home/alcourt/grdx/grdl/notebooks/geolocation"
    "/nfs_home/alcourt/grdx/grdl/notebooks/IO"
    "/nfs_home/alcourt/grdx/grdl/notebooks/catalog"
    "/nfs_home/alcourt/grdx/grdl/notebooks/interpolation"
)
OUTPUT_DIR="/tmp/grdl_notebook_validation"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "════════════════════════════════════════════════════════════"
echo "  GRDL Notebook Validation Suite"
echo "  Timestamp: $TIMESTAMP"
echo "════════════════════════════════════════════════════════════"
echo ""

# Activate environment
eval "$(conda shell.bash hook)"
conda activate grdx

# Verify grdl installation
echo "▶ Verifying grdl installation..."
python -c "import grdl; print(f'  grdl version: {grdl.__version__}')"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Track results
PASSED=0
FAILED=0
FAILED_NOTEBOOKS=()

for notebook_dir in "${NOTEBOOK_DIRS[@]}"; do
    if [ ! -d "$notebook_dir" ]; then
        continue
    fi
    
    for nb in "$notebook_dir"/*.ipynb; do
        if [ ! -f "$nb" ]; then
            continue
        fi
        
        nb_name=$(basename "$nb")
    echo "════════════════════════════════════════════════════════════"
    echo "▶ Validating: $nb_name"
    echo "════════════════════════════════════════════════════════════"
    
    # Execute notebook (with syntax check only - no data files required)
    # Use --execute-only flag and catch import errors
    if python -c "
import sys
import json
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

try:
    with open('$nb', 'r') as f:
        notebook = nbformat.read(f, as_version=4)
    
    # Only execute the first 2 cells (preamble, imports)
    # Skip config cell (cell 3) which has path assertions requiring user data
    preamble_notebook = nbformat.v4.new_notebook()
    preamble_notebook.cells = notebook.cells[:2]
    
    ep = ExecutePreprocessor(timeout=60, kernel_name='python3')
    ep.preprocess(preamble_notebook)
    
    print('✓ PASS (preamble validation)')
    sys.exit(0)
except Exception as e:
    print(f'✗ FAIL: {e}')
    sys.exit(1)
" 2>&1; then
        echo "  ✓ PASS"
        echo ""
        PASSED=$((PASSED + 1))
    else
        echo "  ✗ FAIL"
        echo ""
        FAILED=$((FAILED + 1))
        FAILED_NOTEBOOKS+=("$nb_name")
    fi
    done  # End inner loop (notebooks)
done  # End outer loop (directories)

echo "════════════════════════════════════════════════════════════"
echo "  Validation Summary"
echo "════════════════════════════════════════════════════════════"
echo "  Total notebooks  : $((PASSED + FAILED))"
echo "  Passed           : $PASSED"
echo "  Failed           : $FAILED"
echo ""

if [ $FAILED -gt 0 ]; then
    echo "  Failed notebooks:"
    for nb in "${FAILED_NOTEBOOKS[@]}"; do
        echo "    - $nb"
    done
    echo ""
    exit 1
else
    echo "  ✓ All notebooks validated successfully"
    echo ""
fi
