"""Yamaguchi 4C speed comparison: GRDL (numba JIT) vs polsartools pure Python."""
import sys, time
import numpy as np

sys.path.insert(0, '.')
from grdl.image_processing.decomposition import Yamaguchi4C, CoherencyMatrix
from tests.grdl_polsar_vs_polsartools import polsartools_yam4c_py

rng = np.random.default_rng(0)

def make_t3(n):
    shh = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))).astype(np.complex64)
    shv = (0.3 * (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))).astype(np.complex64)
    svh = shv.copy()
    svv = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))).astype(np.complex64)
    channels = np.stack([shh, shv, svh, svv], axis=0)
    return CoherencyMatrix(window_size=5).compute(channels)

# ── JIT warm-up ─────────────────────────────────────────────────────────────
print("Warming up numba JIT (first-call compilation)...", flush=True)
t3_warm = make_t3(32)
t0 = time.perf_counter()
Yamaguchi4C(model='y4o').decompose_from_t3(t3_warm)
t_compile = time.perf_counter() - t0
print(f"  JIT compile time (one-time cost): {t_compile:.2f} s\n")

REPS = 3

print(f"{'Size':>8}  {'Pixels':>9}  {'GRDL numba':>12}  {'pst Python':>12}  {'Speedup':>9}")
print("-" * 60)

for n, run_pst in [(50, True), (100, True), (200, False), (500, False), (1000, False)]:
    t3 = make_t3(n)
    yam = Yamaguchi4C(model='y4o')

    # GRDL (numba already compiled)
    grdl_ms = min(
        (time.perf_counter(), yam.decompose_from_t3(t3), time.perf_counter())[2]
        - (time.perf_counter(), yam.decompose_from_t3(t3), time.perf_counter())[0]
        for _ in range(REPS)
    )
    # simpler timing
    times = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        yam.decompose_from_t3(t3)
        times.append((time.perf_counter() - t0) * 1000)
    grdl_ms = min(times)

    if run_pst:
        times = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            polsartools_yam4c_py(t3, model='')
            times.append((time.perf_counter() - t0) * 1000)
        pst_ms = min(times)
        speedup = f"{pst_ms / grdl_ms:.0f}x"
        print(f"{n:>8}  {n*n:>9,}  {grdl_ms:>11.1f}ms  {pst_ms:>11.1f}ms  {speedup:>9}")
    else:
        print(f"{n:>8}  {n*n:>9,}  {grdl_ms:>11.1f}ms  {'(skipped)':>12}  {'N/A':>9}")

print()
print("GRDL-numba uses numba.njit parallel=True (prange over pixels).")
print("pst-Python  = polsartools yam4c_fp_py.py double for-loop.")
print("Sizes >100 skipped for polsartools (quadratic time, impractical).")
