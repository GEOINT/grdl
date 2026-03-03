# -*- coding: utf-8 -*-
"""
Detect and CSI Workflow — SICD ship scene exploration.

Opens an UMBRA SICD NITF and extracts a center chip for analysis.
Supports sequential and parallel execution modes, controlled by
flags at the top of ``main()``.

Sequential mode runs Detection and CSI as independent linear
workflows one after the other.  Parallel mode uses grdl-runtime's
``Workflow.branches()`` DAG API so both branches execute
simultaneously — no manual concurrency code.

DAG structure (parallel mode)::

                  ┌─ SublookDecomposition → DominanceDetection
    chip(center) ─┤
                  └─ CSIProcessor

Benchmarks compare per-branch times (sequential) against the
unified parallel workflow to verify wall-clock improvement.

Dependencies
------------
sarkit (or sarpy)
grdl-runtime (optional — enables Workflow orchestration)
grdl-te (optional — enables benchmarking)

Author
------
Duane Smalley
170194430+DDSmalls@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-02-18

Modified
--------
2026-02-27
"""

# Standard library
import sys
from pathlib import Path

# Third-party
import matplotlib
matplotlib.use("QtAgg")
import numpy as np
import yaml

# GRDL
sys.path.insert(0, str(Path.home() / "GRDL"))
from grdl.IO.sar import SICDReader
from grdl.image_processing.sar.sublook import SublookDecomposition
from grdl.image_processing.sar.csi import CSIProcessor
from scipy.ndimage import uniform_filter
from skimage.morphology import opening, closing, footprint_rectangle
from skimage.measure import label

import matplotlib.pyplot as plt

# grdl-runtime (optional)
try:
    from grdl_rt import Workflow
    HAS_GRDL_RT = True
except ImportError:
    HAS_GRDL_RT = False

# grdl-te benchmarking (optional)
try:
    from grdl_te.benchmarking import (
        ActiveBenchmarkRunner,
        BenchmarkSource,
        save_report,
    )
    HAS_GRDL_TE = True
except ImportError:
    HAS_GRDL_TE = False


class DominanceDetection:
    """Sliding-window aperture dominance detection.

    Computes a dominance ratio from a sub-look power stack, thresholds
    at mean + N*sigma, and returns morphologically cleaned labeled
    regions.  Implements ``apply()`` so it can be used as a deferred
    step in a grdl-runtime Workflow.

    Parameters
    ----------
    smooth_win : int
        Uniform filter window size for power smoothing.
    dom_window : int
        Contiguous block length for dominance ratio.
    dom_sigma : float
        Detection threshold in standard deviations above the mean.
    morph_size : int
        Morphological opening/closing kernel size.
    """

    def __init__(self, smooth_win: int = 7, dom_window: int = 3,
                 dom_sigma: float = 3.0, morph_size: int = 3):
        self.smooth_win = smooth_win
        self.dom_window = dom_window
        self.dom_sigma = dom_sigma
        self.morph_size = morph_size

    def apply(self, looks: np.ndarray) -> tuple:
        """Apply dominance detection to a sub-look stack.

        Parameters
        ----------
        looks : np.ndarray
            Sub-look stack, shape ``(num_looks, rows, cols)``.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(dominance, labeled)`` — dominance map and labeled regions.
        """
        print(f"Sublook stack shape: {looks.shape}  dtype: {looks.dtype}")

        eps = np.finfo(np.float64).tiny
        num_looks = looks.shape[0]

        # Sliding window dominance: max contiguous block / total power
        power = np.abs(looks) ** 2
        smooth_power = np.stack([
            uniform_filter(power[i], size=self.smooth_win)
            for i in range(num_looks)
        ])
        total_power = smooth_power.sum(axis=0) + eps

        n_windows = num_looks - self.dom_window + 1
        window_sums = np.stack([
            smooth_power[i:i + self.dom_window].sum(axis=0)
            for i in range(n_windows)
        ])
        dominance = window_sums.max(axis=0) / total_power

        # Threshold: mean + N*sigma
        dom_mu = np.mean(dominance)
        dom_std = np.std(dominance)
        dom_thresh = dom_mu + self.dom_sigma * dom_std
        det_mask = dominance > dom_thresh
        print(f"Dominance threshold: {dom_thresh:.3f}  "
              f"(mu={dom_mu:.3f}, std={dom_std:.4f}, {self.dom_sigma}σ)")

        # Morphological cleanup + labeling
        det_mask = opening(
            det_mask,
            footprint=footprint_rectangle((self.morph_size, self.morph_size)),
        )
        det_mask = closing(
            det_mask,
            footprint=footprint_rectangle((self.morph_size, self.morph_size)),
        )
        labeled = label(det_mask)

        return dominance, labeled


# ===================== Plotting =====================

def plot_results(chip: np.ndarray, dominance: np.ndarray,
                 labeled: np.ndarray, csi_rgb: np.ndarray,
                 n_detections: int, cfg: dict):
    """Display dominance map, CSI composite, and detection overlay.

    Parameters
    ----------
    chip : np.ndarray
        Original complex SAR chip (used for amplitude background).
    dominance : np.ndarray
        Per-pixel dominance ratio.
    labeled : np.ndarray
        Integer-labeled detection regions.
    csi_rgb : np.ndarray
        RGB CSI composite image.
    n_detections : int
        Number of detected objects.
    cfg : dict
        Full config dict (uses ``sublook``, ``detection``, ``display``).
    """
    num_looks = cfg["sublook"]["num_looks"]
    dom_window = cfg["detection"]["dom_window"]
    dom_sigma = cfg["detection"]["sigma"]

    vmax = np.percentile(np.abs(chip), cfg["display"]["vmax_percentile"])
    dom_floor = dom_window / num_looks

    # Dominance image
    fig_dom, ax_dom = plt.subplots(figsize=(10, 10))
    im = ax_dom.imshow(dominance, cmap='inferno', aspect='auto',
                       vmin=dom_floor, vmax=1)
    ax_dom.set_title(f"Aperture Dominance (win={dom_window}, {num_looks} looks)")
    ax_dom.set_xlabel("Column")
    ax_dom.set_ylabel("Row")
    fig_dom.colorbar(im, ax=ax_dom, label="Dominance ratio")
    plt.tight_layout()

    # Linked comparison: CSI vs detections
    fig, (ax_csi, ax_det) = plt.subplots(
        1, 2, figsize=(20, 10), sharex=True, sharey=True,
    )

    ax_csi.imshow(csi_rgb, aspect='auto')
    ax_csi.set_title("CSI")
    ax_csi.set_xlabel("Column")
    ax_csi.set_ylabel("Row")

    ax_det.imshow(np.abs(chip), cmap='gray', aspect='auto', vmax=vmax)
    ax_det.contour(labeled > 0, colors='red', linewidths=0.5)
    ax_det.set_title(f"Detections ({n_detections} objects, {dom_sigma}σ)")
    ax_det.set_xlabel("Column")

    fig.suptitle("Sub-aperture Dominance Detection", fontsize=14)
    plt.tight_layout()
    plt.show()


# ===================== Main =====================

def main():
    # -------- Editable Execution Steps --------
    plotting = False
    benchmarking = True
    sequential = True
    parallel = True
    gpu = False

    # ---------- Configuration ----------
    CONFIG_PATH = Path(__file__).parent / "config.yaml"
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    SICD_PATH = Path(cfg["input"]["sicd_path"])
    CHIP_SIZE = cfg["input"]["chip_size"]

    # ---------- Chip Extraction ----------
    chip_wf = (
        Workflow("Chip", version="1.0.0", modalities=["SAR"])
        .reader(SICDReader)
        .chip("center", size=CHIP_SIZE)
    )
    chip_result = chip_wf.execute(SICD_PATH)
    chip = chip_result.result

    with SICDReader(SICD_PATH) as rdr:
        chip_metadata = rdr.metadata

    # ---------- Sequential Workflows ----------
    det_wf = (
        Workflow("Detection", version="1.0.0", modalities=["SAR"])
        .step(SublookDecomposition,
              num_looks=cfg["sublook"]["num_looks"],
              dimension=cfg["sublook"]["dimension"])
        .step(DominanceDetection,
              smooth_win=cfg["detection"]["smooth_win"],
              dom_window=cfg["detection"]["dom_window"],
              dom_sigma=cfg["detection"]["sigma"],
              morph_size=cfg["detection"]["morph_size"])
    )
    csi_wf = (
        Workflow("CSI", version="1.0.0", modalities=["SAR"])
        .step(CSIProcessor,
              dimension=cfg["sublook"]["dimension"],
              normalization='log')
    )

    # ---------- Parallel Workflow (DAG) ----------
    # chip → ┬─ SublookDecomposition → DominanceDetection
    #        └─———————————— CSIProcessor
    parallel_wf = (
        Workflow("CSI-Detection", version="2.0.0", modalities=["SAR"])
        .branches(
            Workflow.branch("detection")
                .step(SublookDecomposition,
                      id="sublook",
                      num_looks=cfg["sublook"]["num_looks"],
                      dimension=cfg["sublook"]["dimension"])
                .step(DominanceDetection,
                      id="dominance",
                      smooth_win=cfg["detection"]["smooth_win"],
                      dom_window=cfg["detection"]["dom_window"],
                      dom_sigma=cfg["detection"]["sigma"],
                      morph_size=cfg["detection"]["morph_size"]),
            Workflow.branch("csi")
                .step(CSIProcessor,
                      id="csi_proc",
                      dimension=cfg["sublook"]["dimension"],
                      normalization='log'),
        )
    )

    # ---------- Execute ----------
    if not benchmarking:
        if sequential:
            det_result = det_wf.execute(chip, metadata=chip_metadata, prefer_gpu=gpu)
            dominance_seq, labeled_seq = det_result.result
            csi_result = csi_wf.execute(chip, metadata=chip_metadata, prefer_gpu=gpu)
            csi_rgb_seq = csi_result.result
            n_detections_seq = labeled_seq.max()
            print(f"Sequential Detections: {n_detections_seq}")

        if parallel:
            result = parallel_wf.execute(chip, metadata=chip_metadata, prefer_gpu=gpu)
            dominance, labeled = result.step_results["dominance"]
            csi_rgb = result.step_results["csi_proc"]
            n_detections = labeled.max()
            print(f"Parallel Detections: {n_detections}")

        # ---------- Plot ----------
        if plotting:
            if parallel:
                plot_results(chip, dominance, labeled, csi_rgb,
                            n_detections, cfg)
            elif sequential:
                plot_results(chip, dominance_seq, labeled_seq, csi_rgb_seq,
                            n_detections_seq, cfg)

    # ---------- Benchmarking ----------
    if benchmarking:
        if not (HAS_GRDL_RT and HAS_GRDL_TE):
            print("Benchmarking requires grdl-runtime and grdl-te — skipping.")
        else:
            bench_cfg = cfg["benchmark"]
            bench_iterations = bench_cfg["iterations"]
            bench_warmup = bench_cfg["warmup"]

        if bench_cfg["source"] == "synthetic":
            bench_source = BenchmarkSource.synthetic(
                bench_cfg["synthetic_size"], dtype=np.complex64,
            )
        else:
            bench_source = BenchmarkSource.from_array(chip)

        if sequential:
            det_runner = ActiveBenchmarkRunner(
                det_wf, bench_source,
                iterations=bench_iterations, warmup=bench_warmup,
                tags={"workflow": "Detection"},
            )
            csi_runner = ActiveBenchmarkRunner(
                csi_wf, bench_source,
                iterations=bench_iterations, warmup=bench_warmup,
                tags={"workflow": "CSI"},
            )
            det_rec = det_runner.run(metadata=chip_metadata, prefer_gpu=gpu)
            csi_rec = csi_runner.run(metadata=chip_metadata, prefer_gpu=gpu)
            save_report(
                [det_rec, csi_rec],
                f"{Path.cwd()}/../benchmark_reports/"
                "csi_detect_workflow_sequential.txt",
            )

        if parallel:
            unified_runner = ActiveBenchmarkRunner(
                parallel_wf, bench_source,
                iterations=bench_iterations, warmup=bench_warmup,
                tags={"workflow": "CSI-Detection (parallel)"},
            )
            unified_rec = unified_runner.run(metadata=chip_metadata, prefer_gpu=gpu)
            save_report(
                [unified_rec],
                f"{Path.cwd()}/../benchmark_reports/"
                "csi_detect_workflow_parallel.txt",
            )


if __name__ == "__main__":
    main()
