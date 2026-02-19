# -*- coding: utf-8 -*-
"""
Detect and CSI Workflow — SICD ship scene exploration.

Opens an UMBRA SICD NITF and extracts a center chip for analysis.
Demonstrates how to build a grdl-runtime Workflow with deferred
processor construction and custom callable steps, then benchmark it
using grdl-te's ActiveBenchmarkRunner and BenchmarkSource.

Dependencies
------------
sarkit (or sarpy)
grdl-runtime

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
2026-02-19
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
from grdl.data_prep import ChipExtractor
from grdl.image_processing.sar.sublook import SublookDecomposition
from grdl.image_processing.sar.csi import CSIProcessor
from scipy.ndimage import uniform_filter
from skimage.morphology import opening, closing, footprint_rectangle
from skimage.measure import label

import matplotlib.pyplot as plt

# grdl-runtime
from grdl_rt import Workflow
from grdl_rt.execution.dag_executor import DAGExecutor  # noqa: F401

# GRDL-TE benchmarking
from grdl_te.benchmarking import (
    ActiveBenchmarkRunner,
    BenchmarkSource,
    JSONBenchmarkStore,
    print_report,
)


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

    # ---------- Configuration ----------
    CONFIG_PATH = Path(__file__).parent / "config.yaml"
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    SICD_PATH = Path(cfg["input"]["sicd_path"])
    CHIP_SIZE = cfg["input"]["chip_size"]

    # ---------- Shared data ----------
    chip_wf = (
        Workflow("Chip", version="1.0.0", modalities=["SAR"])
        .reader(SICDReader)
        .chip("center", size=CHIP_SIZE)
    )
    with SICDReader(SICD_PATH) as rdr:
        chip_metadata = rdr.metadata
    chip_wf_result = chip_wf.execute(SICD_PATH)
    chip = chip_wf_result.result

    # ---------- Detection workflow ----------
    # reader → chip → SublookDecomposition → DominanceDetection
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
    det_result = det_wf.execute(chip, metadata=chip_metadata)
    dominance, labeled = det_result.result
    n_detections = labeled.max()
    print(f"Detections: {n_detections}")

    # ---------- CSI workflow ----------
    # reader → chip → CSIProcessor
    csi_wf = (
        Workflow("CSI", version="1.0.0", modalities=["SAR"])
        .step(CSIProcessor,
              dimension=cfg["sublook"]["dimension"],
              normalization='log')
    )
    csi_result = csi_wf.execute(chip, metadata=chip_metadata)
    csi_rgb = csi_result.result

    # ---------- Plot ----------
    # Load raw chip for amplitude overlay (no-step workflow returns source)
    # plot_results(chip, dominance, labeled, csi_rgb, n_detections, cfg)

    # ---------- Benchmarking ----------
    store = JSONBenchmarkStore()

    bench_cfg = cfg["benchmark"]
    bench_iterations = bench_cfg["iterations"]
    bench_warmup = bench_cfg["warmup"]

    det_runner = ActiveBenchmarkRunner(
        det_wf,
        BenchmarkSource.from_array(chip),
        iterations=bench_iterations,
        warmup=bench_warmup,
        store=store,
        tags={"workflow": "Detection"},
    )
    csi_runner = ActiveBenchmarkRunner(
        csi_wf,
        BenchmarkSource.from_array(chip),
        iterations=bench_iterations,
        warmup=bench_warmup,
        store=store,
        tags={"workflow": "CSI"},
    )

    det_rec = det_runner.run(metadata=chip_metadata)
    csi_rec = csi_runner.run(metadata=chip_metadata)

    print_report([det_rec, csi_rec])


if __name__ == "__main__":
    main()
