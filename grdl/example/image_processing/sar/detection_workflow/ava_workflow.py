# -*- coding: utf-8 -*-
"""
Ava Workflow — SICD ship scene exploration.

Opens an UMBRA SICD NITF and extracts a 5k x 5k center chip for analysis.

Dependencies
------------
sarkit (or sarpy)

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
2026-02-18
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

def sublook_data(chip: np.ndarray, num_looks: int, dimension: str, metadata) -> np.ndarray:
    sublook = SublookDecomposition(
        metadata,
        num_looks=num_looks,
        dimension=dimension,
    )
    return sublook.decompose(chip)


def dominance_detection(looks: np.ndarray, smooth_win: int, dom_window: int,
                        dom_sigma: float, morph_size: int):
    print(f"Sublook stack shape: {looks.shape}  dtype: {looks.dtype}")

    eps = np.finfo(np.float64).tiny
    num_looks = looks.shape[0]

    # Sliding window dominance: max contiguous block / total power
    power = np.abs(looks) ** 2
    smooth_power = np.stack([
        uniform_filter(power[i], size=smooth_win) for i in range(num_looks)
    ])
    total_power = smooth_power.sum(axis=0) + eps

    n_windows = num_looks - dom_window + 1
    window_sums = np.stack([
        smooth_power[i:i + dom_window].sum(axis=0) for i in range(n_windows)
    ])
    dominance = window_sums.max(axis=0) / total_power

    # Threshold: mean + N*sigma
    dom_mu = np.mean(dominance)
    dom_std = np.std(dominance)
    dom_thresh = dom_mu + dom_sigma * dom_std
    det_mask = dominance > dom_thresh
    print(f"Dominance threshold: {dom_thresh:.3f}  (mu={dom_mu:.3f}, std={dom_std:.4f}, {dom_sigma}σ)")

    # Morphological cleanup + labeling
    det_mask = opening(det_mask, footprint=footprint_rectangle((morph_size, morph_size)))
    det_mask = closing(det_mask, footprint=footprint_rectangle((morph_size, morph_size)))
    labeled = label(det_mask)

    return dominance, labeled
    
    
def main():

    # ---------- Configuration ----------
    CONFIG_PATH = Path(__file__).parent / "config.yaml"
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    SICD_PATH = Path(cfg["input"]["sicd_path"])
    CHIP_SIZE = cfg["input"]["chip_size"]

    # ---------- Load and chip ----------
    reader = SICDReader(SICD_PATH)
    rows, cols = reader.get_shape()
    print(f"Image shape: {rows} x {cols}  dtype: {reader.get_dtype()}")

    ext = ChipExtractor(nrows=int(rows), ncols=int(cols))
    region = ext.chip_at_point(
        row=rows // 2,
        col=cols // 2,
        row_width=CHIP_SIZE,
        col_width=CHIP_SIZE,
    )
    print(f"Chip region: rows [{region.row_start}:{region.row_end}], "
        f"cols [{region.col_start}:{region.col_end}]")

    chip = reader.read_chip(
        region.row_start,
        region.row_end,
        region.col_start,
        region.col_end,
    )
    metadata = reader.metadata
    reader.close()


    # ===================== Algorithm =====================

    ###function1
    looks = sublook_data(chip, num_looks=cfg["sublook"]["num_looks"], dimension=cfg["sublook"]["dimension"], metadata=metadata)

    ###function2
    dominance, labeled = dominance_detection(
        looks,
        smooth_win=cfg["detection"]["smooth_win"],
        dom_window=cfg["detection"]["dom_window"],
        dom_sigma=cfg["detection"]["sigma"],
        morph_size=cfg["detection"]["morph_size"],
    )   


    n_detections = labeled.max()
    print(f"Detections: {n_detections}")

    ### CSI composite
    csi = CSIProcessor(metadata, dimension=cfg["sublook"]["dimension"], normalization='log')
    csi_rgb = csi.apply(chip)

    # ===================== Plotting =====================

    num_looks = cfg["sublook"]["num_looks"]
    dom_window = cfg["detection"]["dom_window"]
    dom_sigma = cfg["detection"]["sigma"]

    vmax = np.percentile(np.abs(chip), cfg["display"]["vmax_percentile"])
    dom_floor = dom_window / num_looks

    # Dominance image
    fig_dom, ax_dom = plt.subplots(figsize=(10, 10))
    im = ax_dom.imshow(dominance, cmap='inferno', aspect='auto', vmin=dom_floor, vmax=1)
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

if __name__ == "__main__":
    main()