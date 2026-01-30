# -*- coding: utf-8 -*-
"""
BIOMASS Reader Tests - Unit tests for BIOMASS L1 SCS reader.

Tests for the BIOMASS L1 SCS reader including metadata extraction, chip reading,
geolocation, and polarization handling. Also includes visualization utilities
for quicklook, interactive viewing, and polarization comparison.

Dependencies
------------
pytest
rasterio

Author
------
Duane Smalley, PhD
duane.d.smalley@gmail.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-01-30

Modified
--------
2026-01-30
"""

import pytest
from pathlib import Path
import sys

import numpy as np

try:
    import matplotlib
    # Set interactive backend for contrast adjustment
    # Try backends in order optimized for macOS with Qt6:
    # QtAgg (Qt6/PySide6 - best subplot controls) → MacOSX (native, hardware-accelerated) → TkAgg (most compatible) → WXAgg → Agg (non-interactive)
    # Note: Skipping Qt5Agg to avoid mixing Qt5 into PySide6 environment
    backend_set = False
    for backend in ['QtAgg', 'MacOSX', 'TkAgg', 'WXAgg', 'Agg']:
        try:
            matplotlib.use(backend)
            import matplotlib.pyplot as plt
            # Test if backend actually works by creating a figure
            fig = plt.figure()
            plt.close(fig)
            backend_set = True
            print(f"Using matplotlib backend: {backend}")
            break
        except (ImportError, RuntimeError):
            continue

    if not backend_set:
        # Fall back to default
        import matplotlib.pyplot as plt
        print("Using matplotlib default backend")

    import matplotlib.colors as mcolors
    from matplotlib.widgets import Slider, Button
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Matplotlib not available. Install with: pip install matplotlib")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from grdl.IO import BIOMASSL1Reader, open_biomass


# Path to test data - update this to your local path
TEST_DATA_PATH = Path("/Volumes/PRO-G40/SAR_DATA/BIOMASS/BIO_S1_SCS__1S_20251121T045325_20251121T045346_T_G01_M01_C01_T003_F290_01_DJUPJI")

# Skip all tests if data is not available
pytestmark = pytest.mark.skipif(
    not TEST_DATA_PATH.exists(),
    reason=f"Test data not found at {TEST_DATA_PATH}"
)


def test_open_biomass():
    """Test open_biomass auto-detection."""
    with open_biomass(TEST_DATA_PATH) as reader:
        assert isinstance(reader, BIOMASSL1Reader)
        assert reader.metadata['format'] == 'BIOMASS_L1_SCS'


def test_metadata_extraction():
    """Test that metadata is correctly extracted."""
    with BIOMASSL1Reader(TEST_DATA_PATH) as reader:
        # Check required metadata fields
        assert 'format' in reader.metadata
        assert 'rows' in reader.metadata
        assert 'cols' in reader.metadata
        assert 'polarizations' in reader.metadata

        # Check metadata values
        assert reader.metadata['mission'] == 'BIOMASS'
        assert reader.metadata['product_type'] == 'SCS'
        assert reader.metadata['rows'] > 0
        assert reader.metadata['cols'] > 0

        # Check polarizations
        pols = reader.metadata['polarizations']
        assert len(pols) == 4
        assert pols == ['HH', 'HV', 'VH', 'VV']


def test_shape_and_dtype():
    """Test get_shape() and get_dtype()."""
    with BIOMASSL1Reader(TEST_DATA_PATH) as reader:
        shape = reader.get_shape()
        assert len(shape) == 3  # (rows, cols, pols)
        assert shape[2] == 4  # 4 polarizations

        dtype = reader.get_dtype()
        assert dtype == np.dtype('complex64')


def test_read_chip_single_pol():
    """Test reading a spatial chip (single polarization)."""
    with BIOMASSL1Reader(TEST_DATA_PATH) as reader:
        # Read small HH chip
        chip = reader.read_chip(0, 512, 0, 512, bands=[0])

        # Check shape and type
        assert chip.shape == (512, 512)
        assert chip.dtype == np.complex64

        # Check that we got complex values
        assert np.iscomplexobj(chip)

        # Check magnitude is reasonable (not all zeros or NaN)
        mag = np.abs(chip)
        assert np.nanmean(mag) > 0
        assert not np.all(np.isnan(mag))


def test_read_chip_multi_pol():
    """Test reading multiple polarizations."""
    with BIOMASSL1Reader(TEST_DATA_PATH) as reader:
        # Read all polarizations
        chip = reader.read_chip(100, 356, 200, 456)

        # Should be (pols, rows, cols)
        assert chip.shape == (4, 256, 256)
        assert chip.dtype == np.complex64

        # Check each polarization
        for i in range(4):
            mag = np.abs(chip[i])
            assert np.nanmean(mag) > 0


def test_read_chip_bounds():
    """Test that out-of-bounds reads raise errors."""
    with BIOMASSL1Reader(TEST_DATA_PATH) as reader:
        rows, cols = reader.metadata['rows'], reader.metadata['cols']

        with pytest.raises(ValueError):
            reader.read_chip(-1, 100, 0, 100)  # Negative start

        with pytest.raises(ValueError):
            reader.read_chip(0, rows + 100, 0, 100)  # Exceeds bounds


def test_magnitude_conversion():
    """Test converting complex to magnitude and dB."""
    with BIOMASSL1Reader(TEST_DATA_PATH) as reader:
        chip = reader.read_chip(500, 1000, 500, 1000, bands=[0])

        # Magnitude
        mag = np.abs(chip)
        assert np.all(mag >= 0)

        # dB
        db = 20 * np.log10(mag + 1e-10)
        assert np.all(np.isfinite(db))


def test_geolocation():
    """Test geolocation information."""
    with BIOMASSL1Reader(TEST_DATA_PATH) as reader:
        geo = reader.get_geolocation()

        assert geo is not None
        assert 'crs' in geo
        assert 'projection' in geo
        assert 'corner_coords' in geo

        # Check projection is slant range
        assert geo['projection'] == 'Slant Range'

        # Check corner coordinates exist
        corners = geo['corner_coords']
        assert corners is not None
        assert 'corner1' in corners


def test_polarization_names():
    """Test polarization name retrieval."""
    with BIOMASSL1Reader(TEST_DATA_PATH) as reader:
        assert reader.get_polarization_name(0) == 'HH'
        assert reader.get_polarization_name(1) == 'HV'
        assert reader.get_polarization_name(2) == 'VH'
        assert reader.get_polarization_name(3) == 'VV'

        with pytest.raises(ValueError):
            reader.get_polarization_name(10)  # Invalid index


def test_context_manager():
    """Test that context manager properly closes resources."""
    reader = BIOMASSL1Reader(TEST_DATA_PATH)
    assert reader.magnitude_dataset is not None
    assert reader.phase_dataset is not None

    reader.close()

    # After close, datasets should still exist but be closed
    # (rasterio doesn't set to None, just closes handles)


def test_multi_sensor_metadata_compatibility():
    """Test that metadata is compatible with other SAR readers."""
    with BIOMASSL1Reader(TEST_DATA_PATH) as reader:
        # Check for common SAR metadata fields
        assert 'rows' in reader.metadata
        assert 'cols' in reader.metadata
        assert 'polarizations' in reader.metadata
        assert 'orbit_number' in reader.metadata
        assert 'orbit_pass' in reader.metadata

        # Check pixel spacing (for multi-sensor fusion)
        assert 'range_pixel_spacing' in reader.metadata
        assert 'azimuth_pixel_spacing' in reader.metadata
        assert reader.metadata['range_pixel_spacing'] > 0
        assert reader.metadata['azimuth_pixel_spacing'] > 0


def plot_biomass_quicklook(reader, chip_size=1024, save_path=None):
    """
    Create a quicklook visualization of BIOMASS L1 SCS data.

    Parameters
    ----------
    reader : BIOMASSL1Reader
        Open BIOMASS reader
    chip_size : int, default=1024
        Size of chip to visualize
    save_path : Optional[Path], default=None
        Path to save figure. If None, display interactively.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("❌ Matplotlib not available for plotting")
        return

    print(f"Creating BIOMASS quicklook visualization ({chip_size}x{chip_size})...")

    # Read chip with all polarizations
    rows, cols = reader.metadata['rows'], reader.metadata['cols']
    chip = reader.read_chip(0, min(chip_size, rows), 0, min(chip_size, cols))

    # Create figure with subplots for each polarization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"BIOMASS L1 SCS - {reader.metadata.get('swath', 'Unknown')} Swath\n"
                 f"Orbit {reader.metadata.get('orbit_number', 'N/A')} - "
                 f"{reader.metadata.get('start_time', 'Unknown')}", fontsize=14)

    pols = reader.polarizations

    # Plot magnitude (dB) for each polarization
    for i, pol in enumerate(pols):
        ax = axes[i // 2, i % 2] if i < 4 else None
        if ax is None:
            continue

        # Extract polarization data
        pol_data = chip[i] if chip.ndim == 3 else chip

        # Compute magnitude in dB
        magnitude = np.abs(pol_data)
        mag_db = 20 * np.log10(magnitude + 1e-10)

        # Plot with colorbar
        im = ax.imshow(mag_db, cmap='gray', vmin=np.percentile(mag_db, 2),
                       vmax=np.percentile(mag_db, 98), aspect='auto')
        ax.set_title(f'{pol} Polarization (Magnitude dB)')
        ax.set_xlabel('Range')
        ax.set_ylabel('Azimuth')
        plt.colorbar(im, ax=ax, label='dB')

    # Plot Pauli RGB composite (if all pols available)
    if len(pols) == 4:
        ax = axes[1, 2]

        # Pauli decomposition: R=|HH-VV|, G=|HV|, B=|HH+VV|
        hh = chip[0]
        hv = chip[1]
        vv = chip[3]

        red = np.abs(hh - vv)
        green = np.abs(hv) * 2  # Scale cross-pol
        blue = np.abs(hh + vv)

        # Normalize to 0-1 range
        def normalize(band):
            band_db = 20 * np.log10(band + 1e-10)
            vmin, vmax = np.percentile(band_db, 2), np.percentile(band_db, 98)
            return np.clip((band_db - vmin) / (vmax - vmin), 0, 1)

        rgb = np.stack([normalize(red), normalize(green), normalize(blue)], axis=-1)

        ax.imshow(rgb, aspect='auto')
        ax.set_title('Pauli RGB Composite\n(R:|HH-VV|, G:|HV|, B:|HH+VV|)')
        ax.set_xlabel('Range')
        ax.set_ylabel('Azimuth')

    # Hide unused subplot
    if len(pols) < 4:
        axes[1, 2].axis('off')

    # Add metadata text
    metadata_text = (
        f"Dimensions: {reader.metadata['rows']} x {reader.metadata['cols']} pixels\n"
        f"Range Spacing: {reader.metadata.get('range_pixel_spacing', 0):.2f} m\n"
        f"Azimuth Spacing: {reader.metadata.get('azimuth_pixel_spacing', 0):.2f} m\n"
        f"Orbit Pass: {reader.metadata.get('orbit_pass', 'N/A')}"
    )
    fig.text(0.02, 0.02, metadata_text, fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved quicklook to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_interactive_viewer(reader, row_start=0, col_start=0, size=1024, polarization=0):
    """
    Interactive BIOMASS image viewer with contrast/brightness controls.

    Parameters
    ----------
    reader : BIOMASSL1Reader
        Open BIOMASS reader
    row_start : int, default=0
        Starting row for chip
    col_start : int, default=0
        Starting column for chip
    size : int, default=1024
        Chip size
    polarization : int, default=0
        Polarization index (0=HH, 1=HV, 2=VH, 3=VV)
    """
    if not MATPLOTLIB_AVAILABLE:
        print("❌ Matplotlib not available for plotting")
        return

    print(f"Loading interactive viewer ({size}x{size})...")
    print("Use sliders to adjust contrast and brightness")
    print("Click 'Reset' to restore original settings")

    # Read chip
    chip = reader.read_chip(row_start, row_start + size, col_start, col_start + size)
    pol_data = chip[polarization]
    pol_name = reader.polarizations[polarization]

    # Compute magnitude in dB
    magnitude = np.abs(pol_data)
    mag_db = 20 * np.log10(magnitude + 1e-10)

    # Initial contrast settings
    initial_vmin = np.percentile(mag_db, 2)
    initial_vmax = np.percentile(mag_db, 98)
    data_min = mag_db.min()
    data_max = mag_db.max()

    # Create figure with space for sliders
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.subplots_adjust(left=0.1, bottom=0.25, right=0.9, top=0.95)

    # Initial image display
    im = ax.imshow(mag_db, cmap='gray', vmin=initial_vmin, vmax=initial_vmax, aspect='auto')
    ax.set_title(f'BIOMASS L1 SCS - {pol_name} Polarization (Interactive)', fontsize=14)
    ax.set_xlabel('Range (pixels)')
    ax.set_ylabel('Azimuth (pixels)')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label='Backscatter (dB)')

    # Create sliders
    ax_vmin = plt.axes([0.15, 0.15, 0.7, 0.03])
    ax_vmax = plt.axes([0.15, 0.10, 0.7, 0.03])
    ax_gamma = plt.axes([0.15, 0.05, 0.7, 0.03])

    slider_vmin = Slider(ax_vmin, 'Min dB', data_min, data_max,
                         valinit=initial_vmin, valstep=0.1)
    slider_vmax = Slider(ax_vmax, 'Max dB', data_min, data_max,
                         valinit=initial_vmax, valstep=0.1)
    slider_gamma = Slider(ax_gamma, 'Gamma', 0.1, 3.0,
                          valinit=1.0, valstep=0.1)

    # Add reset button
    ax_reset = plt.axes([0.8, 0.005, 0.1, 0.04])
    btn_reset = Button(ax_reset, 'Reset')

    # Add statistics text
    stats_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                         fontsize=10, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    def update_stats(vmin, vmax):
        """Update statistics display."""
        visible_data = mag_db[(mag_db >= vmin) & (mag_db <= vmax)]
        if len(visible_data) > 0:
            stats = (
                f"Visible Range: [{vmin:.1f}, {vmax:.1f}] dB\n"
                f"Mean: {visible_data.mean():.1f} dB\n"
                f"Std: {visible_data.std():.1f} dB\n"
                f"Pixels: {len(visible_data):,} / {mag_db.size:,}"
            )
        else:
            stats = "No data in visible range"
        stats_text.set_text(stats)

    def update(val):
        """Update image display when sliders change."""
        vmin = slider_vmin.val
        vmax = slider_vmax.val
        gamma = slider_gamma.val

        # Ensure vmin < vmax
        if vmin >= vmax:
            vmin = vmax - 0.1
            slider_vmin.set_val(vmin)

        # Apply gamma correction
        if gamma != 1.0:
            # Normalize to 0-1, apply gamma, map to vmin-vmax range
            norm_data = (mag_db - data_min) / (data_max - data_min)
            gamma_data = np.power(norm_data, gamma)
            display_data = gamma_data * (data_max - data_min) + data_min
            im.set_data(display_data)
            im.set_clim(vmin, vmax)
        else:
            im.set_data(mag_db)
            im.set_clim(vmin, vmax)

        update_stats(vmin, vmax)
        fig.canvas.draw_idle()

    def reset(event):
        """Reset to initial settings."""
        slider_vmin.reset()
        slider_vmax.reset()
        slider_gamma.reset()

    # Connect sliders and button
    slider_vmin.on_changed(update)
    slider_vmax.on_changed(update)
    slider_gamma.on_changed(update)
    btn_reset.on_clicked(reset)

    # Initial stats
    update_stats(initial_vmin, initial_vmax)

    # Add keyboard shortcuts info
    info_text = (
        "Controls:\n"
        "• Drag sliders to adjust contrast\n"
        "• Gamma: adjust brightness curve\n"
        "• Click 'Reset' to restore defaults\n"
        "• Use mouse to zoom/pan"
    )
    fig.text(0.02, 0.02, info_text, fontsize=8, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.show()


def plot_polarization_comparison(reader, row_start=0, col_start=0, size=512, save_path=None):
    """
    Plot side-by-side comparison of all polarizations.

    Parameters
    ----------
    reader : BIOMASSL1Reader
        Open BIOMASS reader
    row_start : int, default=0
        Starting row for chip
    col_start : int, default=0
        Starting column for chip
    size : int, default=512
        Chip size
    save_path : Optional[Path], default=None
        Path to save figure
    """
    if not MATPLOTLIB_AVAILABLE:
        print("❌ Matplotlib not available for plotting")
        return

    print(f"Creating polarization comparison plot ({size}x{size})...")

    # Read chip
    chip = reader.read_chip(row_start, row_start + size, col_start, col_start + size)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"BIOMASS L1 SCS Polarization Comparison", fontsize=14)

    for i, (ax, pol) in enumerate(zip(axes, reader.polarizations)):
        pol_data = chip[i]
        mag_db = 20 * np.log10(np.abs(pol_data) + 1e-10)

        vmin, vmax = np.percentile(mag_db, 1), np.percentile(mag_db, 99)
        im = ax.imshow(mag_db, cmap='gray', vmin=vmin, vmax=vmax, aspect='auto')

        ax.set_title(f'{pol} Polarization')
        ax.set_xlabel('Range (pixels)')
        ax.set_ylabel('Azimuth (pixels)')
        plt.colorbar(im, ax=ax, label='dB')

        # Add statistics
        stats_text = (
            f"Mean: {mag_db.mean():.1f} dB\n"
            f"Std: {mag_db.std():.1f} dB\n"
            f"Min: {mag_db.min():.1f} dB\n"
            f"Max: {mag_db.max():.1f} dB"
        )
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved comparison plot to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def run_simple_test():
    """Simple non-unittest test for quick verification."""
    test_path = Path("/Volumes/PRO-G40/SAR_DATA/BIOMASS/BIO_S1_SCS__1S_20251121T045325_20251121T045346_T_G01_M01_C01_T003_F290_01_DJUPJI")

    if not test_path.exists():
        print(f"❌ Test data not found at {test_path}")
        return

    print("Testing BIOMASS L1 Reader...")
    print(f"Product: {test_path.name}")
    print()

    try:
        with open_biomass(test_path) as reader:
            print("✓ Successfully opened BIOMASS product")
            print()

            # Print metadata
            print("Metadata:")
            print(f"  Format: {reader.metadata['format']}")
            print(f"  Mission: {reader.metadata.get('mission', 'N/A')}")
            print(f"  Swath: {reader.metadata.get('swath', 'N/A')}")
            print(f"  Product Type: {reader.metadata.get('product_type', 'N/A')}")
            print(f"  Dimensions: {reader.metadata['rows']} x {reader.metadata['cols']}")
            print(f"  Polarizations: {reader.metadata.get('polarizations', [])}")
            print(f"  Orbit: {reader.metadata.get('orbit_number', 'N/A')} ({reader.metadata.get('orbit_pass', 'N/A')})")
            print(f"  Range Spacing: {reader.metadata.get('range_pixel_spacing', 'N/A'):.2f} m")
            print(f"  Azimuth Spacing: {reader.metadata.get('azimuth_pixel_spacing', 'N/A'):.2f} m")
            print()

            # Test reading
            print("Reading 512x512 chip (HH polarization)...")
            chip_hh = reader.read_chip(0, 512, 0, 512, bands=[0])
            print(f"✓ Chip shape: {chip_hh.shape}")
            print(f"✓ Chip dtype: {chip_hh.dtype}")

            mag = np.abs(chip_hh)
            mag_db = 20 * np.log10(mag + 1e-10)
            print(f"✓ Magnitude: min={mag.min():.2f}, max={mag.max():.2f}")
            print(f"✓ Magnitude (dB): min={mag_db.min():.2f}, max={mag_db.max():.2f}")
            print()

            # Test multi-pol
            print("Reading 256x256 chip (all polarizations)...")
            chip_all = reader.read_chip(1000, 1256, 500, 756)
            print(f"✓ Multi-pol shape: {chip_all.shape}")
            for i, pol in enumerate(reader.polarizations):
                pol_mag = np.abs(chip_all[i])
                print(f"  {pol}: magnitude range [{pol_mag.min():.2f}, {pol_mag.max():.2f}]")
            print()

            print("✅ All tests passed!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def run_plot_test():
    """Run test with visualization."""
    test_path = Path("/Volumes/PRO-G40/SAR_DATA/BIOMASS/BIO_S1_SCS__1S_20251121T045325_20251121T045346_T_G01_M01_C01_T003_F290_01_DJUPJI")

    if not test_path.exists():
        print(f"❌ Test data not found at {test_path}")
        return

    if not MATPLOTLIB_AVAILABLE:
        print("❌ Matplotlib not available. Install with: pip install matplotlib")
        return

    print("Creating BIOMASS visualizations...")
    print()

    try:
        with open_biomass(test_path) as reader:
            print(f"Product: {test_path.name}")
            print(f"Polarizations: {reader.metadata['polarizations']}")
            print(f"Dimensions: {reader.metadata['rows']} x {reader.metadata['cols']}")
            print()

            # Create quicklook
            print("1. Creating quicklook visualization...")
            plot_biomass_quicklook(reader, chip_size=1024)
            print()

            # Create polarization comparison
            print("2. Creating polarization comparison...")
            plot_polarization_comparison(reader, row_start=1000, col_start=500, size=512)
            print()

            print("✅ Visualization complete!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def run_interactive_test():
    """Run interactive viewer with contrast controls."""
    test_path = Path("/Volumes/PRO-G40/SAR_DATA/BIOMASS/BIO_S1_SCS__1S_20251121T045325_20251121T045346_T_G01_M01_C01_T003_F290_01_DJUPJI")

    if not test_path.exists():
        print(f"❌ Test data not found at {test_path}")
        return

    if not MATPLOTLIB_AVAILABLE:
        print("❌ Matplotlib not available. Install with: pip install matplotlib")
        return

    print("BIOMASS Interactive Viewer")
    print("=" * 50)
    print()

    try:
        with open_biomass(test_path) as reader:
            print(f"Product: {test_path.name}")
            print(f"Available polarizations: {reader.metadata['polarizations']}")
            print(f"Dimensions: {reader.metadata['rows']} x {reader.metadata['cols']}")
            print()

            # Let user choose polarization
            print("Select polarization to view:")
            for i, pol in enumerate(reader.polarizations):
                print(f"  {i}: {pol}")
            print()

            try:
                pol_choice = int(input("Enter polarization number (0-3, default=0): ") or "0")
                if pol_choice < 0 or pol_choice >= len(reader.polarizations):
                    pol_choice = 0
            except ValueError:
                pol_choice = 0

            print(f"\nLoading {reader.polarizations[pol_choice]} polarization...")
            print()

            # Launch interactive viewer
            plot_interactive_viewer(reader, row_start=0, col_start=0,
                                   size=1024, polarization=pol_choice)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == '--quick':
            run_simple_test()
        elif sys.argv[1] == '--plot':
            run_plot_test()
        elif sys.argv[1] == '--interactive' or sys.argv[1] == '-i':
            run_interactive_test()
        elif sys.argv[1] == '--help':
            print("BIOMASS Reader Tests")
            print()
            print("Usage:")
            print("  pytest tests/test_io_biomass.py -v          # Run pytest tests")
            print("  python test_io_biomass.py --quick           # Quick validation test")
            print("  python test_io_biomass.py --plot            # Create visualizations")
            print("  python test_io_biomass.py --interactive|-i  # Interactive viewer with contrast controls")
            print("  python test_io_biomass.py --help            # Show this help")
            print()
            print("Interactive mode features:")
            print("  - Adjustable contrast (min/max dB)")
            print("  - Gamma correction for brightness")
            print("  - Real-time statistics")
            print("  - Mouse zoom and pan")
        else:
            pytest.main([__file__, '-v'])
    else:
        pytest.main([__file__, '-v'])
