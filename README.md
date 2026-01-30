# GRDL - GEOINT Rapid Development Library

A modular Python library of fundamental building blocks for deriving geospatial intelligence from any 2D correlated imagery -- SAR, electro-optical, multispectral, hyperspectral, space-based, or terrestrial. GRDL provides composable, well-documented components that developers combine into workflows for image processing, machine learning, geospatial transforms, and sensor exploitation.

## What is GRDL?

GEOINT (Geospatial Intelligence) development requires stitching together domain-specific operations -- reading sensor formats, applying signal processing, transforming coordinate systems, chunking large images, preparing data for ML pipelines. Each project tends to re-implement these operations from scratch.

GRDL solves this by providing a library of **small, focused modules** that each do one thing well. If it's 2D correlated data, GRDL is built to work with it. Developers pick the pieces they need and compose them into workflows without inheriting a monolithic framework.

### Design Principles

- **Modular** -- Each module is self-contained with minimal cross-dependencies. Use only what you need.
- **Verbose** -- Functions and classes document their inputs, outputs, assumptions, and units. No guessing what a parameter expects.
- **Composable** -- Modules are designed to chain together. The output of one operation is a natural input to the next.
- **Sensor-agnostic** -- Built around 2D correlated data as the common abstraction. The same tools work whether the source is a SAR sensor, an EO camera, a multispectral imager, or a terrestrial scanner.
- **Not a framework** -- GRDL is a library, not an application. It doesn't impose structure on your project.

## Module Areas

| Domain | Description |
|--------|-------------|
| **Image Processing** | Filtering, enhancement, normalization, and transforms for any raster imagery |
| **Geospatial Transforms** | Coordinate conversions, projections, pixel-to-geographic mapping |
| **Data Preparation** | Chunking, tiling, resampling, and formatting for ML/AI pipelines |
| **Sensor Processing** | Sensor-specific operations (SAR phase history, EO radiometry, MSI band math) |
| **ML/AI Utilities** | Feature extraction, annotation tools, dataset builders, and model integration helpers |
| **I/O** | Readers and writers for common geospatial and imagery formats |

## Installation

```bash
git clone https://github.com/geoint-org/GRDL.git
```

Add the library to your Python path:

```python
import sys
sys.path.insert(0, "/path/to/GRDL")
```

## Project Structure

```

```

## Dependencies

Core dependencies vary by module. Each module documents its own requirements.

- `numpy` -- Used across all modules
- `scipy` -- Signal processing operations
- Additional dependencies listed per module

## Contributing

GRDL grows one well-built module at a time. When adding a new module:

1. Keep it focused -- one concept per module.
2. Make it sensor-agnostic where possible; sensor-specific logic goes in dedicated submodules.
3. Document inputs, outputs, and units in docstrings.
4. Include example usage.
5. List module-specific dependencies.
6. Add sample data to `example_images/` if needed for demos.

## License

MIT License -- see [LICENSE](LICENSE) for details.
