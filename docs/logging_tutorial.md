# GRDL Logging

GRDL uses Python's standard `logging` module. Every module creates its own logger using `logging.getLogger(__name__)`, which mirrors the package hierarchy. GRDL never configures logging itself — that's the consumer's responsibility.

## For GRDL Users

### Basic Setup

By default, Python's logging emits nothing (level WARNING, no handlers). To see GRDL log output, configure a handler in your application:

```python
import logging

logging.basicConfig(level=logging.INFO)
```

### Controlling GRDL Log Levels

Set the level on `'grdl'` to control all GRDL output:

```python
import logging

# Only see warnings and errors from GRDL
logging.getLogger('grdl').setLevel(logging.WARNING)

# See everything, including debug output
logging.getLogger('grdl').setLevel(logging.DEBUG)
```

### Filtering by Submodule

The logger hierarchy follows the package structure. You can silence GRDL broadly and selectively enable specific submodules:

```python
import logging

logging.getLogger('grdl').setLevel(logging.WARNING)                        # quiet by default
logging.getLogger('grdl.image_processing').setLevel(logging.DEBUG)         # verbose for image processing
logging.getLogger('grdl.IO').setLevel(logging.INFO)                        # moderate for IO
logging.getLogger('grdl.image_processing.pipeline').setLevel(logging.INFO) # override just pipeline
```

More specific loggers override less specific ones.

### Available Logger Namespaces

These map directly to GRDL's package structure:

| Logger Name | Covers |
|---|---|
| `grdl` | Everything |
| `grdl.IO` | All IO readers and writers |
| `grdl.IO.sar` | SAR-specific readers (SICD, CPHD, CRSD, BIOMASS) |
| `grdl.IO.multispectral` | Multispectral readers (VIIRS) |
| `grdl.IO.ir` | IR/thermal readers (ASTER) |
| `grdl.image_processing` | All image processors, pipeline, detection |
| `grdl.image_processing.pipeline` | Pipeline execution |
| `grdl.image_processing.ortho` | Orthorectification |
| `grdl.geolocation` | Coordinate transforms and elevation |
| `grdl.data_prep` | Chipping, tiling, normalization |
| `grdl.coregistration` | Image alignment |

### Example: Quiet Library Usage

```python
import logging

# Application-level logging config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s %(message)s',
)

# Silence GRDL except warnings
logging.getLogger('grdl').setLevel(logging.WARNING)

# Your application code
from grdl.IO.sar.sicd import SICDReader
reader = SICDReader('/path/to/sicd.nitf')
data = reader.read()
# Only warnings/errors from GRDL will appear
```

### Example: Debugging a Specific Module

```python
import logging

logging.basicConfig(level=logging.WARNING)

# Turn on debug for just the pipeline
logging.getLogger('grdl.image_processing.pipeline').setLevel(logging.DEBUG)

from grdl.image_processing.pipeline import ProcessorPipeline
pipeline = ProcessorPipeline([...])
result = pipeline.run(image)
# Debug output only from pipeline; everything else is quiet
```

## For GRDL Developers

### Adding Logging to a Module

Create one logger per module at the top of the file, after imports:

```python
import logging

logger = logging.getLogger(__name__)
```

Then use it throughout the module:

```python
logger.debug("Interpolating %d control points", len(gcps))
logger.info("Loaded %s (%d×%d)", path.name, rows, cols)
logger.warning("No GCPs found, falling back to affine transform")
logger.error("Failed to read tile at offset %d: %s", offset, err)
```

### Choosing the Right Level

| Level | Use When | Example |
|---|---|---|
| `DEBUG` | Internal details useful only for diagnosing problems | Array shapes, intermediate values, timing, iteration counts |
| `INFO` | Confirming normal operations completed | File loaded, processing step finished, resource opened |
| `WARNING` | Unexpected situation but execution continues | Fallback used, deprecated code path, missing optional data |
| `ERROR` | An operation failed but the process can continue | One tile in a batch failed, file unreadable |
| `CRITICAL` | Unrecoverable failure | Should be rare; prefer raising exceptions |

### Rules

1. **Never call `logging.basicConfig()` or add handlers in library code.** The consumer configures logging, not the library.
2. **Never use `print()` for diagnostic output.** Use the logger.
3. **Always use `logging.getLogger(__name__)`**, not a hardcoded string. This keeps loggers aligned with the package tree.
4. **Use lazy formatting.** Write `logger.info("Loaded %s", path)`, not `logger.info(f"Loaded {path}")`. The f-string evaluates even when the log level is disabled; `%s` formatting is deferred.
5. **Log at the right level.** Don't log routine operations at WARNING. Don't log errors at INFO.
6. **Don't log sensitive data.** No file system paths beyond the filename, no credentials, no user data.
7. **Keep messages concise.** One line per log call. Include the relevant values, not a paragraph.

### Image Formation Classes

The SAR image formation classes (`FFBPImageFormation`, `RangeDopplerAlgorithm`, `StripmapPFA`) accept a `verbose` parameter. When `verbose=True` (the default), processing milestones and parameters are emitted via `logging` at INFO and DEBUG levels. To see this output, configure a handler:

```python
import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger('grdl.image_processing.sar.image_formation').setLevel(logging.DEBUG)

from grdl.image_processing.sar.image_formation.ffbp import FFBPImageFormation
ffbp = FFBPImageFormation(metadata)
image = ffbp.form_image(signal)
# Stage milestones at INFO, parameter details at DEBUG
```
