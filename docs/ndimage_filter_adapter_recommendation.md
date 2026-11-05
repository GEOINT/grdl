# Runtime-Configurable Ndimage Filter Adapter

Status: design recommendation, not an implemented API.

## Recommendation

Introduce an `NdimageFilter` adapter that follows GRDL's existing
`ImageTransform` and bandwise dispatch model. Use a registry of explicitly
supported operations rather than accepting unrestricted function names.

An illustrative runtime configuration is:

```yaml
filter:
  algorithm: ndimage
  parameters:
    operation: gaussian_filter
    kwargs:
      sigma: 1.5
      truncate: 4.0
      mode: reflect
```

This configuration is proposed syntax; it is not currently supported.

## Operation Registry

Each registry entry should define:

- The SciPy callable and an optional CuPy equivalent.
- Parameter validation, including supported keyword arguments.
- Halo calculation, or global execution when finite support is unknown.
- Complex-input support and whether the operation preserves image shape.

Function signatures alone cannot establish these guarantees. Uniform and
Gaussian filters, median filters, derivatives, distance transforms, and
geometric transforms have different input semantics and tiling requirements.
Unsupported operations, parameters, and input types should fail explicitly.

Apply spatial operations independently to each channel, using the existing
bandwise dispatch contract. Do not mix the channel axis into the spatial
filter dimensions.

## User-Provided Kernels

Register `convolve` and `correlate` to support user-provided coefficients through
`weights`, `origin`, and boundary parameters.

Derive the halo from the kernel footprint and origin. Restrict kernels to
spatial axes so channels are not mixed accidentally. Boundary modes that need
opposite image edges, such as `wrap`, require global execution unless the tile
reader explicitly supports periodic access.

## Existing Named Filters

Keep `MeanFilter` and `GaussianFilter` as convenient named APIs, backed by the
same operation definitions as the generic adapter. Preserve their public APIs,
parameter semantics, complex precision, halo declarations, and CPU/GPU behavior.

## PolSAR Constraints

In `polsar_rgb`, only linear, physically appropriate smoothers should enter
`MatrixSmoothing`. Arbitrary ndimage operations do not necessarily preserve
Hermitian, positive-semidefinite polarimetric matrices.

For matrix averaging, apply the same nonnegative spatial weights to every
matrix element. Normalize weights when the intended operation is an average.
Signed derivative kernels and nonlinear operations should not be treated as
polarimetric matrix smoothers merely because they preserve array shape.

## Verification

- Compare supported operations against direct SciPy references.
- Check single-channel and multichannel dispatch without channel mixing.
- Verify complex-value and precision preservation where supported.
- Compare tiled and whole-image results, including boundary pixels and origins.
- Verify actual GPU execution and numerical agreement for supported CuPy paths.
- Check matrix validity for operations exposed through PolSAR matrix smoothing.

This approach provides runtime flexibility without implicitly promising GPU,
complex-data, or tiled compatibility for every SciPy ndimage function.