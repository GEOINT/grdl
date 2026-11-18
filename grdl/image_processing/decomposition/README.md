# Polarimetric Decompositions

This package implements polarimetric SAR decompositions and related descriptors.
The families below are deliberately separate: quad-pol algorithms use the full
scattering matrix, dual-pol algorithms use two complex scattering channels, and
compact-pol algorithms use a two-channel covariance matrix with a known transmit
polarization. Their similarly named classes are not interchangeable.

Descriptions and references are adapted from the
[full-pol](../../../../notebooks/grdl/image_processing/sar/full_pol_decomposition_tutorial.ipynb),
[dual-pol](../../../../notebooks/grdl/image_processing/sar/dual_pol_decomposition_tutorial.ipynb), and
[compact-pol](../../../../notebooks/grdl/image_processing/sar/compact_pol_decomposition_tutorial.ipynb)
tutorials in the GRDX companion notebooks repository. These links assume the
repositories are checked out side by side. API names, defaults, units, and output
semantics below follow the current implementation where tutorial text differs.

## Quad-Pol (Full-Pol)

### Shared Inputs and Outputs

Import concrete classes from `grdl.image_processing.decomposition`.
Quad-pol classes inherit [PolarimetricDecomposition](base.py), with
`FullPolHAalpha` also using the shared [HAalphaBase](h_a_alpha_base.py) helpers.

| Input | Meaning and requirements |
| --- | --- |
| `shh` | Complex HH scattering channel, shape `(rows, cols)`. |
| `shv` | Complex HV scattering channel, same shape as `shh`. |
| `svh` | Complex VH scattering channel, same shape as `shh`. |
| `svv` | Complex VV scattering channel, same shape as `shh`. |
| `window_size` | Integer boxcar side length in pixels, odd and in `[1, 31]`. Defaults to `7`, except Pauli (`1`). Matrix-based quad-pol `decompose(...)` requires at least `3`; use a precomputed-matrix method when constructing the processor with `window_size=1`. |
| `t3` | Precomputed complex coherency matrix, shape `(3, 3, rows, cols)`, in the Pauli basis. |
| `c3` | Precomputed complex covariance matrix, shape `(3, 3, rows, cols)`, in the lexicographic basis. |

The normal entry point is `decompose(shh, shv, svh, svv)`. It returns a
`dict[str, numpy.ndarray]` whose keys are the class's `component_names` and whose
values have shape `(rows, cols)`. Pauli returns complex amplitudes from scattering
channels; other quad-pol algorithms return real powers or descriptors. Angles are
in degrees. Power units follow the input calibration; outputs are not implicitly
dB or normalized reflectance.

In the three-component reciprocal matrix representation, the cross-pol term uses
the symmetric HV/VH contribution. Span is the matrix trace: total power in that
representation. Surface means odd-bounce/Bragg-like scattering; double-bounce
means even-bounce/dihedral scattering; volume represents distributed or
depolarizing scattering; helix captures asymmetric/circular scattering. These
labels describe a decomposition, not a guaranteed land-cover classification.

Matrix-based channel entry points build a boxcar-averaged matrix internally.
`decompose_from_t3(t3)` and `decompose_from_c3(c3)` use the supplied matrix without
another boxcar average. As in the full-pol tutorial, construct and filter a matrix
once, then share it across algorithms. Refined Lee and IDAN are external filtering
steps, not constructor options of these decomposition classes. Supply Hermitian,
physically valid covariance/coherency matrices; a shape check alone does not make
a matrix physically meaningful.

```python
import numpy as np
from grdl.image_processing.decomposition import (
    CoherencyMatrix, FullPolHAalpha, FreemanDurden3C, CovarianceMatrix,
)

channels = np.stack([shh, shv, svh, svv], axis=0)
t3 = CoherencyMatrix(window_size=7).compute(channels)
haalpha = FullPolHAalpha(window_size=1).decompose_from_t3(t3)
c3 = CovarianceMatrix(window_size=7).compute(channels)
freeman = FreemanDurden3C(window_size=1).decompose_from_c3(c3)
```

### PauliDecomposition

[Implementation](pauli.py). A coherent, normally pixel-by-pixel Pauli basis
transform. Constructive HH/VV interference emphasizes surface scattering;
destructive interference emphasizes double-bounce; cross-pol emphasizes volume.

**Parameters:** `PauliDecomposition(window_size=1)`. A larger odd window
boxcar-averages the complex channels before mixing, rather than averaging powers.

**Inputs:** `decompose(shh, shv=None, svh=None, svv=None)`. HH and VV are required;
at least one cross-pol channel is required. If only HV or VH is supplied, reciprocity
is assumed and the volume amplitude is `sqrt(2)` times that channel. The universal
quad-pol `execute(...)` interface still requires all four explicit channels.

| Output key | Meaning |
| --- | --- |
| `surface` | Complex amplitude $(S_{HH}+S_{VV})/\sqrt{2}$. |
| `double_bounce` | Complex amplitude $(S_{HH}-S_{VV})/\sqrt{2}$. |
| `volume` | Complex amplitude $(S_{HV}+S_{VH})/\sqrt{2}$. |

**Matrix inputs:** `decompose_from_t3(t3)` and `decompose_from_c3(c3)` return
real magnitudes, not complex amplitudes: square roots of the corresponding Pauli
diagonal powers. Matrix averaging has discarded the original coherent phase.
**Default RGB:** `double_bounce`, `volume`, `surface`.

### FullPolHAalpha

[Implementation](h_a_alpha_fp.py). Cloude-Pottier eigenanalysis of T3 describes
scattering randomness and the probability-weighted mean scattering mechanism.
For descending eigenvalues $\lambda_1\ge\lambda_2\ge\lambda_3$, probabilities are
$p_i=\lambda_i/\sum_j\lambda_j$ and entropy is
$H=-\sum_i p_i\log_3 p_i$. Low entropy indicates a dominant mechanism; high entropy
indicates a mixture. Alpha progresses from surface toward dipole/volume and
double-bounce scattering. Anisotropy describes the secondary mechanisms.

**Parameters:** `FullPolHAalpha(window_size=7)`.
**Inputs:** four channels or `decompose_from_t3(t3)`.

| Output key | Meaning / units |
| --- | --- |
| `entropy` | H, dimensionless, `[0, 1]`. |
| `alpha` | Probability-weighted mean scattering angle, `[0, 90]` degrees. |
| `anisotropy` | $(\lambda_2-\lambda_3)/(\lambda_2+\lambda_3)$, `[0, 1]`. |
| `span` | Sum of the eigenvalues, linear power. |
| `lambda_1`, `lambda_2`, `lambda_3` | Descending eigenvalues, linear power. |

**Default RGB:** `entropy`, `alpha`, `anisotropy`, normalized by their physical
ranges. **Reference:** Cloude and Pottier (1997), shared with the dual-pol tutorial.

### FreemanDurden3C

[Implementation](freeman_durden.py). Model-based C3 decomposition into surface,
double-bounce, and randomly oriented dipole-volume contributions. It works best
when these assumed mechanisms are reasonably separated and target rotation is
limited. Unlike the model-free classes below, it assumes electromagnetic scene
models.

**Parameters:** `FreemanDurden3C(window_size=7)`.
**Inputs:** four channels or `decompose_from_c3(c3)`.
**Outputs:** `surface`, `double_bounce`, and `volume` are real linear scattering
powers; `span` is total linear power.
**Default RGB:** `double_bounce`, `volume`, `surface`.
**Reference:** Freeman and Durden (1998).

### ModelFree3C and ModelFree4C

[Implementation](model_free.py). MF3CF and MF4CF use polarimetric purity and a
scattering-type angle derived directly from T3, without prescribing a volume or
surface scene model. MF4CF additionally allocates helix power using an ellipticity
angle. The tutorial calls these model-free three- and four-component decompositions.

**Parameters:** `ModelFree3C(window_size=7)` or `ModelFree4C(window_size=7)`.
**Inputs:** four channels or `decompose_from_t3(t3)` for either class.

| Class | Output keys and meanings |
| --- | --- |
| `ModelFree3C` | `surface`, `double_bounce`, `volume`: linear powers; `span`: total linear power; `theta_fp`: scattering-type angle, `[-45, 45]` degrees. |
| `ModelFree4C` | `surface`, `double_bounce`, `volume`, `helix`: linear powers; `span`: total linear power; `theta_fp`: scattering-type angle and `tau_fp`: ellipticity angle, each `[-45, 45]` degrees. |

**Default RGB:** `double_bounce`, `volume`, `surface` for both classes; helix is
available separately. **References:** Dey et al. (2020, 2021).

### DegreeOfPolarization

[Implementation](dop.py). The Barakat scalar measures polarimetric purity rather
than allocating scattering powers:

$$m=\sqrt{\max\left(1-\frac{27\det T_3}{(\operatorname{tr}T_3)^3},0\right)}.$$

**Parameters:** `DegreeOfPolarization(window_size=7)`.
**Inputs:** four channels or `decompose_from_t3(t3)`.
**Output:** `dop`, dimensionless in `[0, 1]`: zero denotes complete
depolarization and one a fully polarized/rank-one response.
**Default RGB:** grayscale replication of `dop`.
**Reference:** Barakat (1977).

### ShannonEntropy

[Implementation](shannon_entropy.py). Separates information content into an
intensity contribution and a polarimetric contribution. This is not the normalized
Cloude-Pottier `entropy` output. With $I=\operatorname{tr}T_3$ and
$q=1-27\det T_3/I^3$, the implementation uses:

$$H_I=3\ln(e\pi I/3),\qquad H_P=\ln(1-q).$$

**Parameters:** `ShannonEntropy(window_size=7)`.
**Inputs:** four channels or `decompose_from_t3(t3)`.
**Outputs:** `H_intensity` is $H_I$, `H_polarimetric` is $H_P$, and `H_total`
combines them with `nansum`. Natural-log entropy values are not constrained to
`[0, 1]`; intensity entropy depends on the input power scale. Degenerate or
zero-power pixels can have NaN contributions; a valid other contribution is kept
in `H_total` (two NaN contributions yield zero through `nansum`).
**Default RGB:** `H_total`, `H_intensity`, `H_polarimetric`, percentile stretched.
**References:** Morio et al. (2009); Pottier and Lee (2009).

### NeumannDecomposition

[Implementation](neumann.py). Removes target orientation effects by rotating T3,
then extracts orientation, coherence-ratio, phase, and scattering-diversity
descriptors. It is a parameter decomposition, not a three-power model.

**Parameters:** `NeumannDecomposition(window_size=7)`.
**Inputs:** four channels or `decompose_from_t3(t3)`.

| Output key | Meaning / units |
| --- | --- |
| `psi` | De-orientation angle, `[-45, 45]` degrees. |
| `delta_mod` | Dimensionless magnitude ratio $\sqrt{(T'_{22}+T'_{33})/T_{11}}$ with numerical protection; not a bounded degree of polarization. |
| `delta_pha` | Phase of the rotated T12 coherence, degrees from `atan2`. |
| `tau` | Clipped scattering-diversity/depolarization descriptor, `[0, 1]`. |

**Default RGB:** `psi`, `delta_mod`, `tau`.
**Reference:** Neumann et al. (2010).

### PraksParameters

[Implementation](praks.py). Seven descriptors from the span-normalized covariance
matrix $M=C_3/\operatorname{tr}C_3$. These alternatives to H/A/alpha avoid
eigendecomposition.

**Parameters:** `PraksParameters(window_size=7)`.
**Inputs:** four channels or `decompose_from_c3(c3)`.

| Output key | Implemented quantity |
| --- | --- |
| `frobenius_norm` | Squared Frobenius norm $F=\sum_{ij}|M_{ij}|^2$; `[1/3, 1]` for valid nonzero PSD matrices. |
| `scattering_predominance` | $\sqrt{F}$, dimensionless. |
| `scattering_diversity` | $1.5(1-F)$, dimensionless. |
| `degree_purity` | $2\sqrt{\max(F-0.25,0)}$; not clipped to `[0, 1]` and can reach $\sqrt{3}$. |
| `depolarization_index` | $1-\mathrm{degree\_purity}/\sqrt{3}$, dimensionless. |
| `alpha` | $\arccos(\operatorname{Re}M_{11})$, degrees, with the arccos argument clipped. |
| `entropy` | $2.52+0.78\ln|\det(M+0.16I)|/\ln3$, with numerical protection; not Cloude-Pottier entropy. |

**Default RGB:** `alpha`, `scattering_diversity`, `depolarization_index`.
**Reference:** Praks, Koeniguer, and Hallikainen (2009).

### TouziDecomposition

[Implementation](touzi.py). Coherent Target Scattering Vector Model (CTSVM/TSVM)
parameters are extracted for each T3 eigenvector, followed by probability-weighted
means. Scattering type, phase, helicity, and orientation retain distinct physical
interpretations rather than being reduced to powers.

**Parameters:** `TouziDecomposition(window_size=7)`.
**Inputs:** four channels or `decompose_from_t3(t3)`.
**Outputs:** 16 real angle arrays, all in degrees:

| Output keys | Meaning / implemented range |
| --- | --- |
| `alpha1`, `alpha2`, `alpha3`, `alpha_mean` | Scattering-type angles and weighted mean, `[0, 90]`. |
| `phi1`, `phi2`, `phi3`, `phi_mean` | Scattering-mechanism phases and weighted mean, `[-90, 90]`. |
| `tau1`, `tau2`, `tau3`, `tau_mean` | Helicity/asymmetry angles and weighted mean, `[-45, 45]`. |
| `psi1`, `psi2`, `psi3`, `psi_mean` | Raw orientation angles and weighted mean, `[-90, 90]`; the calculation does not fold orientations into `[-45, 45]`. |

Indices 1, 2, and 3 correspond to descending eigenvalue order.
**Default RGB:** `alpha_mean`, absolute `tau_mean`, `psi_mean`, using physical
display ranges. The orientation display clips to `[-45, 45]`, unlike the raw
output. RGB overrides select three mean parameters; `representation` and
percentile limits are ignored. **Reference:** Touzi (2007).

### Yamaguchi4C

[Implementation](yamaguchi.py). Extends model-based scattering power decomposition
with helix power sensitive to asymmetric targets, including urban structures.

**Parameters:** `Yamaguchi4C(window_size=7, model='y4o')`.
`model` is one of:

| Value | Variant |
| --- | --- |
| `'y4o'` | Original four-component model, without rotation. |
| `'y4r'` | Rotation-corrected model that minimizes real T23 coupling. |
| `'y4s'` | Rotation-corrected model with an additional volume-model selection test. |

**Inputs:** four channels, `decompose_from_t3(t3)`, or
`decompose_from_c3(c3)` (C3 is converted internally to T3).
**Outputs:** `surface`, `double_bounce`, `volume`, `helix`, and `span`, all real
linear powers. **Default RGB:** `double_bounce`, `volume`, `surface`.
The per-pixel solver uses optional numba acceleration when available; otherwise
it falls back to Python. **References:** Yamaguchi et al. (2005, 2011).

## Dual-Pol

### Shared Inputs and Parameters

These seven classes inherit [DualPolDecompositionBase](dual_pol_base.py), not
the compact-pol base. Their entry point is `decompose_dual(s_co, s_cross)`:
`s_co` is a complex co-pol channel (HH or VV), and `s_cross` is its complex
cross-pol channel (HV or VH). Both are NumPy arrays of identical `(rows, cols)`
shape. Detected intensity-only GRD bands are not accepted as these complex inputs.

`window_size=7` is the default for every dual-pol class. Use an odd integer in
`[1, 31]`, except `DualPolHAlpha`, which requires `[3, 31]`. The window averages
internally constructed C2/T2 elements. These classes do not expose
`decompose_from_c2` or `decompose_from_t2`; unlike the quad-pol matrix entry points,
their public decomposition methods take scattering channels.

Outputs are dictionaries of real `(rows, cols)` arrays. The meanings of surface,
double-bounce, volume, span, purity, and entropy build on the quad-pol descriptions,
but two channels cannot recover the full quad-pol information. In particular, the
full-pol nine-zone H/alpha classification is not reliably transferable to dual-pol.

The compatibility shim `decompose(shh, shv, svh, svv)` maps `shh` to `s_co` and
`shv` to `s_cross`, ignoring `svh` and `svv`. Prefer the explicit dual-pol method.

### Algorithms

| Class / implementation | Parameters beyond `window_size=7` | Inputs / computation | Output keys and meanings | Default R / G / B |
| --- | --- | --- | --- | --- |
| [DualPolHAlpha](h_alpha_dp.py) | None. | `decompose_dual(s_co, s_cross)`; T2 eigenanalysis, the two-channel counterpart to `FullPolHAalpha`. | `entropy`: base-2 eigenvalue entropy `[0, 1]`; `alpha`: mean scattering angle `[0, 90]` degrees; `anisotropy`: $(\lambda_1-\lambda_2)/(\lambda_1+\lambda_2)$ in `[0, 1]` (different from full-pol A); `span`: total linear power. | `entropy` / `alpha` / `span`. |
| [DegreeOfPolarizationDP](dop_dp.py) | None. | `decompose_dual(s_co, s_cross)`; Barakat purity from C2. | `dop`: $\sqrt{\max(1-4\det C_2/(\operatorname{tr}C_2)^2,0)}$, clipped to `[0, 1]`. Same purity interpretation as quad-pol, but a two-dimensional matrix. | `dop` replicated. |
| [ShannonEntropyDP](shannon_entropy_dp.py) | None. | `decompose_dual(s_co, s_cross)`; intensity/polarimetric entropy partition from C2. | `H_intensity`: $2\ln(e\pi I/2)$; `H_polarimetric`: $\ln|1-q|$ where $q=1-4\det C_2/I^2$; `H_total`: `nansum` of the two terms. Same NaN and power-scale caveats as `ShannonEntropy`. | `H_total` / `H_intensity` / `H_polarimetric`. |
| [ModelFree3CD](model_free_dp.py) | None. | `decompose_dual(s_co, s_cross)`; model-free purity and scattering-type angle from T2. | `surface`, `double_bounce`, `volume`: linear powers; `theta_dp`: scattering-type angle in degrees. Unlike `ModelFree3C`, there is no `span` output key. | `double_bounce` / `volume` / `surface`. |
| [DualPolRadarBuiltUpIndex](dprbi.py) | `percentile_low=2.0`, `percentile_high=98.0`. | `decompose_dual(s_co, s_cross)`; normalized Stokes magnitudes from C2. | `dprbi`: dimensionless built-up/structured-scattering descriptor, $\sqrt{s_{1n}^2+s_{2n}^2+s_{3n}^2}/\sqrt{3}$ in `[0, 1]`. | `dprbi` replicated. |
| [DualPolRadarSurfaceIndex](dprsi.py) | `nesz_db=-16.0`, `percentile_low=2.0`, `percentile_high=98.0`. | `decompose_dual(s_co, s_cross)`; C2 Stokes structure plus entropy and a noise threshold. | `dprsi`: dimensionless surface-sensitive descriptor. Above NESZ, $(1-H)\sqrt{1-s_{1n}^2}$; at/below NESZ, $\sqrt{1-s_{1n}^2}$. | `dprsi` replicated. |
| [ScatteringPowerDP](scattering_power_dp.py) | `method=1`, `nesz_db=-16.0`, `percentile_low=2.0`, `percentile_high=98.0`. `method` must be `1` or `2`. | `decompose_dual(s_co, s_cross)`; combines DpRBI, DpRSI, and purity. Method 1 partitions power using a characteristic angle; method 2 factorizes dominant/subordinate responses. | Method 1: `alpha` in degrees, `double_bounce`, `surface`, `unpolarized` in linear power. Method 2: `double_bounce`, `surface`, `residual` in linear power. | Method 1: `double_bounce` / `unpolarized` / `surface`; method 2: `double_bounce` / `residual` / `surface`. |

The index constructors' percentiles control the *algorithm's Stokes normalization*,
not just display stretching. The indices therefore depend on the image region
used to calculate percentiles. `nesz_db` is noise-equivalent sigma zero in dB;
the code compares it against `10*log10(C11)`, so meaningful thresholding requires
compatible radiometric calibration.

**References:** Cloude and Pottier (1997), Barakat (1977), Morio et al. (2009), and
Pottier and Lee (2009) apply as described above. `ModelFree3CD` follows Dey et al.
(2020, 2021). The index/scattering-power tutorial cites Bhogapurapu et al. (2021)
and Verma et al. (2024).

## Compact-Pol

### Shared Inputs and Parameters

These five classes inherit [CompactPolDecompositionBase](compact_pol_base.py).
Compact polarimetry uses a single transmit polarization and two received channels
(H and V). It is not equivalent to a conventional co-pol/cross-pol dual-pol pair.
There are primarily two compact-pol configurations: right- or left-circular transmit with linear H/V receive, known as hybrid compact polarization, and linear pi/4 transmit with linear H/V receive, known as (slant) linear compact polarization.
The entry point is `decompose_compact(c11, c12, c21, c22)`.

| Input / constructor parameter | Meaning and requirements |
| --- | --- |
| `c11`, `c22` | Real diagonal powers of the received-channel C2 covariance matrix. |
| `c12` | Complex cross-covariance, $\langle r_H r_V^*\rangle$. |
| `c21` | Conjugate cross-covariance, normally `numpy.conj(c12)`. Required by `decompose_compact(...)`. |
| `window_size=7` | Odd boxcar side length in `[1, 31]`. Applies to the supplied covariance elements. Set `1` for already averaged/externally filtered C2 products to avoid averaging twice. |
| `chi=45.0` | Transmit ellipticity angle in degrees, accepted range `[-90, 90]`. Typical settings: `+45` right-circular, `-45` left-circular, `0` linear. |
| `psi=0.0` | Transmit orientation angle in degrees, accepted range `[-90, 90]`. For linear pi/4 transmit, use `chi=0`, `psi=45`. |

All four covariance inputs must be NumPy arrays with the same `(rows, cols)`
shape. Diagonals represent nonnegative power; off-diagonals must be conjugate
for a physically valid Hermitian matrix. The direct method accepts four elements,
not a single `(2, 2, rows, cols)` matrix argument.

Every compact-pol constructor exposes `window_size`, `chi`, and `psi` with the
defaults above. DoP does not depend on transmit angles. MF3CC, M-Chi, and M-Delta
use the sign of `chi` to interpret handedness, but do not otherwise use `psi`;
S-Omega uses both transmit angles explicitly. Merely accepting a parameter does
not guarantee a model is physically appropriate for that transmit mode.

Canonical Stokes parameters are $S_0=C_{11}+C_{22}$,
$S_1=C_{11}-C_{22}$, $S_2=2\operatorname{Re}C_{12}$, and
$S_3=-2\operatorname{Im}C_{12}$. The compact scattering models then reverse S3
for negative `chi` to obtain their transmit-relative convention. This canonical
sign differs from the older compact tutorial text; do not manually change the
sign of C12 to compensate. The compatibility method
`decompose(shh, shv, svh, svv)` treats those four positional arguments as
`c11`, `c12`, `c21`, `c22`, not as scattering channels.

### Algorithms

| Class / implementation | Description and inputs | Output keys and units | Default R / G / B |
| --- | --- | --- | --- |
| [CompactPolDegreeOfPolarization](dop_cp.py) | `decompose_compact(c11, c12, c21, c22)`; mode-independent Barakat purity from averaged C2. See quad-pol `DegreeOfPolarization`; the determinant coefficient is `4` for C2, not `27`. | `dop`, dimensionless `[0, 1]`. | `dop` replicated. |
| [CompactPolModelFree3C](model_free_cp.py) | `decompose_compact(c11, c12, c21, c22)`; MF3CC, the compact counterpart to quad-pol `ModelFree3C`. Purity and a transmit-relative scattering-type angle divide power without a prescribed scene model. | `surface`, `double_bounce`, `volume`: real linear powers; `theta_cp`: scattering-type angle in degrees. No `span` output key. | `double_bounce` / `volume` / `surface`. |
| [CompactPolSOmega](s_omega_cp.py) | `decompose_compact(c11, c12, c21, c22)`; improved S-Omega partition using polarization state, a polarized fraction, and same-/opposite-circular power ratio (CPR). Branches according to CPR above, below, or equal to one. | `surface`, `double_bounce`, `volume`: real linear powers. Intermediate CPR/omega values are not output keys. | `double_bounce` / `volume` / `surface`. |
| [CompactPolMChi](m_chi_cp.py) | `decompose_compact(c11, c12, c21, c22)`; m-chi partition using purity and received-wave ellipticity. | `surface`, `double_bounce`, `volume`: real nonnegative **amplitudes (square roots of powers)**, not linear powers; `m_cp`: purity `[0, 1]`; `chi_cp`: received ellipticity, `[-45, 45]` degrees. | `double_bounce` / `volume` / `surface`. |
| [CompactPolMDelta](m_delta_cp.py) | `decompose_compact(c11, c12, c21, c22)`; m-delta partition using purity and the complete S2-S3 plane. | `surface`, `double_bounce`, `volume`: real nonnegative **amplitudes (square roots of powers)**; `m_cp`: purity `[0, 1]`; `delta_cp`: `atan2(S3, S2)` in degrees, `[-180, 180]`. The volume amplitude uses $\sqrt{S_0(1-m)/2}$, unlike M-Chi's $\sqrt{S_0(1-m)}$. | `double_bounce` / `volume` / `surface`. |

The tutorial's mode-applicability caveats remain important:

| Algorithm | Circular HCP | Linear compact-pol |
| --- | --- | --- |
| DoP | Applicable. | Applicable; algebraic purity is mode-independent. |
| MF3CC | Applicable. | Purity/volume remain useful; surface/double-bounce interpretation is mode-dependent. |
| S-Omega | Applicable; CPR interpretation assumes circular transmit. | Not physically appropriate as a CPR-based decomposition. |
| M-Chi | Applicable. | Surface/double-bounce separation degrades because S3 loses its circular-transmit interpretation. |
| M-Delta | Applicable. | More portable across modes, but reference-angle interpretation changes. |

**References:** Barakat (1977) for DoP; Dey et al. (2020) for MF3CC;
Raney et al. (2012) for M-Chi; Guo et al. (2012) for M-Delta. Raney (2007),
Nord et al. (2009), Stacy and Preiss (2006), and Charbonneau et al. (2010)
provide architecture, mode-comparison, and application context from the tutorial.

```python
from grdl.image_processing.decomposition import CompactPolMChi

mchi = CompactPolMChi(window_size=1, chi=45.0, psi=0.0)
components = mchi.decompose_compact(c11, c12, c12.conj(), c22)
surface_power = components["surface"] ** 2
rgb, rgb_metadata = mchi.to_rgb(components)
```

## Display and Processor Interfaces

All concrete decomposition classes provide:

```python
rgb, metadata = processor.to_rgb(
    components,
    representation="db",
    percentile_low=2.0,
    percentile_high=98.0,
    channels=None,
)
```

| Parameter / result | Meaning |
| --- | --- |
| `components` | Dictionary returned by the appropriate decomposition method. |
| `representation` | `'db'`, `'magnitude'`, or `'power'` where supported. Amplitude RGB paths use `20*log10(abs(z))` for dB; power RGB paths use `10*log10(P)`. Scalar and angle displays can ignore this option. |
| `percentile_low`, `percentile_high` | Display-stretch percentiles, defaults `2.0` and `98.0`; fixed-range angle/normalized-descriptor channels use their physical ranges instead. |
| `channels` | Optional three valid display component names in R/G/B order. Defaults are listed above; accepted overrides are class-specific (for example, Touzi accepts mean parameters only). |
| `rgb` | Channels-first `(3, rows, cols)` float32 array with values in `[0, 1]`. |
| `metadata` | `ImageMetadata` describing the three output RGB channels. |

Inherited `to_magnitude`, `to_power`, and `to_db` are amplitude conversions:
absolute value, squared absolute value, and `20*log10(abs(z))`, respectively.
Do not blindly call `to_power` on an already-power-valued dictionary or square
angle/index outputs. In particular, M-Chi/M-Delta scattering amplitudes differ
from MF3CC/S-Omega scattering powers even when their keys match.

`execute(metadata, source, **kwargs)` returns `(components, updated_metadata)`.
The result remains a dictionary, not an automatically stacked image. Metadata
describes component names, output dimensions, dtype, and channel lineage, with
`axis_order='CYX'` for a subsequent channel stack. Source cubes can be CYX or YXC
when correctly described by metadata; channel labels support polarization-aware
selection rather than assuming an arbitrary band order.

| Family | Explicit execution keywords | Unlabelled canonical source cube |
| --- | --- | --- |
| Quad-pol | All of `shh`, `shv`, `svh`, `svv`. | Four bands `[HH, HV, VH, VV]`. |
| Dual-pol | Both `s_co`, `s_cross`. | Two bands `[co-pol, cross-pol]`. |
| Compact-pol | `c11`, `c22`, plus complex `c12` or both `c12_real`, `c12_imag`; optional `c21` defaults to `conj(c12)`. | Four **real packed covariance** bands `[C11, Re(C12), Im(C12), C22]`, not four scattering channels. |

## Matrix and Stokes Utilities

[pol_matrix.py](pol_matrix.py) also exports four supporting processors. They
construct polarimetric representations rather than decomposing scattering into
the named outputs above. Each takes `window_size=1` by default; valid windows are
odd integers in `[1, 63]`.

| Class | Input method / parameters | Output |
| --- | --- | --- |
| `CovarianceMatrix` | `compute(channels)`; complex channels-first cube, two channels or reciprocal three-/four-channel full-pol data. | Complex C2 `(2, 2, rows, cols)` or C3 `(3, 3, rows, cols)` covariance in the lexicographic basis. |
| `CoherencyMatrix` | `compute(channels)`; same channel conventions. | Complex T2 `(2, 2, rows, cols)` or T3 `(3, 3, rows, cols)` coherency in the Pauli basis. |
| `StokesVector` | `compute(e_x, e_y)`; two complex field-component arrays of identical `(rows, cols)` shape. | Real Stokes stack `(4, rows, cols)` ordered `[S0, S1, S2, S3]`. |
| `KennaughMatrix` | `compute(channels)`; complex three-/four-channel full-pol cube. | Real Kennaugh matrix `(4, 4, rows, cols)`. |

Use these builders rather than assembling matrix elements manually. For examples
of shared matrix filtering and side-by-side decomposition comparisons, see the
three tutorials linked at the top.

## References

The following reading list reuses the tutorial citations. Where notebook and
module bibliographic details disagree, the list retains author/year and title
without choosing unverified volume/page metadata. The notebooks provide the
longer bibliographies and matrix-filtering references.

- Barakat, R. (1977). *Degree of polarization and the principal idempotents of the coherency matrix.* Optica Acta, 24(5), 537-540. [DOI](https://doi.org/10.1080/713819619).
- Cloude, S.R., and Pottier, E. (1997). *An entropy based classification scheme for land applications of polarimetric SAR.* IEEE Transactions on Geoscience and Remote Sensing, 35(1), 68-78.
- Freeman, A., and Durden, S.L. (1998). *A three-component scattering model for polarimetric SAR data.* IEEE Transactions on Geoscience and Remote Sensing, 36(3), 963-973.
- Dey, S., et al. (2020). *Target characterization and scattering power decomposition of full and compact polarimetric SAR data.* [DOI cited in the compact-pol tutorial](https://doi.org/10.1109/TGRS.2019.2953913).
- Dey, S., et al. (2021). Model-free four-component scattering power decomposition; see the full-pol and dual-pol tutorial citations.
- Morio, J., et al. (2009). Shannon entropy/intensity-polarimetric entropy interpretation; see the full-pol and dual-pol tutorial citations.
- Pottier, E., and Lee, J.S. (2009). *Unsupervised classification scheme of POLSAR images based on the complex Wishart distribution and H/A/Alpha polarimetric decomposition theorem.*
- Neumann, M., et al. (2010). De-orientation decomposition and vegetation coherence parameters; see the full-pol tutorial citation.
- Praks, J., Koeniguer, E.C., and Hallikainen, M.T. (2009). *Alternatives to target entropy and alpha angle in SAR polarimetry.*
- Touzi, R. (2007). *Target scattering decomposition in terms of roll-invariant target parameters.*
- Yamaguchi, Y., et al. (2005). *Four-component scattering model for polarimetric SAR image decomposition.* IEEE Transactions on Geoscience and Remote Sensing, 43(8), 1699-1706.
- Yamaguchi, Y., et al. (2011). *Four-component scattering power decomposition with rotation of coherency matrix.* IEEE Transactions on Geoscience and Remote Sensing, 49(6), 2251-2258.
- Bhogapurapu, N., et al. (2021). *Dual-polarimetric descriptors from Sentinel-1 GRD SAR data for crop growth assessment.* ISPRS Journal of Photogrammetry and Remote Sensing, 178, 20-35.
- Verma, et al. (2024). *Enhanced Target Characterization with Dual-Pol Sentinel-1 SAR Data.* IGARSS 2024, 11461-11464.
- Raney, R.K. (2007). *Hybrid-polarity SAR architecture.* IEEE Transactions on Geoscience and Remote Sensing, 45(11), 3397-3404.
- Raney, R.K., et al. (2012). *The m-chi decomposition of hybrid dual-polarimetric radar data with application to lunar craters.* Journal of Geophysical Research: Planets, 117(E5). [DOI](https://doi.org/10.1029/2011JE003986).
- Guo, R., et al. (2012). *M-delta decomposition for compact polarimetric SAR.* Proceedings of IEEE IGARSS, Munich, Germany.
- Nord, M.E., et al. (2009). *Comparison of compact polarimetric synthetic aperture radar modes.* IEEE Transactions on Geoscience and Remote Sensing, 47(1), 174-188. [DOI](https://doi.org/10.1109/TGRS.2008.2000925).
- Stacy, N., and Preiss, M. (2006). *Compact polarimetric analysis of X-band SAR data.* Proceedings of EUSAR, Dresden, Germany.
- Charbonneau, F.J., et al. (2010). *Compact polarimetry overview and applications assessment.* Canadian Journal of Remote Sensing, 36(S2), S298-S315. [DOI](https://doi.org/10.5589/m10-062).