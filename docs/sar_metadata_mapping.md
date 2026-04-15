# GRDL SAR Metadata Parameter Mapping

Cross-reference of metadata parameters across SAR input formats supported by `grdl`.
Identifies common (mappable) parameters and format-exclusive parameters.

## 1. Source Model Files (SAR)

| Format | Metadata Class | Model File |
|---|---|---|
| Base | `ImageMetadata` | `grdl/IO/models/base.py` |
| SICD | `SICDMetadata` | `grdl/IO/models/sicd.py` |
| SIDD | `SIDDMetadata` | `grdl/IO/models/sidd.py` |
| CPHD | `CPHDMetadata` | `grdl/IO/models/cphd.py` |
| CRSD | `ImageMetadata` | `grdl/IO/sar/crsd.py` (uses base class only) |
| Sentinel-1 SLC | `Sentinel1SLCMetadata` | `grdl/IO/models/sentinel1_slc.py` |
| TerraSAR-X/TDX | `TerraSARMetadata` | `grdl/IO/models/terrasar.py` |
| BIOMASS | `BIOMASSMetadata` | `grdl/IO/models/biomass.py` |
| NISAR | `NISARMetadata` | `grdl/IO/models/nisar.py` |
| GeoTIFF (SAR fallback) | `ImageMetadata` | uses base class only |

---

## 2. Universal Base Parameters (All Formats)

From `ImageMetadata` in `grdl/IO/models/base.py` - every format inherits these:

| Parameter | Type | Description |
|---|---|---|
| `format` | `str` | Format identifier (e.g., `GeoTIFF`, `SICD`, `HDF5`) |
| `rows` | `int` | Number of image rows (lines) |
| `cols` | `int` | Number of image columns (samples) |
| `dtype` | `str` | NumPy dtype string (e.g., `float32`, `complex64`) |
| `bands` | `int?` | Number of spectral bands |
| `crs` | `str?` | Coordinate reference system string (e.g., `EPSG:4326`) |
| `nodata` | `float?` | No-data sentinel value |
| `transform` | `Any?` | Pixel-to-world transform (Affine or equivalent) |
| `bounds` | `Any?` | Spatial bounds object/tuple |
| `pixel_resolution` | `Any?` | Pixel spacing/resolution metadata |
| `extras` | `dict` | Catch-all for additional format-specific fields |

---

## 3. Common Parameters Mappable Across SAR Formats

These semantic concepts exist in multiple SAR formats with different access paths.

### 3.1 Acquisition Start/Stop Time

| Format | Access Path |
|---|---|
| SICD | `timeline.collect_start` + `timeline.collect_duration` |
| CPHD | `pvp.tx_time` / `pvp.rcv_time` (per-pulse arrays) |
| SIDD | `exploitation_features.collections[].collection_date_time` + `collection_duration` |
| Sentinel-1 | `product_info.start_time` / `product_info.stop_time` |
| TerraSAR | `product_info.start_time_utc` / `product_info.stop_time_utc` |
| BIOMASS | `start_time` / `stop_time` |
| NISAR | `identification.zero_doppler_start_time` / `identification.zero_doppler_end_time` |

### 3.2 Platform / Mission Name

| Format | Access Path |
|---|---|
| SICD | `collection_info.collector_name` |
| CPHD | `collection_info.collector_name` |
| SIDD | `exploitation_features.collections[].sensor_name` |
| Sentinel-1 | `product_info.mission` (S1A/S1B/S1C) |
| TerraSAR | `product_info.mission` + `product_info.satellite` |
| BIOMASS | `mission` |
| NISAR | `identification.mission_id` |

### 3.3 Radar Mode

| Format | Access Path |
|---|---|
| SICD | `collection_info.radar_mode.mode_type` (SPOTLIGHT / STRIPMAP / DYNAMIC STRIPMAP) |
| CPHD | `collection_info.radar_mode` |
| SIDD | `exploitation_features.collections[].radar_mode.mode_type` |
| Sentinel-1 | `product_info.mode` (IW / EW / SM) |
| TerraSAR | `product_info.imaging_mode` (SM / HS / SL / SC / ST) |

### 3.4 Polarization

| Format | Access Path |
|---|---|
| SICD | `radar_collection.tx_polarization` + `radar_collection.rcv_channels[].tx_rcv_polarization` |
| SIDD | `exploitation_features.collections[].polarizations[].tx_polarization / rcv_polarization` |
| Sentinel-1 | `swath_info.polarization` |
| TerraSAR | `product_info.polarization_list` + `product_info.polarization_mode` |
| BIOMASS | `polarizations` + `num_polarizations` |
| NISAR | `polarization`, `available_polarizations`, `swath_parameters.polarizations` |

### 3.5 Orbit Direction (Ascending/Descending)

| Format | Access Path |
|---|---|
| Sentinel-1 | `product_info.orbit_pass` |
| TerraSAR | `product_info.orbit_direction` |
| BIOMASS | `orbit_pass` |
| NISAR | `identification.orbit_pass_direction` |

### 3.6 Orbit Number

| Format | Access Path |
|---|---|
| Sentinel-1 | `product_info.absolute_orbit` + `product_info.relative_orbit` |
| TerraSAR | `product_info.absolute_orbit` |
| BIOMASS | `orbit_number` |
| NISAR | `identification.absolute_orbit_number` + `identification.track_number` |

### 3.7 Orbit State Vectors (Platform Position/Velocity)

| Format | Representation |
|---|---|
| SICD | `position.arp_poly` - polynomial form (XYZPoly) |
| CPHD | `pvp.tx_pos`, `pvp.rcv_pos`, `pvp.tx_vel`, `pvp.rcv_vel` - per-pulse arrays |
| SIDD | `measurement.arp_poly` - polynomial form |
| Sentinel-1 | `orbit_state_vectors[]` - discrete time/position/velocity |
| TerraSAR | `orbit_state_vectors[]` - discrete time/position/velocity |
| NISAR | `orbit.time`, `orbit.position`, `orbit.velocity` - dense arrays |

### 3.8 Geolocation / Image Corners

| Format | Access Path |
|---|---|
| SICD | `geo_data.image_corners` + `geo_data.scp` (scene center point) |
| CPHD | `scene_coordinates.corner_points` + `scene_coordinates.iarp_ecf / iarp_llh` |
| SIDD | `geo_data.image_corners` |
| Sentinel-1 | `geolocation_grid[]` (tie-point grid: line/pixel/lat/lon/height) |
| TerraSAR | `scene_info.scene_extent` + `geolocation_grid[]` |
| BIOMASS | `corner_coords` + `gcps` |
| NISAR | `geolocation_grid.coordinate_x / coordinate_y` (3D grid with height planes) |

### 3.9 Pixel / Sample Spacing

| Format | Access Path |
|---|---|
| SICD | `grid.row.ss` (range), `grid.col.ss` (azimuth) |
| SIDD | `measurement.plane_projection.sample_spacing.row / col` |
| Sentinel-1 | `swath_info.range_pixel_spacing` / `swath_info.azimuth_pixel_spacing` |
| TerraSAR | `image_info.row_spacing` / `image_info.col_spacing` |
| BIOMASS | `range_pixel_spacing` / `azimuth_pixel_spacing` |
| NISAR (RSLC) | `swath_parameters.slant_range_spacing` / `swath_parameters.scene_center_along_track_spacing` |
| NISAR (GSLC) | `grid_parameters.x_coordinate_spacing` / `grid_parameters.y_coordinate_spacing` |

### 3.10 Incidence / Grazing Angle

| Format | Access Path |
|---|---|
| SICD | `scpcoa.incidence_ang` / `scpcoa.graze_ang` |
| CPHD | `reference_geometry.graze_angle_deg` |
| SIDD | `exploitation_features.collections[].geometry.graze` |
| Sentinel-1 | `swath_info.incidence_angle_mid` + `geolocation_grid[].incidence_angle` |
| TerraSAR | `scene_info.incidence_angle_near / far / center` + `geolocation_grid[].incidence_angle` |
| NISAR | `geolocation_grid.incidence_angle` (grid) |

### 3.11 Center Frequency / Radar Frequency

| Format | Access Path |
|---|---|
| SICD | `radar_collection.tx_frequency.min / max` (derive center) |
| CPHD | `global_params.center_frequency` (property from `fx_band_min / max`) |
| Sentinel-1 | `swath_info.radar_frequency` |
| TerraSAR | `radar_params.center_frequency` |
| NISAR | `swath_parameters.acquired_center_frequency` / `swath_parameters.processed_center_frequency` |

### 3.12 Bandwidth

| Format | Access Path |
|---|---|
| SICD | `radar_collection.waveform[].tx_rf_bandwidth` |
| CPHD | `global_params.bandwidth` (property from `fx_band_min / max`) |
| TerraSAR | `radar_params.range_bandwidth` |
| NISAR | `swath_parameters.acquired_range_bandwidth` / `swath_parameters.processed_range_bandwidth` |

### 3.13 PRF (Pulse Repetition Frequency)

| Format | Access Path |
|---|---|
| SICD | `timeline.ipp[].ipp_poly` (implicit) |
| Sentinel-1 | `swath_info.azimuth_frequency` |
| TerraSAR | `radar_params.prf` |
| BIOMASS | `prf` |
| NISAR | `swath_parameters.nominal_acquisition_prf` |

### 3.14 Radiometric Calibration

| Format | Representation |
|---|---|
| SICD | `radiometric` - sigma0/beta0/gamma0 scale factor polynomials (Poly2D) |
| SIDD | `radiometric` - reuses SICD radiometric type |
| Sentinel-1 | `calibration_vectors[]` - LUTs (`sigma_nought`, `beta_nought`, `gamma`, `dn`) |
| TerraSAR | `calibration.calibration_constant` (scalar Ks) |
| NISAR | `calibration.sigma0 / beta0 / gamma0` (grids) |

### 3.15 Noise

| Format | Access Path |
|---|---|
| SICD | `radiometric.noise_level` (type + polynomial) |
| Sentinel-1 | `noise_range_vectors[]` + `noise_azimuth_vectors[]` (separate LUTs) |
| TerraSAR | `calibration.noise_equivalent_beta_nought` (scalar NESZ) |

### 3.16 Doppler Centroid

| Format | Access Path |
|---|---|
| SICD | `rma.inca.dop_centroid_poly` (Poly2D) |
| CPHD | `pvp.a_fdop` (per-pulse array) |
| Sentinel-1 | `doppler_centroids[]` (polynomial per azimuth time) |
| TerraSAR | `doppler_info.doppler_centroid_coefficients` |

### 3.17 Look Direction / Side of Track

| Format | Access Path |
|---|---|
| SICD | `scpcoa.side_of_track` (L/R) |
| CPHD | `reference_geometry.side_of_track` |
| TerraSAR | `product_info.look_direction` (RIGHT/LEFT) |
| NISAR | `identification.look_direction` (right/left) |

### 3.18 Classification

| Format | Access Path |
|---|---|
| SICD | `collection_info.classification` |
| CPHD | `collection_info.classification` |
| SIDD | `product_creation.classification.classification` |

### 3.19 Product Type

| Format | Access Path |
|---|---|
| Sentinel-1 | `product_info.product_type` |
| TerraSAR | `product_info.product_type` (SSC / MGD / GEC / EEC) |
| BIOMASS | `product_type` (SCS) |
| NISAR | `product_type` / `identification.product_type` (RSLC / GSLC) |

### 3.20 Processing Software / Version

| Format | Access Path |
|---|---|
| SICD | `image_creation.application` |
| Sentinel-1 | `product_info.ipf_version` |
| TerraSAR | `product_info.processor_version` / `processing_info.processor_version` |
| NISAR | `processing_info.software_version` |

---

## 4. Format-Exclusive Parameters

### 4.1 SICD-Only (NGA Standard Sections)

**Top-level section handles:**
- `backend`, `collection_info`, `image_creation`, `image_data`, `geo_data`, `grid`, `timeline`, `position`, `radar_collection`, `image_formation`, `scpcoa`, `radiometric`, `antenna`, `error_statistics`, `match_info`, `rg_az_comp`, `pfa`, `rma`

**ImageCreation:**
- `application`, `date_time`, `site`, `profile`

**Grid (spatial frequency domain):**
- `image_plane`, `type`
- Row/col direction parameters: `uvect_ecf`, `imp_resp_wid`, `imp_resp_bw`, `k_ctr`, `delta_k1`, `delta_k2`, `delta_k_coa_poly`, `wgt_type`, `wgt_funct`
- `time_coa_poly`

**Timeline / IPP:**
- `ipp[].t_start`, `t_end`, `ipp_start`, `ipp_end`, `ipp_poly`

**Position polynomials:**
- `arp_poly`, `grp_poly`, `tx_apc_poly`, `rcv_apc`

**SCPCOA geometry:**
- `scp_time`, `arp_pos`, `arp_vel`, `arp_acc`
- `slant_range`, `ground_range`, `doppler_cone_ang`
- `twist_ang`, `slope_ang`, `azim_ang`, `layover_ang`

**Antenna (tx/rcv/two-way):**
- `x_axis_poly`, `y_axis_poly`, `freq_zero`, `eb` (electrical boresight)
- `array` (gain/phase polynomial), `elem` (gain/phase polynomial)
- `gain_bs_poly`, `eb_freq_shift`, `ml_freq_dilation`

**ErrorStatistics:**
- `composite_scp` (rg, az, rg_az)
- `monostatic` (PosVelErr with correlation coefficients)
- `radar_sensor` (range_bias, clock_freq_sf, transmit_freq_sf, range_bias_decorr)

**MatchInfo:**
- `match_types[].type_id`, `current_index`, `match_collections[]`

**ImageFormation:**
- `image_form_algo` (PFA / RGAZCOMP / RMA)
- `t_start_proc`, `t_end_proc`, `tx_frequency_proc`
- `image_beam_comp`, `az_autofocus`, `rg_autofocus`
- `rcv_chan_proc`, `polarization_hv_angle_poly`, `polarization_calibration`

**RgAzComp:**
- `az_sf`, `kaz_poly`

**PFA (Polar Format Algorithm):**
- `fpn`, `ipn`, `polar_ang_ref_time`, `polar_ang_poly`
- `spatial_freq_sf_poly`, `krg1`, `krg2`, `kaz1`, `kaz2`
- `st_deskew` (applied, `st_ds_phase_poly`)

**RMA / INCA:**
- `rm_ref` (pos_ref, vel_ref, dop_cone_ang_ref)
- `inca` (time_ca_poly, r_ca_scp, freq_zero, d_rate_sf_poly, dop_centroid_poly, dop_centroid_coa)

**Waveform parameters:**
- `tx_pulse_length`, `tx_rf_bandwidth`, `tx_freq_start`, `tx_fm_rate`
- `rcv_window_length`, `adc_sample_rate`, `rcv_if_bandwidth`
- `rcv_freq_start`, `rcv_demod_type`, `rcv_fm_rate`

**ImageData pixel types:**
- `pixel_type` (RE32F_IM32F / RE16I_IM16I / AMP8I_PHS8I)
- `amp_table`, `scp_pixel`, `full_image`

### 4.2 SIDD-Only

**Top-level section handles:**
- `backend`, `num_images`, `image_index`, `product_creation`, `display`, `geo_data`, `measurement`, `exploitation_features`, `downstream_reprocessing`, `error_statistics`, `radiometric`, `match_info`, `compression`, `digital_elevation_data`, `product_processing`, `annotations`

**Display:**
- `pixel_type` (MONO8I / MONO16I / RGB24I / etc.)
- `num_bands`, `default_band_display`
- `dynamic_range_adjustment` (algorithm_type, DRA parameters, DRA overrides)

**Measurement:**
- `projection_type` (PlaneProjection / GeographicProjection / CylindricalProjection / PolynomialProjection)
- `plane_projection` (reference_point, sample_spacing, time_coa_poly, product_plane)
- `pixel_footprint`, `arp_flag`

**ExploitationFeatures:**
- Collection `geometry` (azimuth, slope, squint, graze, tilt, doppler_cone_angle)
- Collection `phenomenology` (shadow, layover angle+magnitude, multi_path, ground_track)
- `products[].resolution` (row/col), `ellipticity`, `north`

**DownstreamReprocessing:**
- `geometric_chip` (chip_size, original corners)
- `processing_events` (application_name, applied_date_time, interpolation_method, descriptors)

**Compression:**
- JPEG2000 subtype (num_wavelet_levels, num_bands)

**DigitalElevationData:**
- `geographic_coordinates` (longitude/latitude density, reference_origin)
- `geopositioning` (coordinate_system_type, geodetic_datum, reference_ellipsoid, vertical_datum, sounding_datum, false_origin, utm_grid_zone_number)
- `positional_accuracy` (absolute_horizontal/vertical, point_to_point_horizontal/vertical)
- `null_value`

**ProductProcessing:**
- `processing_modules` (module_name, name, parameters)

**Annotations:**
- `annotations[]` (identifier, spatial_reference_system, objects)

### 4.3 CPHD-Only

**Top-level section handles:**
- `channels`, `pvp`, `global_params`, `collection_info`, `tx_waveform`, `rcv_parameters`, `antenna_pattern`, `scene_coordinates`, `reference_geometry`, `dwell`, `num_channels`

**Per-Vector Parameters (PVP) - all per-pulse arrays:**
- `tx_time`, `tx_pos`, `rcv_time`, `rcv_pos`, `srp_pos`
- `fx1`, `fx2` (start/stop frequency per pulse)
- `tx_vel`, `rcv_vel`
- `sc0`, `scss` (signal start index, sample spacing)
- `signal` (validity indicator)
- `a_fdop`, `a_frr1`, `a_frr2`
- `amp_sf` (amplitude scale factor)
- `toa1`, `toa2` (TOA start/end)

**Channel descriptors:**
- `identifier`, `num_vectors`, `num_samples`, `signal_array_byte_offset`

**Global domain parameters:**
- `domain_type` (FX / TOA)
- `phase_sgn` (+1 or -1)
- `fx_band_min`, `fx_band_max`
- `toa_swath_min`, `toa_swath_max`

**TxWaveform:**
- `lfm_rate`, `pulse_length`

**RcvParameters:**
- `window_length`, `sample_rate`

**AntennaPattern:**
- `freq_zero`, `gain_zero`, `gain_poly` (2D)
- `eb_dcx_poly`, `eb_dcy_poly`
- `acf_x_poly`, `acf_y_poly`, `apc_offset`

**SceneCoordinates:**
- `earth_model`, `iarp_ecf`, `iarp_llh`
- `image_area_x`, `image_area_y`, `corner_points`

**ReferenceGeometry:**
- `ref_time`, `srp_ecf`, `srp_llh`
- `twist_angle_deg`, `slope_angle_deg`, `layover_angle_deg`, `azimuth_angle_deg`

**DwellPolynomial:**
- `cod_time_poly`, `dwell_time_poly`

### 4.4 CRSD-Only

Uses bare `ImageMetadata` - no dedicated metadata fields beyond the base class.

### 4.5 Sentinel-1-Only

**Top-level section handles:**
- `product_info`, `swath_info`, `bursts`, `orbit_state_vectors`, `geolocation_grid`, `doppler_centroids`, `doppler_fm_rates`, `calibration_vectors`, `noise_range_vectors`, `noise_azimuth_vectors`, `num_bursts`, `lines_per_burst`, `samples_per_burst`

**Burst descriptors:**
- `index`, `azimuth_time`, `azimuth_anx_time`, `sensing_time`
- `byte_offset`, `first_valid_sample`, `last_valid_sample`
- `first_line`, `last_line`, `lines_per_burst`, `samples_per_burst`

**TOPS-specific:**
- `swath_info.azimuth_steering_rate`

**Noise vectors:**
- Range: `noise_range_lut` per azimuth time
- Azimuth: `first_azimuth_line`, `last_azimuth_line`, `first_range_sample`, `last_range_sample`, `noise_azimuth_lut`

**Doppler FM rate:**
- `doppler_fm_rates[]` with polynomial coefficients

**Product info:**
- `transmit_receive_polarization` (DV/DH/SV/SH class)
- `processing_facility`, `processing_time`, `ipf_version`

**Swath info:**
- `slant_range_time`, `azimuth_time_interval`, `range_sampling_rate`

**Geolocation grid:**
- `elevation_angle` per tie point

### 4.6 TerraSAR-X / TanDEM-X-Only

**Top-level section handles:**
- `product_info`, `scene_info`, `image_info`, `radar_params`, `orbit_state_vectors`, `geolocation_grid`, `calibration`, `doppler_info`, `processing_info`

**Image info:**
- `sample_type` (COMPLEX / DETECTED)
- `data_format` (COSAR / GEOTIFF)
- `bits_per_sample`
- `projection` (SLANTRANGE / GROUNDRANGE / MAP)

**Radar params:**
- `chirp_duration`, `adc_sampling_rate`

**Scene info:**
- `center_lat`, `center_lon`, `heading_angle`
- Separate `incidence_angle_near`, `incidence_angle_far`, `incidence_angle_center`

**Calibration:**
- `calibration_constant` (scalar Ks)
- `noise_equivalent_beta_nought` (scalar NESZ)
- `calibration_type`

**Processing info:**
- `processing_level`, `range_looks`, `azimuth_looks`

**Product info:**
- `polarization_mode` (SINGLE / DUAL / TWIN / QUAD)
- `generation_time`

### 4.7 BIOMASS-Only

**Top-level section handles:**
- `mission`, `swath`, `product_type`, `start_time`, `stop_time`, `orbit_number`, `orbit_pass`, `polarizations`, `num_polarizations`, `range_pixel_spacing`, `azimuth_pixel_spacing`, `pixel_type`, `pixel_representation`, `projection`, `nodata_value`, `corner_coords`, `prf`, `gcps`

- `pixel_representation` (e.g., `Abs Phase` - magnitude + phase TIFF decomposition)
- `gcps` - Ground Control Points as `(x, y, z, row, col)` tuples
- `nodata_value` (format-specific, separate from base `nodata`)
- `swath` identifier
- `num_polarizations` as explicit count field
- `corner_coords` as dict of named corners `{name: (lat, lon)}`

### 4.8 NISAR-Only

**Top-level section handles:**
- `product_type`, `radar_band`, `frequency`, `polarization`, `available_frequencies`, `available_polarizations`, `identification`, `orbit`, `attitude`, `swath_parameters`, `grid_parameters`, `geolocation_grid`, `calibration`, `processing_info`

**Attitude:**
- `quaternions`, `euler_angles`, `attitude_type`

**Frequency sub-bands:**
- `frequency` (A / B), `available_frequencies`

**Dual product types:**
- `swath_parameters` (RSLC) vs `grid_parameters` (GSLC)

**GSLC grid:**
- `x_coordinates`, `y_coordinates`, `epsg` (map projection)

**Swath parameters:**
- Separate `acquired_*` vs `processed_*` center frequency and bandwidth
- `processed_azimuth_bandwidth`
- `number_of_sub_swaths`
- `zero_doppler_time` vector, `slant_range` vector

**Geolocation grid:**
- 3D with `height_above_ellipsoid` axis
- `elevation_angle`

**Identification:**
- `frame_number`, `track_number`, `granule_id`
- `bounding_polygon` (WKT)
- `is_geocoded`, `instrument_name`, `processing_type`, `product_version`

**Orbit:**
- `reference_epoch`, `orbit_type`, `interp_method`

**Processing:**
- `algorithms` dict

### 4.9 GeoTIFF (SAR Fallback)

Uses bare `ImageMetadata` - no SAR-specific fields. Geospatial values may be populated via first-class base fields (`transform`, `bounds`, `pixel_resolution`) and/or `extras`.
