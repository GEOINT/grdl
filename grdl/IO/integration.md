# Non-SAR Remote Sensing Platform Integration Guide

Reference of open and commercially accessible remote sensing platforms, organized by modality. Covers platforms, download portals, file formats, and GRDL reader implementation priorities.

## Electro-Optical Multispectral

### Tier 1: Fully Open / Free

| Platform | Agency | Resolution | Bands | Revisit | Format | Status |
|----------|--------|-----------|-------|---------|--------|--------|
| Landsat 8/9 | USGS/NASA | 15 m PAN, 30 m MS, 100 m TIR | 11 (0.43-12.51 um) | 8 days combined | COG/GeoTIFF | Active |
| Sentinel-2A/B/C | ESA/Copernicus | 10/20/60 m | 13 (443-2190 nm) | 5 days | JP2 (SAFE), COG on AWS | Active |
| MODIS (Terra/Aqua) | NASA | 250/500/1000 m | 36 (0.4-14.4 um) | 1-2 days | HDF4 (HDF-EOS) | Active (EOL ~2026-27) |
| VIIRS (NPP, NOAA-20/21) | NASA/NOAA | 375/750 m | 22 (0.41-12.5 um) | ~12 hrs | HDF5, NetCDF | Active |
| ASTER (Terra) | NASA/METI | 15/30/90 m | 14 (9 functional) | 16 days | HDF-EOS (HDF4) | Active (limited) |
| HLS | NASA GSFC | 30 m | 11-13 | 2-3 days | COG | Active |
| NAIP | USDA | 30-60 cm | 3-4 (RGB/RGBN) | 2-3 yr cycle | GeoTIFF, JP2 | Active (aerial) |
| Maxar Open Data | Maxar/Vantor | 30-50 cm | 3-4 (RGB/RGBN) | Event-based | COG | Active (CC 4.0) |
| CBERS-4/4A | INPE/CAST | 2-55 m | 4+PAN | 5-26 days | GeoTIFF | Active |
| VENuS | CNES/ISA | 5.3 m | 12 (420-910 nm) | 2 days (selected sites) | COG | Decommissioned (archive) |

**Landsat 8/9** -- Workhorse of civilian EO. Collection 2 delivered as COG. Download: [USGS EarthExplorer](https://earthexplorer.usgs.gov), [Landsat on AWS](https://registry.opendata.aws/landsat-8/).

**Sentinel-2** -- Highest-resolution free global multispectral. Native format is SAFE directory with JP2 imagery; COG available on AWS via Element 84. Sentinel-2C launched Sep 2024, replaced 2A operationally Jan 2025. Download: [Copernicus Data Space](https://dataspace.copernicus.eu), [Sentinel-2 on AWS](https://registry.opendata.aws/sentinel-2/).

**MODIS** -- 20+ year archive, near-daily global. Aqua science data expected to end ~Aug 2026; Terra ~Jan 2027. Transitioning to VIIRS. Download: [LAADS DAAC](https://ladsweb.modaps.eosdis.nasa.gov), [NASA Earthdata](https://earthdata.nasa.gov).

**VIIRS** -- MODIS successor. Three active satellites (NPP, NOAA-20, NOAA-21). Download: [LAADS DAAC](https://ladsweb.modaps.eosdis.nasa.gov), [NOAA CLASS](https://www.avl.class.noaa.gov).

**ASTER** -- SWIR detector failed Apr 2008; VNIR stopped Apr 2023; TIR still active. On-demand L2/L3 processing decommissioned Dec 2025. Archive remains. Download: [USGS EarthExplorer](https://earthexplorer.usgs.gov), [LP DAAC](https://lpdaac.usgs.gov).

**HLS** -- Harmonized Landsat-Sentinel product. Radiometrically cross-calibrated, gridded on Sentinel-2 MGRS tiles. Download: [NASA Earthdata](https://search.earthdata.nasa.gov), [Planetary Computer](https://planetarycomputer.microsoft.com/dataset/group/hls).

**NAIP** -- US-only aerial imagery, 30-60 cm. Not a satellite. Download: [USGS EarthExplorer](https://earthexplorer.usgs.gov), [National Map](https://apps.nationalmap.gov/downloader/), [NAIP Hub](https://naip-usdaonline.hub.arcgis.com/).

**Maxar Open Data** -- VHR imagery released under CC 4.0 for disaster response events. STAC catalog on AWS. Download: [Maxar Open Data on AWS](https://registry.opendata.aws/maxar-open-data/).

**CBERS-4/4A** -- Brazil-China program, fully open. Download: [INPE Catalog](http://www.dgi.inpe.br/catalogo/explore), [CBERS on AWS](https://registry.opendata.aws/cbers/).

**VENuS** -- 12 narrow VNIR bands at 5.3 m, decommissioned Aug 2024. Archive on AWS. Download: [VENuS on AWS](https://registry.opendata.aws/venus-l2a-cogs/).

### Tier 2: Commercial with Free Access Pathways

| Platform | Agency | Resolution | Bands | Revisit | Format | Free Access |
|----------|--------|-----------|-------|---------|--------|-------------|
| PlanetScope | Planet Labs | 3 m | 4-8 | Daily | GeoTIFF | NICFI, NASA CSDA, ESA Earthnet |
| SkySat | Planet Labs | 50 cm | 4+PAN | 12x/day | GeoTIFF | ESA Earthnet, NASA CSDA |
| WorldView-2/3 | Maxar/Vantor | 31-46 cm PAN, 1.2-1.8 m MS | 8-29 | <1 day | NITF, GeoTIFF | NASA CSDA, ESA TPM |
| WorldView Legion | Maxar/Vantor | 34 cm PAN, 1.36 m MS | 8+PAN | 15x/day | NITF, GeoTIFF | NASA CSDA, ESA TPM |
| Pleiades 1A/1B | CNES/Airbus | 50 cm PAN, 2 m MS | 4+PAN | Daily | DIMAP (GeoTIFF+XML), JP2 | ESA TPM, Dinamis |
| Pleiades Neo | Airbus | 30 cm PAN, 1.2 m MS | 6+PAN | 2x/day | DIMAP (GeoTIFF+XML) | ESA TPM |
| SPOT 6 | Airbus | 1.5 m PAN, 6 m MS | 4+PAN | Daily | DIMAP (GeoTIFF+XML) | ESA TPM, Dinamis |
| Gaofen (16m WFV) | CNSA/CRESDA | 16 m (WFV open) | 4-8 | 2 days | TIFF+XML | 16m WFV open via CNSA-GEO |
| SuperView-1 | Siwei (China) | 50 cm PAN, 2 m MS | 4+PAN | ~1 day | GeoTIFF | Preview only |
| Jilin-1 | Chang Guang | 30 cm-1.2 m | 4+PAN | Multi-daily | GeoTIFF | None (commercial) |

**Free access programs:**
- [NASA CSDA](https://www.earthdata.nasa.gov/about/csda) -- US government researchers; covers Planet and Maxar
- [ESA Third Party Missions](https://earth.esa.int/eogateway/missions) -- European researchers, proposal-based
- [Planet NICFI](https://www.planet.com/nicfi/) -- Tropical forest monitoring, open access
- [Dinamis](https://dinamis.data-terra.org/) -- French institutional users, SPOT/Pleiades

### Planned

| Platform | Agency | Resolution | Bands | Launch | Notes |
|----------|--------|-----------|-------|--------|-------|
| Landsat Next | NASA/USGS | ~10-15 m | 26 | 2030-31 | 3-satellite constellation, 6-day revisit |

---

## Hyperspectral

| Platform | Agency | Spectral Range | Bands | Resolution | Format | Download | Status |
|----------|--------|---------------|-------|-----------|--------|----------|--------|
| PRISMA | ASI (Italy) | 400-2500 nm | 239+PAN | 30 m (5 m PAN) | HDF5 (.he5) | [prisma.asi.it](https://prisma.asi.it) | Active (EOL ~2026) |
| EnMAP | DLR/GFZ | 420-2450 nm | 228 | 30 m | GeoTIFF, COG, JP2, ENVI | [EOWEB](https://eoweb.dlr.de), [EOC Geoservice](https://geoservice.dlr.de/web/datasets/enmap) | Active (EOL Sep 2026) |
| EMIT | NASA/JPL (ISS) | 381-2493 nm | 285 | 60 m | NetCDF4 | [NASA Earthdata](https://search.earthdata.nasa.gov) | Active |
| DESIS | DLR/Teledyne (ISS) | 400-1000 nm | 235 | 30 m | COG | [Teledyne TCloud](https://teledyne.tcloudhost.com), [DLR Geoservice](https://geoservice.dlr.de/data-assets/hxom21uqeo90.html) | End-of-life ~2025 (archive) |
| HISUI | JAXA/METI (ISS) | 400-2500 nm | 185 | 20x31 m | HDF5 | [Tellus](https://www.tellusxdp.com/en-us/catalog/data/hisui.html) | Active |
| Hyperion (EO-1) | NASA/USGS | 357-2576 nm | 220 | 30 m | GeoTIFF | [EarthExplorer](https://earthexplorer.usgs.gov) | Archived (2000-2017) |
| PACE/OCI | NASA | 340-2260 nm | 289 | 1.2 km | NetCDF4 | [Ocean Color Web](https://oceancolor.gsfc.nasa.gov) | Active |

**Planned:**

| Platform | Agency | Bands | Resolution | Launch |
|----------|--------|-------|-----------|--------|
| CHIME (Sentinel-10) | ESA/Copernicus | >200 (400-2500 nm) | 30 m | 2028/2030 |
| SBG | NASA/JPL | VSWIR + TIR | ~30/60 m | 2028+ |

---

## Thermal / Infrared

| Platform | Agency | Thermal Range | Bands | Resolution | Format | Download | Status |
|----------|--------|--------------|-------|-----------|--------|----------|--------|
| ECOSTRESS (ISS) | NASA/JPL | 8-12.5 um | 5 TIR + 1 SWIR | ~70 m | HDF5, COG | [NASA Earthdata](https://search.earthdata.nasa.gov) | Active |
| Landsat 8/9 TIRS | NASA/USGS | 10.6-12.5 um | 2 | 100 m (resampled 30 m) | GeoTIFF (COG) | [EarthExplorer](https://earthexplorer.usgs.gov) | Active |
| ASTER TIR | NASA/METI | 8.1-11.7 um | 5 | 90 m | HDF-EOS | [EarthExplorer](https://earthexplorer.usgs.gov) | Active |
| MODIS thermal | NASA | 3.7-14.4 um | 16 emissive | 1000 m | HDF-EOS | [LAADS DAAC](https://ladsweb.modaps.eosdis.nasa.gov) | Active (aging) |
| VIIRS thermal | NASA/NOAA | 3.7-12.5 um | I4, I5, M12-M16 | 375/750 m | HDF5, NetCDF | [LAADS DAAC](https://ladsweb.modaps.eosdis.nasa.gov) | Active |

---

## LiDAR / Altimetry

| Platform | Agency | Instrument | Footprint | Format | Download | Status |
|----------|--------|-----------|-----------|--------|----------|--------|
| ICESat-2 | NASA | ATLAS (532 nm photon-counting) | ~14 m, 6 beams | HDF5 | [NSIDC](https://nsidc.org/data/icesat-2/data), [OpenAltimetry](https://openaltimetry.org) | Active |
| GEDI (ISS) | NASA/UMD | Full-waveform 1064 nm | ~25 m, 8 tracks | HDF5 | [LP DAAC](https://lpdaac.usgs.gov), [ORNL DAAC](https://daac.ornl.gov) | Active (EOL Jan 2031) |
| CALIPSO | NASA/CNES | CALIOP (532+1064 nm) | 333 m along-track | HDF4 | [NASA ASDC](https://asdc.larc.nasa.gov/project/CALIPSO) | Decommissioned (2006-2023 archive) |
| USGS 3DEP | USGS | Airborne LiDAR | 2-8+ pts/m2 | LAZ, GeoTIFF (DEMs) | [LiDAR Explorer](https://apps.nationalmap.gov/lidar-explorer/), [3DEP on AWS](https://registry.opendata.aws/usgs-lidar/) | Ongoing |

---

## Geostationary Weather / Environmental

| Platform | Agency | Bands | Resolution | Cadence | Format | Download | Status |
|----------|--------|-------|-----------|---------|--------|----------|--------|
| GOES-16/18/19 | NOAA/NASA | 16 (0.47-13.3 um) | 0.5-2 km | 5-10 min | NetCDF4 | [NOAA CLASS](https://www.avl.class.noaa.gov), [GOES on AWS](https://registry.opendata.aws/noaa-goes/) | Active |
| Himawari-8/9 | JMA (Japan) | 16 (0.47-13.3 um) | 0.5-2 km | 2.5-10 min | HSD (Himawari Standard) | [Himawari on AWS](https://registry.opendata.aws/noaa-himawari/), [JAXA P-Tree](https://www.eorc.jaxa.jp/ptree/) | Active |
| Meteosat MSG/MTG | EUMETSAT | 12 (SEVIRI) | 1-3 km | 5-15 min | HRIT/LRIT, NetCDF | [EUMETSAT Data Store](https://data.eumetsat.int) | Active |
| INSAT-3D/3DR | ISRO (India) | 6 (VIS-TIR) | 1-8 km | 15 min | HDF5 | [MOSDAC](https://mosdac.gov.in) | Active |

---

## Ocean Color

| Platform | Agency | Spectral Range | Bands | Resolution | Format | Download | Status |
|----------|--------|---------------|-------|-----------|--------|----------|--------|
| PACE/OCI | NASA | 340-2260 nm | 289 | 1.2 km | NetCDF4 | [Ocean Color Web](https://oceancolor.gsfc.nasa.gov) | Active |
| Sentinel-3 OLCI | ESA/EUMETSAT | 400-1020 nm | 21 | 300 m | NetCDF4 (SAFE) | [Copernicus Data Space](https://dataspace.copernicus.eu) | Active |
| MODIS ocean color | NASA | 412-869 nm | 9 | 1000 m | HDF-EOS, NetCDF | [Ocean Color Web](https://oceancolor.gsfc.nasa.gov) | Active |

---

## Data Distribution Portals

| Portal | URL | Key Missions | API | Registration |
|--------|-----|-------------|-----|--------------|
| USGS EarthExplorer | https://earthexplorer.usgs.gov | Landsat, ASTER, NAIP | M2M REST, STAC | Free EROS account |
| NASA Earthdata | https://earthdata.nasa.gov | All NASA missions | CMR, CMR-STAC, AppEEARS | Free EDL login |
| Copernicus Data Space | https://dataspace.copernicus.eu | Sentinel-1/2/3/5P | STAC, OData, openEO | Free |
| Google Earth Engine | https://earthengine.google.com | 900+ datasets | Proprietary (`ee` Python) | Free (research) |
| Planetary Computer | https://planetarycomputer.microsoft.com | Landsat, S-2, HLS, NAIP, MODIS | STAC (`pystac-client`) | None for search |
| AWS Open Data | https://registry.opendata.aws | Landsat, S-2, NAIP, GOES, Maxar | STAC (Element 84) | None |
| Planet Explorer | https://planet.com/explorer | PlanetScope, SkySat | REST, Orders API | Commercial |
| Maxar Open Data | https://maxar.com/open-data | WorldView, GeoEye | STAC on AWS S3 | None |
| NOAA CLASS | https://www.avl.class.noaa.gov | GOES, JPSS/VIIRS | Web-based ordering | Free |
| JAXA G-Portal | https://gportal.jaxa.jp | ALOS, GCOM, GPM | `jaxa-earth-api` | Free |
| ASI PRISMA | https://prisma.asi.it | PRISMA hyperspectral | Web portal only | Free |
| DLR EOWEB | https://eoweb.dlr.de | EnMAP hyperspectral | EOWEB, EOC Geoservice | Free |

**Key ecosystem libraries:**
- `earthaccess` -- NASA Earthdata auth, CMR search, download/streaming
- `pystac-client` -- STAC API client (Planetary Computer, AWS, Copernicus)
- `planetary-computer` -- Azure SAS token signing for Planetary Computer
- `sentinelsat` -- Copernicus Sentinel search/download
- `planet` -- Planet SDK for Python (PlanetScope, SkySat)

---

## File Formats and GRDL Reader Status

| Format | Extensions | Used By | Python Backend | GRDL Status |
|--------|-----------|---------|---------------|-------------|
| GeoTIFF / COG | `.tif`, `.tiff` | Landsat, Planet, Maxar, HLS, NAIP, EnMAP | `rasterio` | **Implemented** |
| NITF | `.nitf`, `.ntf` | WorldView, SkySat, NGA products | `rasterio`/GDAL | **Implemented** |
| HDF5 / HDF-EOS5 | `.h5`, `.he5`, `.hdf5` | NASA (MODIS, VIIRS, ICESat-2, GEDI, EMIT), PRISMA, JAXA | `h5py`, `xarray` | **Implemented** |
| JPEG2000 | `.jp2`, `.j2k` | Sentinel-2, EnMAP, Pleiades | `rasterio`/GDAL, `glymur` | Not implemented |
| HDF4 / HDF-EOS | `.hdf`, `.he4` | MODIS (legacy), ASTER, CALIPSO | `pyhdf`, GDAL | Not implemented |
| NetCDF-4 | `.nc`, `.nc4` | NOAA (GOES, JPSS), Sentinel-3/5P, PACE, ocean color | `xarray`, `netCDF4` | Not implemented |
| SAFE | `.SAFE` (dir) | All Sentinel missions | XML + format readers | Not implemented |
| ENVI | `.hdr` + `.bil`/`.bip`/`.bsq` | EnMAP, AVIRIS, airborne hyperspectral | `spectral`, numpy mmap | Not implemented |
| LAS/LAZ | `.las`, `.laz` | USGS 3DEP, LiDAR datasets | `laspy`, PDAL | Not implemented (point cloud) |
| Zarr | `.zarr` (dir) | Planetary Computer, ERA5, cloud-native | `zarr`, `xarray` | Not implemented |
| DIMAP | `.dim` + GeoTIFF/JP2 | Pleiades, SPOT (Airbus) | XML + `rasterio` | Not implemented |

---

## Reader Implementation Priorities

### Priority 1 -- HDF5Reader

Unlocks the largest number of new platforms: NASA Earthdata (MODIS, VIIRS, ASTER, ICESat-2, GEDI, EMIT), PRISMA hyperspectral, JAXA missions.

- **Backend:** `h5py`
- **Key challenge:** HDF5 is hierarchical -- datasets live at arbitrary paths. Need convention for dataset selection (by path, by index, or auto-detect).
- **Scope:** Single-dataset raster reads. Not a general HDF5 browser.

### Priority 2 -- NetCDF4Reader

Shares HDF5 infrastructure (NetCDF-4 is built on HDF5). Unlocks NOAA (GOES, JPSS/VIIRS), Sentinel-3 OLCI/SLSTR, Sentinel-5P, PACE ocean color, climate data.

- **Backend:** `xarray` with `h5netcdf` or `netCDF4` engine
- **Key challenge:** CF conventions, coordinate variables, multi-dimensional data
- **Scope:** Variable-based reads with coordinate subsetting

### Priority 3 -- JP2Reader

Unlocks Sentinel-2 native format (SAFE/JP2). Note: `rasterio` may already handle JP2 if GDAL was compiled with OpenJPEG support.

- **Backend:** `rasterio` (GDAL JP2 driver) or `glymur` (for S-2 non-standard 15-bit)
- **Key challenge:** Sentinel-2 encodes 15-bit data in JP2; some GDAL builds lack JP2 support
- **Scope:** Single-band JP2 reads, consistent with GeoTIFFReader API

### Priority 4 -- ENVIReader

Unlocks airborne/spaceborne hyperspectral workflows (EnMAP BSQ/BIL/BIP, AVIRIS, HySpex).

- **Backend:** `spectral` (SPy) or raw numpy memmap
- **Key challenge:** Three interleave modes (BSQ, BIL, BIP), ASCII header parsing
- **Scope:** Band-sequential and band-interleaved reads with wavelength metadata

### Priority 5 -- SAFEReader

Makes Sentinel products "just work" as directories. Parses `manifest.safe` XML, delegates to JP2/GeoTIFF/NetCDF readers for actual imagery.

- **Backend:** `xml.etree` + existing GRDL readers
- **Key challenge:** Complex directory structure varies by Sentinel mission
- **Scope:** Sentinel-2 SAFE initially, extensible to S-1/S-3

### Lower Priority

- **DIMAP** -- Airbus Pleiades/SPOT. XML manifest + GeoTIFF/JP2. Commercial data access.
- **Zarr** -- Better served by `xarray` integration than a dedicated reader.
- **LAS/LAZ** -- Point cloud data, fundamentally different from 2D raster model.
- **HDF4** -- Legacy. MODIS HDF4 being superseded by HDF5/NetCDF products.

---

## Format-to-Platform Coverage Matrix

Shows which platforms each reader implementation would unlock:

| Reader | Platforms Unlocked |
|--------|--------------------|
| GeoTIFF (**done**) | Landsat, HLS, NAIP, Maxar Open Data, Planet, CBERS, Gaofen, VENuS, EnMAP (optional), 3DEP DEMs |
| NITF (**done**) | WorldView, SkySat, NGA products |
| HDF5 (**done**) | MODIS, VIIRS, ASTER, ICESat-2, GEDI, EMIT, PRISMA, ECOSTRESS, HISUI, INSAT |
| NetCDF4 (Priority 2) | GOES, JPSS/VIIRS, Sentinel-3 OLCI/SLSTR, Sentinel-5P, PACE/OCI, Himawari, Meteosat |
| JP2 (Priority 3) | Sentinel-2 (native), Pleiades, SPOT, EnMAP (optional) |
| ENVI (Priority 4) | EnMAP, AVIRIS, HySpex, airborne hyperspectral |
| SAFE (Priority 5) | Sentinel-1/2/3 (directory-level access) |
