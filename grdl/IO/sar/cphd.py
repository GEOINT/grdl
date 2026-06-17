# -*- coding: utf-8 -*-
"""
CPHD Reader - Compensated Phase History Data format.

NGA standard for phase history data (unfocused SAR). Uses sarkit as the
primary backend with sarpy as fallback.

The sarkit path parses the full NGA CPHD 1.1.0 XML tree into the typed
:class:`grdl.IO.models.cphd.CPHDMetadata` hierarchy: every required and
every optional element is populated when present.

Dependencies
------------
sarkit (primary) or sarpy (fallback)

Author
------
Duane Smalley, PhD
170194430+DDSmalls@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-02-09

Modified
--------
2026-06-07
"""

# Standard library
import logging
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple, Union

# Third-party
import numpy as np

# GRDL internal
from grdl.IO.base import ImageReader
from grdl.IO.models.cphd import (
    CPHDAddedSupportArray,
    CPHDAntCoordFrame,
    CPHDAntenna,
    CPHDAntennaPattern,
    CPHDAntEB,
    CPHDAntGainPhaseArray,
    CPHDAntGainPhaseFreqEntry,
    CPHDAntGainPhasePoly,
    CPHDAntPatternFull,
    CPHDAntPhaseCenter,
    CPHDAntPolRef,
    CPHDBistaticError,
    CPHDBistaticGeometry,
    CPHDBistaticPlatform,
    CPHDBistaticPlatformError,
    CPHDBistaticRadarSensorError,
    CPHDChannel,
    CPHDChannelAntennaRefs,
    CPHDChannelParameters,
    CPHDChannelSection,
    CPHDChannelTxRcvRefs,
    CPHDCODTime,
    CPHDCollectionInfo,
    CPHDCreationInfo,
    CPHDData,
    CPHDDwell,
    CPHDDwellTime,
    CPHDDwellTimeArray,
    CPHDDwellTimesRef,
    CPHDEBFreqShiftSF,
    CPHDErrorDecorrFunc,
    CPHDErrorParameters,
    CPHDFxNoiseProfilePoint,
    CPHDGeoInfo,
    CPHDGeoLine,
    CPHDGeoPolygon,
    CPHDGlobal,
    CPHDHAESurface,
    CPHDIASegment,
    CPHDIAZArray,
    CPHDImageArea,
    CPHDImageGrid,
    CPHDIonoError,
    CPHDIonoParameters,
    CPHDLFMEclipse,
    CPHDMatchCollection,
    CPHDMatchInfo,
    CPHDMatchType,
    CPHDMetadata,
    CPHDMLFreqDilationSF,
    CPHDMonoRadarSensorError,
    CPHDMonostaticError,
    CPHDMonostaticGeometry,
    CPHDNoiseLevel,
    CPHDParameter,
    CPHDPlanarSurface,
    CPHDPolarization,
    CPHDPolRef,
    CPHDPosVelCorrCoefs,
    CPHDPosVelErr,
    CPHDProductInfo,
    CPHDPVP,
    CPHDReferenceGeometry,
    CPHDReferenceSurface,
    CPHDRcvParameters,
    CPHDSceneCoordinates,
    CPHDSupportArraySize,
    CPHDSupportArrays,
    CPHDTimeline,
    CPHDTOAExtended,
    CPHDTropoError,
    CPHDTropoParameters,
    CPHDTxRcv,
    CPHDTxWaveform,
)
from grdl.IO.sar._backend import require_sar_backend

logger = logging.getLogger(__name__)


# ===================================================================
# XML helpers
# ===================================================================

def _to_float(text: Optional[str]) -> Optional[float]:
    if text is None or text == '':
        return None
    return float(text)


def _to_int(text: Optional[str]) -> Optional[int]:
    if text is None or text == '':
        return None
    return int(text)


def _to_bool(text: Optional[str]) -> Optional[bool]:
    if text is None or text == '':
        return None
    return text.strip().lower() in ('true', '1')


def _findtext(elem: Any, path: str) -> Optional[str]:
    """``elem.findtext(path)`` that tolerates ``elem is None``."""
    if elem is None:
        return None
    return elem.findtext(path)


def _parse_xyz(elem: Any) -> Optional[np.ndarray]:
    """Parse an ``XYZType`` (X, Y, Z children) into ``np.ndarray (3,)``."""
    if elem is None:
        return None
    x = _to_float(elem.findtext('{*}X'))
    y = _to_float(elem.findtext('{*}Y'))
    z = _to_float(elem.findtext('{*}Z'))
    if x is None and y is None and z is None:
        return None
    return np.array([x or 0.0, y or 0.0, z or 0.0])


def _parse_xy(elem: Any) -> Optional[Tuple[float, float]]:
    """Parse an ``XYType`` (X, Y children) into ``(x, y)``."""
    if elem is None:
        return None
    x = _to_float(elem.findtext('{*}X'))
    y = _to_float(elem.findtext('{*}Y'))
    if x is None and y is None:
        return None
    return (x or 0.0, y or 0.0)


def _parse_lat_lon(elem: Any) -> Optional[Tuple[float, float]]:
    if elem is None:
        return None
    lat = _to_float(elem.findtext('{*}Lat'))
    lon = _to_float(elem.findtext('{*}Lon'))
    if lat is None and lon is None:
        return None
    return (lat or 0.0, lon or 0.0)


def _parse_lat_lon_hae(elem: Any) -> Optional[np.ndarray]:
    if elem is None:
        return None
    lat = _to_float(elem.findtext('{*}Lat'))
    lon = _to_float(elem.findtext('{*}Lon'))
    hae = _to_float(elem.findtext('{*}HAE'))
    if lat is None and lon is None and hae is None:
        return None
    return np.array([lat or 0.0, lon or 0.0, hae or 0.0])


def _parse_poly1d(elem: Any) -> Optional[np.ndarray]:
    """Parse a Poly1D element into a 1D coefficient array."""
    if elem is None:
        return None
    coefs = elem.findall('{*}Coef')
    if not coefs:
        text = elem.text
        if text is not None and text.strip():
            try:
                return np.array([float(text)])
            except ValueError:
                return None
        return None
    max_exp = max(
        int(c.get('exponent1', c.get('exponent', i)))
        for i, c in enumerate(coefs)
    )
    result = np.zeros(max_exp + 1)
    for i, c in enumerate(coefs):
        exp = int(c.get('exponent1', c.get('exponent', i)))
        result[exp] = float(c.text)
    return result


def _parse_poly2d(elem: Any) -> Optional[np.ndarray]:
    """Parse a Poly2D element into a 2D coefficient array."""
    if elem is None:
        return None
    coefs = elem.findall('{*}Coef')
    if not coefs:
        return None
    max_e1 = max(int(c.get('exponent1', '0')) for c in coefs)
    max_e2 = max(int(c.get('exponent2', '0')) for c in coefs)
    result = np.zeros((max_e1 + 1, max_e2 + 1))
    for c in coefs:
        e1 = int(c.get('exponent1', '0'))
        e2 = int(c.get('exponent2', '0'))
        result[e1, e2] = float(c.text)
    return result


def _parse_xyz_poly(elem: Any) -> Optional[np.ndarray]:
    """Parse an XYZPoly into shape ``(K, 3)`` where K is the order + 1."""
    if elem is None:
        return None
    x_poly = _parse_poly1d(elem.find('{*}X'))
    y_poly = _parse_poly1d(elem.find('{*}Y'))
    z_poly = _parse_poly1d(elem.find('{*}Z'))
    if x_poly is None and y_poly is None and z_poly is None:
        return None
    max_order = max(
        len(x_poly) if x_poly is not None else 0,
        len(y_poly) if y_poly is not None else 0,
        len(z_poly) if z_poly is not None else 0,
    )
    result = np.zeros((max_order, 3))
    if x_poly is not None:
        result[:len(x_poly), 0] = x_poly
    if y_poly is not None:
        result[:len(y_poly), 1] = y_poly
    if z_poly is not None:
        result[:len(z_poly), 2] = z_poly
    return result


def _parse_parameters(elems: Iterable[Any]) -> List[CPHDParameter]:
    """Parse a sequence of ``Parameter`` elements (name attr + body)."""
    out: List[CPHDParameter] = []
    for p in elems:
        out.append(CPHDParameter(
            name=p.get('name', '') or '',
            value=p.text or '',
        ))
    return out


def _parse_image_area(elem: Any) -> Optional[CPHDImageArea]:
    """Parse an ``AreaType`` element."""
    if elem is None:
        return None
    x1y1 = _parse_xy(elem.find('{*}X1Y1'))
    x2y2 = _parse_xy(elem.find('{*}X2Y2'))
    polygon = None
    poly_elem = elem.find('{*}Polygon')
    if poly_elem is not None:
        verts = []
        for v in poly_elem.findall('{*}Vertex'):
            xy = _parse_xy(v)
            if xy is not None:
                verts.append(xy)
        if verts:
            polygon = np.array(verts)
    if x1y1 is None and x2y2 is None and polygon is None:
        return None
    return CPHDImageArea(x1y1=x1y1, x2y2=x2y2, polygon=polygon)


def _parse_decorr_func(elem: Any) -> Optional[CPHDErrorDecorrFunc]:
    if elem is None:
        return None
    return CPHDErrorDecorrFunc(
        corr_coef_zero=_to_float(elem.findtext('{*}CorrCoefZero')),
        decorr_rate=_to_float(elem.findtext('{*}DecorrRate')),
    )


# ===================================================================
# CPHDReader
# ===================================================================

class CPHDReader(ImageReader):
    """Read CPHD (Compensated Phase History Data) format.

    CPHD is the NGA standard for phase history data. This reader uses
    sarkit as the primary backend with sarpy as fallback. The full
    CPHD 1.1.0 XML metadata tree is parsed into the typed
    :class:`CPHDMetadata` hierarchy when sarkit is available.

    Parameters
    ----------
    filepath : str or Path
        Path to the CPHD file.

    Attributes
    ----------
    filepath : Path
        Path to the CPHD file.
    metadata : CPHDMetadata
        Fully populated typed metadata.
    backend : str
        Active backend (``'sarkit'`` or ``'sarpy'``).

    Raises
    ------
    ImportError
        If neither sarkit nor sarpy is installed.
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is not valid CPHD.

    Notes
    -----
    CPHD contains phase history data, not formed imagery. Use focusing
    algorithms to generate images from CPHD.

    Examples
    --------
    >>> from grdl.IO.sar import CPHDReader
    >>> with CPHDReader('data.cphd') as reader:
    ...     shape = reader.get_shape()
    ...     data = reader.read_chip(0, 100, 0, 200)
    """

    _enforce_2d: bool = True  # CPHD is always single-channel phase history

    def __init__(self, filepath: Union[str, Path]) -> None:
        self.backend = require_sar_backend('CPHD')
        logger.info("CPHD backend selected: %s", self.backend)
        super().__init__(filepath)

    # ------------------------------------------------------------------
    # Metadata loading dispatcher
    # ------------------------------------------------------------------

    def _load_metadata(self) -> None:
        """Load CPHD metadata using the active backend."""
        if self.backend == 'sarkit':
            self._load_metadata_sarkit()
        else:
            self._load_metadata_sarpy()

    # ==================================================================
    # SARKIT BACKEND
    # ==================================================================

    def _load_metadata_sarkit(self) -> None:
        """Load metadata via sarkit, parsing the full CPHD 1.1.0 XML tree."""
        import sarkit.cphd

        try:
            self._file_handle = open(str(self.filepath), 'rb')
            self._reader = sarkit.cphd.Reader(self._file_handle)
            xml = self._reader.metadata.xmltree
            self._xmltree = xml

            # Some CPHD writers emit inconsistent per-channel PVP byte
            # offsets; repair them in-place before any PVP read so the
            # signal reader lands on the correct bytes for channels > 0.
            self._repair_pvp_offsets_sarkit(xml)

            collection_info = self._parse_collection_info_sarkit(xml)
            global_params = self._parse_global_sarkit(xml)
            scene_coordinates = self._parse_scene_coordinates_sarkit(xml)
            data = self._parse_data_sarkit(xml)
            channel_section = self._parse_channel_section_sarkit(xml)
            support_arrays = self._parse_support_arrays_sarkit(xml)
            dwell = self._parse_dwell_sarkit(xml)
            reference_geometry = self._parse_reference_geometry_sarkit(xml)
            antenna = self._parse_antenna_sarkit(xml)
            tx_rcv = self._parse_tx_rcv_sarkit(xml)
            error_parameters = self._parse_error_parameters_sarkit(xml)
            product_info = self._parse_product_info_sarkit(xml)
            geo_info = [
                self._parse_geo_info_sarkit(g)
                for g in xml.findall('{*}GeoInfo')
            ]
            match_info = self._parse_match_info_sarkit(xml)

            channels = list(data.channels) if data is not None else []
            self._default_channel = channels[0].identifier if channels else ''

            pvp = self._load_pvp_sarkit(self._default_channel)

            tx_waveform = (
                tx_rcv.tx_waveforms[0]
                if tx_rcv and tx_rcv.tx_waveforms else None
            )
            rcv_parameters = (
                tx_rcv.rcv_parameters[0]
                if tx_rcv and tx_rcv.rcv_parameters else None
            )
            antenna_pattern = self._project_antenna_pattern(antenna)

            first_ch = channels[0] if channels else CPHDChannel()
            self.metadata = CPHDMetadata(
                format='CPHD',
                rows=first_ch.num_vectors,
                cols=first_ch.num_samples,
                dtype='complex64',
                collection_info=collection_info,
                global_params=global_params,
                scene_coordinates=scene_coordinates,
                data=data,
                channel_section=channel_section,
                pvp=pvp,
                dwell=dwell,
                reference_geometry=reference_geometry,
                support_arrays=support_arrays,
                antenna=antenna,
                tx_rcv=tx_rcv,
                error_parameters=error_parameters,
                product_info=product_info,
                geo_info=geo_info,
                match_info=match_info,
                channels=channels,
                num_channels=len(channels),
                tx_waveform=tx_waveform,
                rcv_parameters=rcv_parameters,
                antenna_pattern=antenna_pattern,
                extras={'backend': 'sarkit'},
            )

            logger.info(
                "Loaded CPHD %s (%d x %d) via sarkit",
                self.filepath.name,
                first_ch.num_vectors,
                first_ch.num_samples,
            )
            logger.debug(
                "CPHD channels=%d, has_waveform=%s, has_antenna=%s, "
                "has_error=%s",
                len(channels),
                tx_waveform is not None,
                antenna is not None,
                error_parameters is not None,
            )

        except Exception as e:
            raise ValueError(f"Failed to load CPHD metadata: {e}") from e

    # ------------------------------------------------------------------
    # 1. CollectionID
    # ------------------------------------------------------------------

    def _parse_collection_info_sarkit(
        self, xml: Any,
    ) -> Optional[CPHDCollectionInfo]:
        ci = xml.find('{*}CollectionID')
        if ci is None:
            return None
        radar_mode = ci.find('{*}RadarMode')
        return CPHDCollectionInfo(
            collector_name=ci.findtext('{*}CollectorName'),
            illuminator_name=ci.findtext('{*}IlluminatorName'),
            core_name=ci.findtext('{*}CoreName'),
            collect_type=ci.findtext('{*}CollectType'),
            radar_mode=_findtext(radar_mode, '{*}ModeType'),
            radar_mode_id=_findtext(radar_mode, '{*}ModeID'),
            classification=ci.findtext('{*}Classification'),
            release_info=ci.findtext('{*}ReleaseInfo'),
            country_code=ci.findtext('{*}CountryCode'),
            parameters=_parse_parameters(ci.findall('{*}Parameter')),
        )

    # ------------------------------------------------------------------
    # 2. Global
    # ------------------------------------------------------------------

    def _parse_global_sarkit(self, xml: Any) -> Optional[CPHDGlobal]:
        g = xml.find('{*}Global')
        if g is None:
            return None

        sgn_text = g.findtext('{*}SGN')
        if sgn_text is None:
            sgn_text = g.findtext('{*}PhaseSGN')
        phase_sgn = int(sgn_text) if sgn_text else -1

        timeline = None
        tl = g.find('{*}Timeline')
        if tl is not None:
            timeline = CPHDTimeline(
                collection_start=tl.findtext('{*}CollectionStart'),
                rcv_collection_start=tl.findtext('{*}RcvCollectionStart'),
                tx_time1=_to_float(tl.findtext('{*}TxTime1')),
                tx_time2=_to_float(tl.findtext('{*}TxTime2')),
            )

        tropo = None
        tp = g.find('{*}TropoParameters')
        if tp is not None:
            tropo = CPHDTropoParameters(
                n0=_to_float(tp.findtext('{*}N0')),
                ref_height=tp.findtext('{*}RefHeight'),
            )

        iono = None
        ip = g.find('{*}IonoParameters')
        if ip is not None:
            iono = CPHDIonoParameters(
                tecv=_to_float(ip.findtext('{*}TECV')),
                f2_height=_to_float(ip.findtext('{*}F2Height')),
            )

        return CPHDGlobal(
            domain_type=g.findtext('{*}DomainType'),
            phase_sgn=phase_sgn,
            timeline=timeline,
            fx_band_min=_to_float(g.findtext('{*}FxBand/{*}FxMin')),
            fx_band_max=_to_float(g.findtext('{*}FxBand/{*}FxMax')),
            toa_swath_min=_to_float(g.findtext('{*}TOASwath/{*}TOAMin')),
            toa_swath_max=_to_float(g.findtext('{*}TOASwath/{*}TOAMax')),
            tropo_parameters=tropo,
            iono_parameters=iono,
        )

    # ------------------------------------------------------------------
    # 3. SceneCoordinates
    # ------------------------------------------------------------------

    def _parse_scene_coordinates_sarkit(
        self, xml: Any,
    ) -> Optional[CPHDSceneCoordinates]:
        sc = xml.find('{*}SceneCoordinates')
        if sc is None:
            return None

        iarp_ecf = _parse_xyz(sc.find('{*}IARP/{*}ECF'))
        iarp_llh = _parse_lat_lon_hae(sc.find('{*}IARP/{*}LLH'))

        # ReferenceSurface (Planar XOR HAE)
        reference_surface = None
        rs = sc.find('{*}ReferenceSurface')
        if rs is not None:
            planar = None
            hae = None
            p = rs.find('{*}Planar')
            if p is not None:
                planar = CPHDPlanarSurface(
                    u_iax=_parse_xyz(p.find('{*}uIAX')),
                    u_iay=_parse_xyz(p.find('{*}uIAY')),
                )
            h = rs.find('{*}HAE')
            if h is not None:
                hae = CPHDHAESurface(
                    u_iax_ll=_parse_lat_lon(h.find('{*}uIAXLL')),
                    u_iay_ll=_parse_lat_lon(h.find('{*}uIAYLL')),
                )
            if planar is not None or hae is not None:
                reference_surface = CPHDReferenceSurface(
                    planar=planar, hae=hae,
                )

        image_area = _parse_image_area(sc.find('{*}ImageArea'))
        extended_area = _parse_image_area(sc.find('{*}ExtendedArea'))

        # Convenience tuples (legacy callers)
        image_area_x = None
        image_area_y = None
        if image_area is not None:
            if image_area.x1y1 is not None and image_area.x2y2 is not None:
                image_area_x = (image_area.x1y1[0], image_area.x2y2[0])
                image_area_y = (image_area.x1y1[1], image_area.x2y2[1])

        # Image area corner points (4 IACPs)
        corner_points = None
        iacp_root = sc.find('{*}ImageAreaCornerPoints')
        if iacp_root is not None:
            pts = []
            for cp in iacp_root.findall('{*}IACP'):
                ll = _parse_lat_lon(cp)
                if ll is not None:
                    pts.append([ll[0], ll[1]])
            if pts:
                corner_points = np.array(pts)

        # ImageGrid
        image_grid = self._parse_image_grid_sarkit(sc.find('{*}ImageGrid'))

        return CPHDSceneCoordinates(
            earth_model=sc.findtext('{*}EarthModel'),
            iarp_ecf=iarp_ecf,
            iarp_llh=iarp_llh,
            reference_surface=reference_surface,
            image_area=image_area,
            image_area_x=image_area_x,
            image_area_y=image_area_y,
            corner_points=corner_points,
            extended_area=extended_area,
            image_grid=image_grid,
        )

    def _parse_image_grid_sarkit(
        self, ig: Any,
    ) -> Optional[CPHDImageGrid]:
        if ig is None:
            return None
        iarp_loc = None
        iarp_loc_elem = ig.find('{*}IARPLocation')
        if iarp_loc_elem is not None:
            line = _to_float(iarp_loc_elem.findtext('{*}Line'))
            sample = _to_float(iarp_loc_elem.findtext('{*}Sample'))
            if line is not None or sample is not None:
                iarp_loc = (line or 0.0, sample or 0.0)

        iax = ig.find('{*}IAXExtent')
        iay = ig.find('{*}IAYExtent')

        segments: List[CPHDIASegment] = []
        seg_list = ig.find('{*}SegmentList')
        if seg_list is not None:
            for seg in seg_list.findall('{*}Segment'):
                seg_poly = None
                sp = seg.find('{*}SegmentPolygon')
                if sp is not None:
                    pts = []
                    for sv in sp.findall('{*}SV'):
                        line = _to_float(sv.findtext('{*}Line'))
                        sample = _to_float(sv.findtext('{*}Sample'))
                        if line is not None or sample is not None:
                            pts.append([line or 0.0, sample or 0.0])
                    if pts:
                        seg_poly = np.array(pts)
                segments.append(CPHDIASegment(
                    identifier=seg.findtext('{*}Identifier') or '',
                    start_line=_to_int(seg.findtext('{*}StartLine')) or 0,
                    start_sample=_to_int(seg.findtext('{*}StartSample')) or 0,
                    end_line=_to_int(seg.findtext('{*}EndLine')) or 0,
                    end_sample=_to_int(seg.findtext('{*}EndSample')) or 0,
                    segment_polygon=seg_poly,
                ))

        return CPHDImageGrid(
            identifier=ig.findtext('{*}Identifier'),
            iarp_location=iarp_loc,
            line_spacing=_to_float(_findtext(iax, '{*}LineSpacing')),
            first_line=_to_int(_findtext(iax, '{*}FirstLine')),
            num_lines=_to_int(_findtext(iax, '{*}NumLines')),
            sample_spacing=_to_float(_findtext(iay, '{*}SampleSpacing')),
            first_sample=_to_int(_findtext(iay, '{*}FirstSample')),
            num_samples=_to_int(_findtext(iay, '{*}NumSamples')),
            segments=segments,
        )

    # ------------------------------------------------------------------
    # 4. Data
    # ------------------------------------------------------------------

    def _parse_data_sarkit(self, xml: Any) -> Optional[CPHDData]:
        d = xml.find('{*}Data')
        if d is None:
            return None

        channels: List[CPHDChannel] = []
        for ch in d.findall('{*}Channel'):
            channels.append(CPHDChannel(
                identifier=ch.findtext('{*}Identifier') or '',
                num_vectors=_to_int(ch.findtext('{*}NumVectors')) or 0,
                num_samples=_to_int(ch.findtext('{*}NumSamples')) or 0,
                signal_array_byte_offset=_to_int(
                    ch.findtext('{*}SignalArrayByteOffset'),
                ),
                pvp_array_byte_offset=_to_int(
                    ch.findtext('{*}PVPArrayByteOffset'),
                ),
                compressed_signal_size=_to_int(
                    ch.findtext('{*}CompressedSignalSize'),
                ),
            ))

        support: List[CPHDSupportArraySize] = []
        for sa in d.findall('{*}SupportArray'):
            support.append(CPHDSupportArraySize(
                identifier=sa.findtext('{*}Identifier') or '',
                num_rows=_to_int(sa.findtext('{*}NumRows')) or 0,
                num_cols=_to_int(sa.findtext('{*}NumCols')) or 0,
                bytes_per_element=_to_int(
                    sa.findtext('{*}BytesPerElement'),
                ) or 0,
                array_byte_offset=_to_int(
                    sa.findtext('{*}ArrayByteOffset'),
                ) or 0,
            ))

        return CPHDData(
            signal_array_format=d.findtext('{*}SignalArrayFormat'),
            num_bytes_pvp=_to_int(d.findtext('{*}NumBytesPVP')),
            num_cphd_channels=_to_int(d.findtext('{*}NumCPHDChannels')),
            signal_compression_id=d.findtext('{*}SignalCompressionID'),
            channels=channels,
            num_support_arrays=_to_int(d.findtext('{*}NumSupportArrays')),
            support_arrays=support,
        )

    # ------------------------------------------------------------------
    # 5. Channel section
    # ------------------------------------------------------------------

    def _parse_channel_section_sarkit(
        self, xml: Any,
    ) -> Optional[CPHDChannelSection]:
        c = xml.find('{*}Channel')
        if c is None:
            return None

        params: List[CPHDChannelParameters] = []
        for p in c.findall('{*}Parameters'):
            params.append(self._parse_channel_parameters_sarkit(p))

        added: List[CPHDParameter] = []
        ap = c.find('{*}AddedParameters')
        if ap is not None:
            added = _parse_parameters(ap.findall('{*}Parameter'))

        return CPHDChannelSection(
            ref_ch_id=c.findtext('{*}RefChId'),
            fx_fixed_cphd=_to_bool(c.findtext('{*}FXFixedCPHD')),
            toa_fixed_cphd=_to_bool(c.findtext('{*}TOAFixedCPHD')),
            srp_fixed_cphd=_to_bool(c.findtext('{*}SRPFixedCPHD')),
            parameters=params,
            added_parameters=added,
        )

    def _parse_channel_parameters_sarkit(
        self, p: Any,
    ) -> CPHDChannelParameters:
        polarization = None
        pol = p.find('{*}Polarization')
        if pol is not None:
            polarization = CPHDPolarization(
                tx_pol=pol.findtext('{*}TxPol'),
                rcv_pol=pol.findtext('{*}RcvPol'),
                tx_pol_ref=self._parse_pol_ref(pol.find('{*}TxPolRef')),
                rcv_pol_ref=self._parse_pol_ref(pol.find('{*}RcvPolRef')),
            )

        toa_extended = None
        toa_ext = p.find('{*}TOAExtended')
        if toa_ext is not None:
            lfm_eclipse = None
            le = toa_ext.find('{*}LFMEclipse')
            if le is not None:
                lfm_eclipse = CPHDLFMEclipse(
                    fx_early_low=_to_float(le.findtext('{*}FxEarlyLow')),
                    fx_early_high=_to_float(le.findtext('{*}FxEarlyHigh')),
                    fx_late_low=_to_float(le.findtext('{*}FxLateLow')),
                    fx_late_high=_to_float(le.findtext('{*}FxLateHigh')),
                )
            toa_extended = CPHDTOAExtended(
                toa_ext_saved=_to_float(toa_ext.findtext('{*}TOAExtSaved')),
                lfm_eclipse=lfm_eclipse,
            )

        dwell_times = None
        dt = p.find('{*}DwellTimes')
        if dt is not None:
            dwell_times = CPHDDwellTimesRef(
                cod_id=dt.findtext('{*}CODId'),
                dwell_id=dt.findtext('{*}DwellId'),
                dta_id=dt.findtext('{*}DTAId'),
                use_dta=_to_bool(dt.findtext('{*}UseDTA')),
            )

        antenna_refs = None
        an = p.find('{*}Antenna')
        if an is not None:
            antenna_refs = CPHDChannelAntennaRefs(
                tx_apc_id=an.findtext('{*}TxAPCId'),
                tx_apat_id=an.findtext('{*}TxAPATId'),
                rcv_apc_id=an.findtext('{*}RcvAPCId'),
                rcv_apat_id=an.findtext('{*}RcvAPATId'),
            )

        tx_rcv_refs = None
        tr = p.find('{*}TxRcv')
        if tr is not None:
            tx_rcv_refs = CPHDChannelTxRcvRefs(
                tx_wf_ids=[e.text for e in tr.findall('{*}TxWFId') if e.text],
                rcv_ids=[e.text for e in tr.findall('{*}RcvId') if e.text],
            )

        noise_level = None
        nl = p.find('{*}NoiseLevel')
        if nl is not None:
            profile: List[CPHDFxNoiseProfilePoint] = []
            fnp = nl.find('{*}FxNoiseProfile')
            if fnp is not None:
                for pt in fnp.findall('{*}Point'):
                    profile.append(CPHDFxNoiseProfilePoint(
                        fx=_to_float(pt.findtext('{*}Fx')),
                        pn=_to_float(pt.findtext('{*}PN')),
                    ))
            noise_level = CPHDNoiseLevel(
                pn_ref=_to_float(nl.findtext('{*}PNRef')),
                bn_ref=_to_float(nl.findtext('{*}BNRef')),
                fx_noise_profile=profile,
            )

        return CPHDChannelParameters(
            identifier=p.findtext('{*}Identifier'),
            ref_vector_index=_to_int(p.findtext('{*}RefVectorIndex')),
            fx_fixed=_to_bool(p.findtext('{*}FXFixed')),
            toa_fixed=_to_bool(p.findtext('{*}TOAFixed')),
            srp_fixed=_to_bool(p.findtext('{*}SRPFixed')),
            signal_normal=_to_bool(p.findtext('{*}SignalNormal')),
            polarization=polarization,
            fx_c=_to_float(p.findtext('{*}FxC')),
            fx_bw=_to_float(p.findtext('{*}FxBW')),
            fx_bw_noise=_to_float(p.findtext('{*}FxBWNoise')),
            toa_saved=_to_float(p.findtext('{*}TOASaved')),
            toa_extended=toa_extended,
            dwell_times=dwell_times,
            image_area=_parse_image_area(p.find('{*}ImageArea')),
            antenna=antenna_refs,
            tx_rcv=tx_rcv_refs,
            pt_ref=_to_float(p.findtext('{*}TgtRefLevel/{*}PTRef')),
            noise_level=noise_level,
        )

    @staticmethod
    def _parse_pol_ref(elem: Any) -> Optional[CPHDPolRef]:
        if elem is None:
            return None
        return CPHDPolRef(
            amp_h=_to_float(elem.findtext('{*}AmpH')),
            amp_v=_to_float(elem.findtext('{*}AmpV')),
            phase_v=_to_float(elem.findtext('{*}PhaseV')),
        )

    # ------------------------------------------------------------------
    # 6. PVP — array load (signal block) and definition table
    # ------------------------------------------------------------------

    def _repair_pvp_offsets_sarkit(self, xml: Any) -> None:
        """Repair inconsistent per-channel PVP byte offsets in the XML.

        Some CPHD writers store ``Data/Channel/PVPArrayByteOffset`` values
        that are not laid out as a contiguous PVP block (e.g. scaled by
        ``NumSamples``), so ``read_pvps`` lands on the wrong bytes for
        channels after the first. Recompute each offset as
        ``cumulative_NumVectors * pvp_bytes_per_vector`` and rewrite it
        in-place. No-op when offsets are already consistent.

        Parameters
        ----------
        xml : xml.etree.ElementTree.Element
            The sarkit CPHD metadata XML tree (mutated in-place).
        """
        try:
            from sarkit.cphd._io import get_pvp_dtype
        except Exception as exc:  # pragma: no cover - sarkit internals moved
            logger.debug("PVP offset repair skipped (sarkit API): %s", exc)
            return

        try:
            pvp_bytes = get_pvp_dtype(xml).itemsize
        except Exception as exc:
            logger.debug("PVP offset repair skipped (dtype): %s", exc)
            return

        cumulative = 0
        for ch_elem in xml.findall('{*}Data/{*}Channel'):
            nv_text = ch_elem.findtext('{*}NumVectors')
            if nv_text is None:
                return
            nv = int(nv_text)
            off_elem = ch_elem.find('{*}PVPArrayByteOffset')
            expected = cumulative * pvp_bytes
            actual = int(off_elem.text) if off_elem is not None else expected
            if off_elem is not None and actual != expected:
                logger.info(
                    "Repairing PVP byte offset for channel %s: %d -> %d",
                    ch_elem.findtext('{*}Identifier'), actual, expected,
                )
                off_elem.text = str(expected)
            cumulative += nv

    def read_pvp(self, channel: Union[int, str] = 0) -> CPHDPVP:
        """Read the per-vector parameters (PVP) for a single channel.

        This is the public, backend-dispatching accessor for per-channel
        PVP data. The ``metadata.pvp`` attribute holds only the first
        channel; multi-channel processing (e.g. STAP) should call this
        method for each channel.

        Parameters
        ----------
        channel : int or str
            Channel index (0-based) or channel identifier string.
            Default is 0 (the first channel).

        Returns
        -------
        CPHDPVP
            Per-vector parameter arrays for the requested channel.

        Raises
        ------
        ValueError
            If ``channel`` is a string that does not match any channel
            identifier, or an index outside the channel range.
        """
        channels = self.metadata.channels
        if isinstance(channel, str):
            ids = [c.identifier for c in channels]
            if channel not in ids:
                raise ValueError(
                    f"Unknown channel identifier {channel!r}; "
                    f"available: {ids}"
                )
            index = ids.index(channel)
        else:
            index = int(channel)
            if index < 0 or index >= len(channels):
                raise ValueError(
                    f"Channel index {index} out of range "
                    f"(0..{len(channels) - 1})"
                )

        if self.backend == 'sarkit':
            return self._load_pvp_sarkit(channels[index].identifier)
        return self._load_pvp_sarpy(index)

    def _load_pvp_sarkit(self, channel_id: str) -> CPHDPVP:
        """Read PVP arrays for a channel via the sarkit signal reader."""
        if not channel_id:
            return CPHDPVP()
        try:
            pvp_data = self._reader.read_pvps(channel_id)
        except Exception as e:
            logger.warning("read_pvps failed for %s: %s", channel_id, e)
            return CPHDPVP()

        def _f(name: str) -> Optional[np.ndarray]:
            try:
                arr = pvp_data[name]
                return np.asarray(arr)
            except (KeyError, IndexError, ValueError):
                return None

        added: dict = {}
        try:
            field_names = pvp_data.dtype.names or ()
        except AttributeError:
            field_names = ()
        known = {
            'TxTime', 'TxPos', 'TxVel', 'RcvTime', 'RcvPos', 'RcvVel',
            'SRPPos', 'AmpSF', 'aFDOP', 'aFRR1', 'aFRR2', 'FX1', 'FX2',
            'FXN1', 'FXN2', 'TOA1', 'TOA2', 'TOAE1', 'TOAE2',
            'TDTropoSRP', 'TDIonoSRP', 'SC0', 'SCSS', 'SIGNAL',
            'TxACX', 'TxACY', 'TxEB', 'RcvACX', 'RcvACY', 'RcvEB',
        }
        for name in field_names:
            if name not in known:
                added[name] = _f(name)

        return CPHDPVP(
            tx_time=_f('TxTime'),
            tx_pos=_f('TxPos'),
            tx_vel=_f('TxVel'),
            rcv_time=_f('RcvTime'),
            rcv_pos=_f('RcvPos'),
            rcv_vel=_f('RcvVel'),
            srp_pos=_f('SRPPos'),
            fx1=_f('FX1'),
            fx2=_f('FX2'),
            toa1=_f('TOA1'),
            toa2=_f('TOA2'),
            td_tropo_srp=_f('TDTropoSRP'),
            sc0=_f('SC0'),
            scss=_f('SCSS'),
            a_fdop=_f('aFDOP'),
            a_frr1=_f('aFRR1'),
            a_frr2=_f('aFRR2'),
            amp_sf=_f('AmpSF'),
            signal=_f('SIGNAL'),
            fxn1=_f('FXN1'),
            fxn2=_f('FXN2'),
            toae1=_f('TOAE1'),
            toae2=_f('TOAE2'),
            td_iono_srp=_f('TDIonoSRP'),
            tx_acx=_f('TxACX'),
            tx_acy=_f('TxACY'),
            tx_eb=_f('TxEB'),
            rcv_acx=_f('RcvACX'),
            rcv_acy=_f('RcvACY'),
            rcv_eb=_f('RcvEB'),
            added_pvps=added,
        )

    # ------------------------------------------------------------------
    # 7. SupportArray
    # ------------------------------------------------------------------

    def _parse_support_arrays_sarkit(
        self, xml: Any,
    ) -> Optional[CPHDSupportArrays]:
        sa = xml.find('{*}SupportArray')
        if sa is None:
            return None

        def _core(e: Any) -> dict:
            nodata = e.findtext('{*}NODATA')
            return dict(
                identifier=e.findtext('{*}Identifier') or '',
                element_format=e.findtext('{*}ElementFormat'),
                x0=_to_float(e.findtext('{*}X0')),
                y0=_to_float(e.findtext('{*}Y0')),
                xss=_to_float(e.findtext('{*}XSS')),
                yss=_to_float(e.findtext('{*}YSS')),
                nodata=bytes.fromhex(nodata) if nodata else None,
            )

        iaz = [CPHDIAZArray(**_core(e)) for e in sa.findall('{*}IAZArray')]
        agp = [
            CPHDAntGainPhaseArray(**_core(e))
            for e in sa.findall('{*}AntGainPhase')
        ]
        dta = [
            CPHDDwellTimeArray(**_core(e))
            for e in sa.findall('{*}DwellTimeArray')
        ]

        added: List[CPHDAddedSupportArray] = []
        for e in sa.findall('{*}AddedSupportArray'):
            base = _core(e)
            added.append(CPHDAddedSupportArray(
                **base,
                x_units=e.findtext('{*}XUnits'),
                y_units=e.findtext('{*}YUnits'),
                z_units=e.findtext('{*}ZUnits'),
                parameters=_parse_parameters(e.findall('{*}Parameter')),
            ))

        if not iaz and not agp and not dta and not added:
            return None
        return CPHDSupportArrays(
            iaz_arrays=iaz,
            ant_gain_phase=agp,
            dwell_time_arrays=dta,
            added_support_arrays=added,
        )

    # ------------------------------------------------------------------
    # 8. Dwell
    # ------------------------------------------------------------------

    def _parse_dwell_sarkit(self, xml: Any) -> Optional[CPHDDwell]:
        dw = xml.find('{*}Dwell')
        if dw is None:
            return None

        cod_times: List[CPHDCODTime] = []
        for ct in dw.findall('{*}CODTime'):
            cod_times.append(CPHDCODTime(
                identifier=ct.findtext('{*}Identifier') or '',
                cod_time_poly=_parse_poly2d(ct.find('{*}CODTimePoly')),
            ))

        dwell_times: List[CPHDDwellTime] = []
        for d in dw.findall('{*}DwellTime'):
            dwell_times.append(CPHDDwellTime(
                identifier=d.findtext('{*}Identifier') or '',
                dwell_time_poly=_parse_poly2d(d.find('{*}DwellTimePoly')),
            ))

        return CPHDDwell(
            num_cod_times=_to_int(dw.findtext('{*}NumCODTimes')),
            cod_times=cod_times,
            num_dwell_times=_to_int(dw.findtext('{*}NumDwellTimes')),
            dwell_times=dwell_times,
        )

    # ------------------------------------------------------------------
    # 9. ReferenceGeometry
    # ------------------------------------------------------------------

    def _parse_reference_geometry_sarkit(
        self, xml: Any,
    ) -> Optional[CPHDReferenceGeometry]:
        rg = xml.find('{*}ReferenceGeometry')
        if rg is None:
            return None

        srp_ecf = _parse_xyz(rg.find('{*}SRP/{*}ECF'))
        srp_iac = _parse_xyz(rg.find('{*}SRP/{*}IAC'))
        srp_llh = _parse_lat_lon_hae(rg.find('{*}SRP/{*}LLH'))

        monostatic = None
        bistatic = None
        flat_side = None
        flat_graze = None
        flat_az = None
        flat_twist = None
        flat_slope = None
        flat_layover = None

        m = rg.find('{*}Monostatic')
        if m is not None:
            monostatic = CPHDMonostaticGeometry(
                arp_pos=_parse_xyz(m.find('{*}ARPPos')),
                arp_vel=_parse_xyz(m.find('{*}ARPVel')),
                side_of_track=m.findtext('{*}SideOfTrack'),
                slant_range=_to_float(m.findtext('{*}SlantRange')),
                ground_range=_to_float(m.findtext('{*}GroundRange')),
                doppler_cone_angle=_to_float(
                    m.findtext('{*}DopplerConeAngle'),
                ),
                graze_angle=_to_float(m.findtext('{*}GrazeAngle')),
                incidence_angle=_to_float(m.findtext('{*}IncidenceAngle')),
                azimuth_angle=_to_float(m.findtext('{*}AzimuthAngle')),
                twist_angle=_to_float(m.findtext('{*}TwistAngle')),
                slope_angle=_to_float(m.findtext('{*}SlopeAngle')),
                layover_angle=_to_float(m.findtext('{*}LayoverAngle')),
            )
            flat_side = monostatic.side_of_track
            flat_graze = monostatic.graze_angle
            flat_az = monostatic.azimuth_angle
            flat_twist = monostatic.twist_angle
            flat_slope = monostatic.slope_angle
            flat_layover = monostatic.layover_angle

        b = rg.find('{*}Bistatic')
        if b is not None:
            bistatic = CPHDBistaticGeometry(
                azimuth_angle=_to_float(b.findtext('{*}AzimuthAngle')),
                azimuth_angle_rate=_to_float(
                    b.findtext('{*}AzimuthAngleRate'),
                ),
                bistatic_angle=_to_float(b.findtext('{*}BistaticAngle')),
                bistatic_angle_rate=_to_float(
                    b.findtext('{*}BistaticAngleRate'),
                ),
                graze_angle=_to_float(b.findtext('{*}GrazeAngle')),
                twist_angle=_to_float(b.findtext('{*}TwistAngle')),
                slope_angle=_to_float(b.findtext('{*}SlopeAngle')),
                layover_angle=_to_float(b.findtext('{*}LayoverAngle')),
                tx_platform=self._parse_bistatic_platform(
                    b.find('{*}TxPlatform'),
                ),
                rcv_platform=self._parse_bistatic_platform(
                    b.find('{*}RcvPlatform'),
                ),
            )
            if flat_az is None:
                flat_az = bistatic.azimuth_angle
            if flat_graze is None:
                flat_graze = bistatic.graze_angle
            if flat_twist is None:
                flat_twist = bistatic.twist_angle
            if flat_slope is None:
                flat_slope = bistatic.slope_angle
            if flat_layover is None:
                flat_layover = bistatic.layover_angle

        return CPHDReferenceGeometry(
            ref_time=_to_float(rg.findtext('{*}ReferenceTime')),
            srp_ecf=srp_ecf,
            srp_iac=srp_iac,
            srp_llh=srp_llh,
            srp_cod_time=_to_float(rg.findtext('{*}SRPCODTime')),
            srp_dwell_time=_to_float(rg.findtext('{*}SRPDwellTime')),
            side_of_track=flat_side,
            graze_angle_deg=flat_graze,
            azimuth_angle_deg=flat_az,
            twist_angle_deg=flat_twist,
            slope_angle_deg=flat_slope,
            layover_angle_deg=flat_layover,
            monostatic=monostatic,
            bistatic=bistatic,
        )

    @staticmethod
    def _parse_bistatic_platform(
        elem: Any,
    ) -> Optional[CPHDBistaticPlatform]:
        if elem is None:
            return None
        return CPHDBistaticPlatform(
            time=_to_float(elem.findtext('{*}Time')),
            pos=_parse_xyz(elem.find('{*}Pos')),
            vel=_parse_xyz(elem.find('{*}Vel')),
            side_of_track=elem.findtext('{*}SideOfTrack'),
            slant_range=_to_float(elem.findtext('{*}SlantRange')),
            ground_range=_to_float(elem.findtext('{*}GroundRange')),
            doppler_cone_angle=_to_float(
                elem.findtext('{*}DopplerConeAngle'),
            ),
            graze_angle=_to_float(elem.findtext('{*}GrazeAngle')),
            incidence_angle=_to_float(elem.findtext('{*}IncidenceAngle')),
            azimuth_angle=_to_float(elem.findtext('{*}AzimuthAngle')),
        )

    # ------------------------------------------------------------------
    # 10. Antenna
    # ------------------------------------------------------------------

    def _parse_antenna_sarkit(self, xml: Any) -> Optional[CPHDAntenna]:
        ant = xml.find('{*}Antenna')
        if ant is None:
            return None

        acfs: List[CPHDAntCoordFrame] = []
        for acf in ant.findall('{*}AntCoordFrame'):
            acfs.append(CPHDAntCoordFrame(
                identifier=acf.findtext('{*}Identifier') or '',
                x_axis_poly=_parse_xyz_poly(acf.find('{*}XAxisPoly')),
                y_axis_poly=_parse_xyz_poly(acf.find('{*}YAxisPoly')),
                use_acf_pvp=_to_bool(acf.findtext('{*}UseACFPVP')),
            ))

        apcs: List[CPHDAntPhaseCenter] = []
        for apc in ant.findall('{*}AntPhaseCenter'):
            apcs.append(CPHDAntPhaseCenter(
                identifier=apc.findtext('{*}Identifier') or '',
                acf_id=apc.findtext('{*}ACFId'),
                apc_xyz=_parse_xyz(apc.find('{*}APCXYZ')),
            ))

        patterns: List[CPHDAntPatternFull] = []
        for ap in ant.findall('{*}AntPattern'):
            patterns.append(self._parse_ant_pattern_full(ap))

        return CPHDAntenna(
            num_acfs=_to_int(ant.findtext('{*}NumACFs')),
            num_apcs=_to_int(ant.findtext('{*}NumAPCs')),
            num_ant_pats=_to_int(ant.findtext('{*}NumAntPats')),
            ant_coord_frames=acfs,
            ant_phase_centers=apcs,
            ant_patterns=patterns,
        )

    def _parse_ant_pattern_full(self, ap: Any) -> CPHDAntPatternFull:
        eb_sf = None
        ebsf = ap.find('{*}EBFreqShiftSF')
        if ebsf is not None:
            eb_sf = CPHDEBFreqShiftSF(
                dcx_sf=_to_float(ebsf.findtext('{*}DCXSF')),
                dcy_sf=_to_float(ebsf.findtext('{*}DCYSF')),
            )
        ml_sf = None
        mlsf = ap.find('{*}MLFreqDilationSF')
        if mlsf is not None:
            ml_sf = CPHDMLFreqDilationSF(
                dcx_sf=_to_float(mlsf.findtext('{*}DCXSF')),
                dcy_sf=_to_float(mlsf.findtext('{*}DCYSF')),
            )
        ant_pol_ref = None
        apr = ap.find('{*}AntPolRef')
        if apr is not None:
            ant_pol_ref = CPHDAntPolRef(
                amp_x=_to_float(apr.findtext('{*}AmpX')),
                amp_y=_to_float(apr.findtext('{*}AmpY')),
                phase_y=_to_float(apr.findtext('{*}PhaseY')),
            )

        eb = None
        eb_elem = ap.find('{*}EB')
        if eb_elem is not None:
            eb = CPHDAntEB(
                dcx_poly=_parse_poly1d(eb_elem.find('{*}DCXPoly')),
                dcy_poly=_parse_poly1d(eb_elem.find('{*}DCYPoly')),
                use_eb_pvp=_to_bool(eb_elem.findtext('{*}UseEBPVP')),
            )

        def _gp(e: Any) -> Optional[CPHDAntGainPhasePoly]:
            if e is None:
                return None
            return CPHDAntGainPhasePoly(
                gain_poly=_parse_poly2d(e.find('{*}GainPoly')),
                phase_poly=_parse_poly2d(e.find('{*}PhasePoly')),
                ant_gp_id=e.findtext('{*}AntGPId'),
            )

        gp_arrays: List[CPHDAntGainPhaseFreqEntry] = []
        for gpa in ap.findall('{*}GainPhaseArray'):
            gp_arrays.append(CPHDAntGainPhaseFreqEntry(
                freq=_to_float(gpa.findtext('{*}Freq')),
                array_id=gpa.findtext('{*}ArrayId'),
                element_id=gpa.findtext('{*}ElementId'),
            ))

        return CPHDAntPatternFull(
            identifier=ap.findtext('{*}Identifier') or '',
            freq_zero=_to_float(ap.findtext('{*}FreqZero')),
            gain_zero=_to_float(ap.findtext('{*}GainZero')),
            eb_freq_shift=_to_bool(ap.findtext('{*}EBFreqShift')),
            eb_freq_shift_sf=eb_sf,
            ml_freq_dilation=_to_bool(ap.findtext('{*}MLFreqDilation')),
            ml_freq_dilation_sf=ml_sf,
            gain_bs_poly=_parse_poly1d(ap.find('{*}GainBSPoly')),
            ant_pol_ref=ant_pol_ref,
            eb=eb,
            array=_gp(ap.find('{*}Array')),
            element=_gp(ap.find('{*}Element')),
            gain_phase_arrays=gp_arrays,
        )

    @staticmethod
    def _project_antenna_pattern(
        antenna: Optional[CPHDAntenna],
    ) -> Optional[CPHDAntennaPattern]:
        """Project the first AntPattern + first ACF into the legacy struct.

        Used for backward compatibility with consumers (e.g. ``rda.py``)
        that read ``meta.antenna_pattern.{gain_zero, gain_poly,
        acf_x_poly, acf_y_poly}``.
        """
        if antenna is None:
            return None
        pattern = antenna.ant_patterns[0] if antenna.ant_patterns else None
        acf = (
            antenna.ant_coord_frames[0]
            if antenna.ant_coord_frames else None
        )
        apc = (
            antenna.ant_phase_centers[0]
            if antenna.ant_phase_centers else None
        )
        if pattern is None and acf is None and apc is None:
            return None
        gain_poly = pattern.array.gain_poly if (
            pattern and pattern.array
        ) else None
        eb_dcx = pattern.eb.dcx_poly if (pattern and pattern.eb) else None
        eb_dcy = pattern.eb.dcy_poly if (pattern and pattern.eb) else None
        return CPHDAntennaPattern(
            freq_zero=pattern.freq_zero if pattern else None,
            gain_zero=pattern.gain_zero if pattern else None,
            gain_poly=gain_poly,
            eb_dcx_poly=eb_dcx,
            eb_dcy_poly=eb_dcy,
            acf_x_poly=acf.x_axis_poly if acf else None,
            acf_y_poly=acf.y_axis_poly if acf else None,
            apc_offset=apc.apc_xyz if apc else None,
        )

    # ------------------------------------------------------------------
    # 11. TxRcv
    # ------------------------------------------------------------------

    def _parse_tx_rcv_sarkit(self, xml: Any) -> Optional[CPHDTxRcv]:
        tr = xml.find('{*}TxRcv')
        if tr is None:
            return None

        tx_wfs: List[CPHDTxWaveform] = []
        for wf in tr.findall('{*}TxWFParameters'):
            tx_wfs.append(CPHDTxWaveform(
                identifier=wf.findtext('{*}Identifier'),
                pulse_length=_to_float(wf.findtext('{*}PulseLength')),
                rf_bandwidth=_to_float(wf.findtext('{*}RFBandwidth')),
                freq_center=_to_float(wf.findtext('{*}FreqCenter')),
                lfm_rate=_to_float(wf.findtext('{*}LFMRate')),
                polarization=wf.findtext('{*}Polarization'),
                power=_to_float(wf.findtext('{*}Power')),
            ))

        rcvs: List[CPHDRcvParameters] = []
        for rcv in tr.findall('{*}RcvParameters'):
            rcvs.append(CPHDRcvParameters(
                identifier=rcv.findtext('{*}Identifier'),
                window_length=_to_float(rcv.findtext('{*}WindowLength')),
                sample_rate=_to_float(rcv.findtext('{*}SampleRate')),
                if_filter_bw=_to_float(rcv.findtext('{*}IFFilterBW')),
                freq_center=_to_float(rcv.findtext('{*}FreqCenter')),
                lfm_rate=_to_float(rcv.findtext('{*}LFMRate')),
                polarization=rcv.findtext('{*}Polarization'),
                path_gain=_to_float(rcv.findtext('{*}PathGain')),
            ))

        return CPHDTxRcv(
            num_tx_wfs=_to_int(tr.findtext('{*}NumTxWFs')),
            tx_waveforms=tx_wfs,
            num_rcvs=_to_int(tr.findtext('{*}NumRcvs')),
            rcv_parameters=rcvs,
        )

    # ------------------------------------------------------------------
    # 12. ErrorParameters
    # ------------------------------------------------------------------

    def _parse_error_parameters_sarkit(
        self, xml: Any,
    ) -> Optional[CPHDErrorParameters]:
        ep = xml.find('{*}ErrorParameters')
        if ep is None:
            return None
        return CPHDErrorParameters(
            monostatic=self._parse_monostatic_error(ep.find('{*}Monostatic')),
            bistatic=self._parse_bistatic_error(ep.find('{*}Bistatic')),
        )

    def _parse_pos_vel_err(self, elem: Any) -> Optional[CPHDPosVelErr]:
        if elem is None:
            return None
        cc = None
        cc_elem = elem.find('{*}CorrCoefs')
        if cc_elem is not None:
            cc = CPHDPosVelCorrCoefs(
                p1_p2=_to_float(cc_elem.findtext('{*}P1P2')),
                p1_p3=_to_float(cc_elem.findtext('{*}P1P3')),
                p1_v1=_to_float(cc_elem.findtext('{*}P1V1')),
                p1_v2=_to_float(cc_elem.findtext('{*}P1V2')),
                p1_v3=_to_float(cc_elem.findtext('{*}P1V3')),
                p2_p3=_to_float(cc_elem.findtext('{*}P2P3')),
                p2_v1=_to_float(cc_elem.findtext('{*}P2V1')),
                p2_v2=_to_float(cc_elem.findtext('{*}P2V2')),
                p2_v3=_to_float(cc_elem.findtext('{*}P2V3')),
                p3_v1=_to_float(cc_elem.findtext('{*}P3V1')),
                p3_v2=_to_float(cc_elem.findtext('{*}P3V2')),
                p3_v3=_to_float(cc_elem.findtext('{*}P3V3')),
                v1_v2=_to_float(cc_elem.findtext('{*}V1V2')),
                v1_v3=_to_float(cc_elem.findtext('{*}V1V3')),
                v2_v3=_to_float(cc_elem.findtext('{*}V2V3')),
            )
        return CPHDPosVelErr(
            frame=elem.findtext('{*}Frame'),
            p1=_to_float(elem.findtext('{*}P1')),
            p2=_to_float(elem.findtext('{*}P2')),
            p3=_to_float(elem.findtext('{*}P3')),
            v1=_to_float(elem.findtext('{*}V1')),
            v2=_to_float(elem.findtext('{*}V2')),
            v3=_to_float(elem.findtext('{*}V3')),
            corr_coefs=cc,
            position_decorr=_parse_decorr_func(
                elem.find('{*}PositionDecorr'),
            ),
        )

    def _parse_monostatic_error(
        self, m: Any,
    ) -> Optional[CPHDMonostaticError]:
        if m is None:
            return None
        rs = m.find('{*}RadarSensor')
        radar_sensor = None
        if rs is not None:
            radar_sensor = CPHDMonoRadarSensorError(
                range_bias=_to_float(rs.findtext('{*}RangeBias')),
                clock_freq_sf=_to_float(rs.findtext('{*}ClockFreqSF')),
                collection_start_time=_to_float(
                    rs.findtext('{*}CollectionStartTime'),
                ),
                range_bias_decorr=_parse_decorr_func(
                    rs.find('{*}RangeBiasDecorr'),
                ),
            )
        tropo = None
        te = m.find('{*}TropoError')
        if te is not None:
            tropo = CPHDTropoError(
                tropo_range_vertical=_to_float(
                    te.findtext('{*}TropoRangeVertical'),
                ),
                tropo_range_slant=_to_float(
                    te.findtext('{*}TropoRangeSlant'),
                ),
                tropo_range_decorr=_parse_decorr_func(
                    te.find('{*}TropoRangeDecorr'),
                ),
            )
        iono = None
        ie = m.find('{*}IonoError')
        if ie is not None:
            iono = CPHDIonoError(
                iono_range_vertical=_to_float(
                    ie.findtext('{*}IonoRangeVertical'),
                ),
                iono_range_rate_vertical=_to_float(
                    ie.findtext('{*}IonoRangeRateVertical'),
                ),
                iono_rg_rg_rate_cc=_to_float(
                    ie.findtext('{*}IonoRgRgRateCC'),
                ),
                iono_range_vert_decorr=_parse_decorr_func(
                    ie.find('{*}IonoRangeVertDecorr'),
                ),
            )

        added: List[CPHDParameter] = []
        ap = m.find('{*}AddedParameters')
        if ap is not None:
            added = _parse_parameters(ap.findall('{*}Parameter'))

        return CPHDMonostaticError(
            pos_vel_err=self._parse_pos_vel_err(m.find('{*}PosVelErr')),
            radar_sensor=radar_sensor,
            tropo_error=tropo,
            iono_error=iono,
            added_parameters=added,
        )

    def _parse_bistatic_error(
        self, b: Any,
    ) -> Optional[CPHDBistaticError]:
        if b is None:
            return None

        def _platform(e: Any) -> Optional[CPHDBistaticPlatformError]:
            if e is None:
                return None
            rs = e.find('{*}RadarSensor')
            radar_sensor = None
            if rs is not None:
                radar_sensor = CPHDBistaticRadarSensorError(
                    delay_bias=_to_float(rs.findtext('{*}DelayBias')),
                    clock_freq_sf=_to_float(rs.findtext('{*}ClockFreqSF')),
                    collection_start_time=_to_float(
                        rs.findtext('{*}CollectionStartTime'),
                    ),
                )
            return CPHDBistaticPlatformError(
                pos_vel_err=self._parse_pos_vel_err(e.find('{*}PosVelErr')),
                radar_sensor=radar_sensor,
            )

        added: List[CPHDParameter] = []
        ap = b.find('{*}AddedParameters')
        if ap is not None:
            added = _parse_parameters(ap.findall('{*}Parameter'))

        return CPHDBistaticError(
            tx_platform=_platform(b.find('{*}TxPlatform')),
            rcv_platform=_platform(b.find('{*}RcvPlatform')),
            added_parameters=added,
        )

    # ------------------------------------------------------------------
    # 13. ProductInfo
    # ------------------------------------------------------------------

    def _parse_product_info_sarkit(
        self, xml: Any,
    ) -> Optional[CPHDProductInfo]:
        pi = xml.find('{*}ProductInfo')
        if pi is None:
            return None
        creation_info: List[CPHDCreationInfo] = []
        for ci in pi.findall('{*}CreationInfo'):
            creation_info.append(CPHDCreationInfo(
                application=ci.findtext('{*}Application'),
                date_time=ci.findtext('{*}DateTime'),
                site=ci.findtext('{*}Site'),
                parameters=_parse_parameters(ci.findall('{*}Parameter')),
            ))
        return CPHDProductInfo(
            profile=pi.findtext('{*}Profile'),
            creation_info=creation_info,
            parameters=_parse_parameters(pi.findall('{*}Parameter')),
        )

    # ------------------------------------------------------------------
    # 14. GeoInfo (recursive)
    # ------------------------------------------------------------------

    def _parse_geo_info_sarkit(self, elem: Any) -> CPHDGeoInfo:
        if elem is None:
            return CPHDGeoInfo()
        points: List[Tuple[float, float]] = []
        for pt in elem.findall('{*}Point'):
            ll = _parse_lat_lon(pt)
            if ll is not None:
                points.append(ll)
        lines: List[CPHDGeoLine] = []
        for ln in elem.findall('{*}Line'):
            endpoints: List[Tuple[float, float]] = []
            for ep in ln.findall('{*}Endpoint'):
                ll = _parse_lat_lon(ep)
                if ll is not None:
                    endpoints.append(ll)
            lines.append(CPHDGeoLine(endpoints=endpoints))
        polygons: List[CPHDGeoPolygon] = []
        for poly in elem.findall('{*}Polygon'):
            verts: List[Tuple[float, float]] = []
            for v in poly.findall('{*}Vertex'):
                ll = _parse_lat_lon(v)
                if ll is not None:
                    verts.append(ll)
            polygons.append(CPHDGeoPolygon(vertices=verts))
        nested = [
            self._parse_geo_info_sarkit(g)
            for g in elem.findall('{*}GeoInfo')
        ]
        return CPHDGeoInfo(
            name=elem.get('name'),
            desc=_parse_parameters(elem.findall('{*}Desc')),
            points=points,
            lines=lines,
            polygons=polygons,
            geo_infos=nested,
        )

    # ------------------------------------------------------------------
    # 15. MatchInfo
    # ------------------------------------------------------------------

    def _parse_match_info_sarkit(
        self, xml: Any,
    ) -> Optional[CPHDMatchInfo]:
        mi = xml.find('{*}MatchInfo')
        if mi is None:
            return None
        match_types: List[CPHDMatchType] = []
        for mt in mi.findall('{*}MatchType'):
            collections: List[CPHDMatchCollection] = []
            for mc in mt.findall('{*}MatchCollection'):
                collections.append(CPHDMatchCollection(
                    index=_to_int(mc.get('index')),
                    core_name=mc.findtext('{*}CoreName'),
                    match_index=_to_int(mc.findtext('{*}MatchIndex')),
                    parameters=_parse_parameters(
                        mc.findall('{*}Parameter'),
                    ),
                ))
            match_types.append(CPHDMatchType(
                index=_to_int(mt.get('index')),
                type_id=mt.findtext('{*}TypeID'),
                current_index=_to_int(mt.findtext('{*}CurrentIndex')),
                num_match_collections=_to_int(
                    mt.findtext('{*}NumMatchCollections'),
                ),
                match_collections=collections,
            ))
        return CPHDMatchInfo(
            num_match_types=_to_int(mi.findtext('{*}NumMatchTypes')),
            match_types=match_types,
        )

    # ==================================================================
    # SARPY BACKEND (best-effort fallback)
    # ==================================================================

    def _load_metadata_sarpy(self) -> None:
        """Load metadata via sarpy. Best-effort field coverage."""
        from sarpy.io.phase_history.converter import open_phase_history

        try:
            self._reader = open_phase_history(str(self.filepath))
            self._sarpy_meta = self._reader.cphd_meta

            channels: List[CPHDChannel] = []
            for ch in self._sarpy_meta.Data.Channels:
                channels.append(CPHDChannel(
                    identifier=ch.Identifier or '',
                    num_vectors=ch.NumVectors,
                    num_samples=ch.NumSamples,
                    signal_array_byte_offset=getattr(
                        ch, 'SignalArrayByteOffset', None,
                    ),
                    pvp_array_byte_offset=getattr(
                        ch, 'PVPArrayByteOffset', None,
                    ),
                    compressed_signal_size=getattr(
                        ch, 'CompressedSignalSize', None,
                    ),
                ))

            collection_info = self._collection_info_sarpy()
            global_params = self._global_sarpy()
            tx_rcv = self._tx_rcv_sarpy()
            pvp = self._load_pvp_sarpy(channel=0)

            tx_waveform = (
                tx_rcv.tx_waveforms[0]
                if tx_rcv and tx_rcv.tx_waveforms else None
            )
            rcv_parameters = (
                tx_rcv.rcv_parameters[0]
                if tx_rcv and tx_rcv.rcv_parameters else None
            )

            data = CPHDData(
                num_cphd_channels=getattr(
                    self._sarpy_meta.Data, 'NumCPHDChannels', None,
                ),
                num_bytes_pvp=getattr(
                    self._sarpy_meta.Data, 'NumBytesPVP', None,
                ),
                signal_array_format=getattr(
                    self._sarpy_meta.Data, 'SignalArrayFormat', None,
                ),
                signal_compression_id=getattr(
                    self._sarpy_meta.Data, 'SignalCompressionID', None,
                ),
                channels=channels,
            )

            first_ch = channels[0] if channels else CPHDChannel()
            self.metadata = CPHDMetadata(
                format='CPHD',
                rows=first_ch.num_vectors,
                cols=first_ch.num_samples,
                dtype='complex64',
                collection_info=collection_info,
                global_params=global_params,
                scene_coordinates=None,
                data=data,
                channel_section=None,
                pvp=pvp,
                dwell=None,
                reference_geometry=None,
                support_arrays=None,
                antenna=None,
                tx_rcv=tx_rcv,
                error_parameters=None,
                product_info=None,
                geo_info=[],
                match_info=None,
                channels=channels,
                num_channels=len(channels),
                tx_waveform=tx_waveform,
                rcv_parameters=rcv_parameters,
                antenna_pattern=None,
                extras={'backend': 'sarpy'},
            )

            logger.info(
                "Loaded CPHD %s (%d x %d) via sarpy",
                self.filepath.name,
                first_ch.num_vectors,
                first_ch.num_samples,
            )

        except Exception as e:
            raise ValueError(f"Failed to load CPHD metadata: {e}") from e

    def _collection_info_sarpy(self) -> Optional[CPHDCollectionInfo]:
        ci = getattr(self._sarpy_meta, 'CollectionID', None)
        if ci is None:
            ci = getattr(self._sarpy_meta, 'CollectionInfo', None)
        if ci is None:
            return None
        radar_mode_obj = getattr(ci, 'RadarMode', None)
        params: List[CPHDParameter] = []
        sarpy_params = getattr(ci, 'Parameters', None)
        if sarpy_params:
            for p in sarpy_params:
                params.append(CPHDParameter(
                    name=getattr(p, 'name', '') or '',
                    value=getattr(p, 'value', '') or '',
                ))
        return CPHDCollectionInfo(
            collector_name=getattr(ci, 'CollectorName', None),
            illuminator_name=getattr(ci, 'IlluminatorName', None),
            core_name=getattr(ci, 'CoreName', None),
            collect_type=getattr(ci, 'CollectType', None),
            radar_mode=(
                getattr(radar_mode_obj, 'ModeType', None)
                if radar_mode_obj else None
            ),
            radar_mode_id=(
                getattr(radar_mode_obj, 'ModeID', None)
                if radar_mode_obj else None
            ),
            classification=getattr(ci, 'Classification', None),
            release_info=getattr(ci, 'ReleaseInfo', None),
            country_code=getattr(ci, 'CountryCode', None),
            parameters=params,
        )

    def _global_sarpy(self) -> Optional[CPHDGlobal]:
        g = getattr(self._sarpy_meta, 'Global', None)
        if g is None:
            return None
        sgn_val = getattr(g, 'SGN', None)
        if sgn_val is None:
            sgn_val = getattr(g, 'PhaseSGN', None)
        phase_sgn = int(sgn_val) if sgn_val is not None else -1
        fx_band = getattr(g, 'FxBand', None)
        toa_swath = getattr(g, 'TOASwath', None)

        timeline = None
        tl = getattr(g, 'Timeline', None)
        if tl is not None:
            cs = getattr(tl, 'CollectionStart', None)
            timeline = CPHDTimeline(
                collection_start=cs.isoformat() if cs is not None else None,
                rcv_collection_start=(
                    getattr(tl, 'RcvCollectionStart', None).isoformat()
                    if getattr(tl, 'RcvCollectionStart', None) is not None
                    else None
                ),
                tx_time1=(
                    float(tl.TxTime1)
                    if getattr(tl, 'TxTime1', None) is not None else None
                ),
                tx_time2=(
                    float(tl.TxTime2)
                    if getattr(tl, 'TxTime2', None) is not None else None
                ),
            )

        tropo = None
        tp = getattr(g, 'TropoParameters', None)
        if tp is not None:
            tropo = CPHDTropoParameters(
                n0=getattr(tp, 'N0', None),
                ref_height=getattr(tp, 'RefHeight', None),
            )

        iono = None
        ip = getattr(g, 'IonoParameters', None)
        if ip is not None:
            iono = CPHDIonoParameters(
                tecv=getattr(ip, 'TECV', None),
                f2_height=getattr(ip, 'F2Height', None),
            )

        return CPHDGlobal(
            domain_type=getattr(g, 'DomainType', None),
            phase_sgn=phase_sgn,
            timeline=timeline,
            fx_band_min=(
                float(fx_band.get_array()[0])
                if fx_band is not None else None
            ),
            fx_band_max=(
                float(fx_band.get_array()[1])
                if fx_band is not None else None
            ),
            toa_swath_min=(
                float(toa_swath.get_array()[0])
                if toa_swath is not None else None
            ),
            toa_swath_max=(
                float(toa_swath.get_array()[1])
                if toa_swath is not None else None
            ),
            tropo_parameters=tropo,
            iono_parameters=iono,
        )

    def _tx_rcv_sarpy(self) -> Optional[CPHDTxRcv]:
        txrcv = getattr(self._sarpy_meta, 'TxRcv', None)
        if txrcv is None:
            return None
        tx_wfs: List[CPHDTxWaveform] = []
        for wf in getattr(txrcv, 'TxWFParameters', None) or []:
            tx_wfs.append(CPHDTxWaveform(
                identifier=getattr(wf, 'Identifier', None),
                pulse_length=(
                    float(wf.PulseLength)
                    if getattr(wf, 'PulseLength', None) is not None
                    else None
                ),
                rf_bandwidth=(
                    float(wf.RFBandwidth)
                    if getattr(wf, 'RFBandwidth', None) is not None
                    else None
                ),
                freq_center=(
                    float(wf.FreqCenter)
                    if getattr(wf, 'FreqCenter', None) is not None
                    else None
                ),
                lfm_rate=(
                    float(wf.LFMRate)
                    if getattr(wf, 'LFMRate', None) is not None else None
                ),
                polarization=getattr(wf, 'Polarization', None),
                power=(
                    float(wf.Power)
                    if getattr(wf, 'Power', None) is not None else None
                ),
            ))
        rcvs: List[CPHDRcvParameters] = []
        for rcv in getattr(txrcv, 'RcvParameters', None) or []:
            rcvs.append(CPHDRcvParameters(
                identifier=getattr(rcv, 'Identifier', None),
                window_length=(
                    float(rcv.WindowLength)
                    if getattr(rcv, 'WindowLength', None) is not None
                    else None
                ),
                sample_rate=(
                    float(rcv.SampleRate)
                    if getattr(rcv, 'SampleRate', None) is not None
                    else None
                ),
                if_filter_bw=(
                    float(rcv.IFFilterBW)
                    if getattr(rcv, 'IFFilterBW', None) is not None
                    else None
                ),
                freq_center=(
                    float(rcv.FreqCenter)
                    if getattr(rcv, 'FreqCenter', None) is not None
                    else None
                ),
                lfm_rate=(
                    float(rcv.LFMRate)
                    if getattr(rcv, 'LFMRate', None) is not None else None
                ),
                polarization=getattr(rcv, 'Polarization', None),
                path_gain=(
                    float(rcv.PathGain)
                    if getattr(rcv, 'PathGain', None) is not None else None
                ),
            ))
        return CPHDTxRcv(
            num_tx_wfs=getattr(txrcv, 'NumTxWFs', None),
            tx_waveforms=tx_wfs,
            num_rcvs=getattr(txrcv, 'NumRcvs', None),
            rcv_parameters=rcvs,
        )

    def _load_pvp_sarpy(self, channel: int = 0) -> CPHDPVP:
        pvp_raw = self._reader.read_pvp_array(channel)
        meta_pvp = self._sarpy_meta.PVP
        pvp_dict = meta_pvp.to_dict() if meta_pvp else {}
        pvp_keys = set(pvp_dict.keys())

        def _f(name: str) -> Optional[np.ndarray]:
            if name not in pvp_keys:
                return None
            items = list(pvp_dict.items())
            idx = [i for i, (k, _) in enumerate(items) if k == name]
            if not idx:
                return None
            return np.array([val[idx[0]] for val in pvp_raw])

        return CPHDPVP(
            tx_time=_f('TxTime'),
            tx_pos=_f('TxPos'),
            tx_vel=_f('TxVel'),
            rcv_time=_f('RcvTime'),
            rcv_pos=_f('RcvPos'),
            rcv_vel=_f('RcvVel'),
            srp_pos=_f('SRPPos'),
            fx1=_f('FX1'),
            fx2=_f('FX2'),
            toa1=_f('TOA1'),
            toa2=_f('TOA2'),
            td_tropo_srp=_f('TDTropoSRP'),
            sc0=_f('SC0'),
            scss=_f('SCSS'),
            a_fdop=_f('aFDOP'),
            a_frr1=_f('aFRR1'),
            a_frr2=_f('aFRR2'),
            amp_sf=_f('AmpSF'),
            signal=_f('SIGNAL'),
            fxn1=_f('FXN1'),
            fxn2=_f('FXN2'),
            toae1=_f('TOAE1'),
            toae2=_f('TOAE2'),
            td_iono_srp=_f('TDIonoSRP'),
        )

    # ==================================================================
    # IO methods
    # ==================================================================

    def read_chip(
        self,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        bands: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Read phase history vectors.

        Parameters
        ----------
        row_start : int
            Starting vector index.
        row_end : int
            Ending vector index.
        col_start : int
            Starting sample index.
        col_end : int
            Ending sample index.
        bands : Optional[List[int]]
            Channel indices to read. If None, reads the first channel.

        Returns
        -------
        np.ndarray
            Phase history data.
        """
        channel = bands[0] if bands else 0

        if self.backend == 'sarkit':
            ch_id = self.metadata.channels[channel].identifier
            data = self._reader.read_signal(
                ch_id,
                start_vector=row_start,
                stop_vector=row_end,
            )[:, col_start:col_end]
        else:
            data = self._reader.read_chip(
                index=channel,
                dim1_range=(row_start, row_end),
                dim2_range=(col_start, col_end),
            )
        return self._assert_2d(
            data,
            context=f'{type(self).__name__}.read_chip',
            strict=self._enforce_2d,
        )

    def read_full(self, bands: Optional[List[int]] = None) -> np.ndarray:
        """Read full phase history data.

        Parameters
        ----------
        bands : Optional[List[int]]
            Channel indices to read. If None, reads the first channel.

        Returns
        -------
        np.ndarray
            Phase history data for the specified channel.
        """
        channel = bands[0] if bands else 0

        if self.backend == 'sarkit':
            ch_id = self.metadata.channels[channel].identifier
            data = self._reader.read_signal(ch_id)
        else:
            data = self._reader.read(index=channel)
        return self._assert_2d(
            data,
            context=f'{type(self).__name__}.read_full',
            strict=self._enforce_2d,
        )

    def get_shape(self) -> Tuple[int, ...]:
        """Get phase history dimensions for first channel."""
        first_ch = self.metadata.channels[0]
        return (first_ch.num_vectors, first_ch.num_samples)

    def get_dtype(self) -> np.dtype:
        """Get data type — ``complex64`` for CPHD."""
        return np.dtype('complex64')

    def close(self) -> None:
        """Close the reader and release resources."""
        if self.backend == 'sarkit':
            if hasattr(self, '_reader') and self._reader is not None:
                self._reader.done()
            if hasattr(self, '_file_handle') and self._file_handle is not None:
                self._file_handle.close()
        else:
            if hasattr(self, '_reader') and self._reader is not None:
                self._reader.close()
