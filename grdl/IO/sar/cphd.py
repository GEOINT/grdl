# -*- coding: utf-8 -*-
"""
CPHD Reader - Compensated Phase History Data format.

NGA standard for phase history data (unfocused SAR). Uses sarkit as the
primary backend with sarpy as fallback.

Dependencies
------------
sarkit (primary) or sarpy (fallback)

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
2026-02-09

Modified
--------
2026-02-12
"""

# Standard library
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party
import numpy as np

# GRDL internal
from grdl.IO.base import ImageReader
from grdl.IO.models import ImageMetadata
from grdl.IO.models.cphd import (
    CPHDMetadata,
    CPHDChannel,
    CPHDPVP,
    CPHDGlobal,
    CPHDCollectionInfo,
    CPHDTxWaveform,
    CPHDRcvParameters,
    CPHDAntennaPattern,
    CPHDSceneCoordinates,
    CPHDReferenceGeometry,
    CPHDDwellPolynomial,
)
from grdl.IO.sar._backend import (
    _HAS_SARKIT,
    _HAS_SARPY,
    require_sar_backend,
)


class CPHDReader(ImageReader):
    """Read CPHD (Compensated Phase History Data) format.

    CPHD is the NGA standard for phase history data. This reader uses
    sarkit as the primary backend with sarpy as fallback.

    Parameters
    ----------
    filepath : str or Path
        Path to the CPHD file.

    Attributes
    ----------
    filepath : Path
        Path to the CPHD file.
    metadata : Dict[str, Any]
        Standardized metadata dictionary.
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

    def __init__(self, filepath: Union[str, Path]) -> None:
        self.backend = require_sar_backend('CPHD')
        super().__init__(filepath)

    def _load_metadata(self) -> None:
        """Load CPHD metadata using the active backend."""
        if self.backend == 'sarkit':
            self._load_metadata_sarkit()
        else:
            self._load_metadata_sarpy()

    def _load_metadata_sarkit(self) -> None:
        """Load metadata via sarkit."""
        import sarkit.cphd

        try:
            self._file_handle = open(str(self.filepath), 'rb')
            self._reader = sarkit.cphd.Reader(self._file_handle)

            xml = self._reader.metadata.xmltree

            # Extract channel information
            channel_list: list[CPHDChannel] = []
            for ch_elem in xml.findall('{*}Data/{*}Channel'):
                ch_id = ch_elem.findtext('{*}Identifier')
                num_vectors = int(ch_elem.findtext('{*}NumVectors'))
                num_samples = int(ch_elem.findtext('{*}NumSamples'))
                offset_text = ch_elem.findtext(
                    '{*}SignalArrayByteOffset'
                )
                channel_list.append(CPHDChannel(
                    identifier=ch_id or '',
                    num_vectors=num_vectors,
                    num_samples=num_samples,
                    signal_array_byte_offset=(
                        int(offset_text) if offset_text else None
                    ),
                ))

            # Store first channel ID for default operations
            self._default_channel = (
                channel_list[0].identifier if channel_list else ''
            )
            self._xmltree = xml

            # Collection info
            collection_info = CPHDCollectionInfo(
                collector_name=xml.findtext(
                    '{*}CollectionInfo/{*}CollectorName'
                ),
                core_name=xml.findtext(
                    '{*}CollectionInfo/{*}CoreName'
                ),
                classification=xml.findtext(
                    '{*}CollectionInfo/{*}Classification'
                ),
                collect_type=xml.findtext(
                    '{*}CollectionInfo/{*}CollectType'
                ),
                radar_mode=xml.findtext(
                    '{*}CollectionInfo/{*}RadarMode/{*}ModeType'
                ),
                radar_mode_id=xml.findtext(
                    '{*}CollectionInfo/{*}RadarMode/{*}ModeID'
                ),
            )

            # Global parameters
            domain_type = xml.findtext('{*}Global/{*}DomainType')
            sgn_text = xml.findtext('{*}Global/{*}PhaseSGN')
            if sgn_text is None:
                sgn_text = xml.findtext('{*}Global/{*}SGN')
            phase_sgn = int(sgn_text) if sgn_text else -1
            fx_min_text = xml.findtext('{*}Global/{*}FxBand/{*}FxMin')
            fx_max_text = xml.findtext('{*}Global/{*}FxBand/{*}FxMax')
            toa_min_text = xml.findtext(
                '{*}Global/{*}TOASwath/{*}TOAMin'
            )
            toa_max_text = xml.findtext(
                '{*}Global/{*}TOASwath/{*}TOAMax'
            )
            global_params = CPHDGlobal(
                domain_type=domain_type,
                phase_sgn=phase_sgn,
                fx_band_min=(
                    float(fx_min_text) if fx_min_text else None
                ),
                fx_band_max=(
                    float(fx_max_text) if fx_max_text else None
                ),
                toa_swath_min=(
                    float(toa_min_text) if toa_min_text else None
                ),
                toa_swath_max=(
                    float(toa_max_text) if toa_max_text else None
                ),
            )

            # TxRcv parameters
            tx_waveform = self._parse_tx_waveform_sarkit(xml)
            rcv_parameters = self._parse_rcv_params_sarkit(xml)

            # Antenna pattern
            antenna_pattern = self._parse_antenna_sarkit(xml)

            # Scene coordinates
            scene_coordinates = self._parse_scene_coordinates_sarkit(
                xml,
            )

            # Reference geometry
            reference_geometry = (
                self._parse_reference_geometry_sarkit(xml)
            )

            # Dwell polynomials
            dwell = self._parse_dwell_sarkit(xml)

            # PVP (Per-Vector Parameters) for first channel
            pvp = self._load_pvp_sarkit(self._default_channel)

            # Use first channel dimensions as rows/cols
            first_ch = channel_list[0] if channel_list else CPHDChannel()
            self.metadata = CPHDMetadata(
                format='CPHD',
                rows=first_ch.num_vectors,
                cols=first_ch.num_samples,
                dtype='complex64',
                channels=channel_list,
                pvp=pvp,
                global_params=global_params,
                collection_info=collection_info,
                tx_waveform=tx_waveform,
                rcv_parameters=rcv_parameters,
                antenna_pattern=antenna_pattern,
                scene_coordinates=scene_coordinates,
                reference_geometry=reference_geometry,
                dwell=dwell,
                num_channels=len(channel_list),
                extras={'backend': 'sarkit'},
            )

        except Exception as e:
            raise ValueError(f"Failed to load CPHD metadata: {e}") from e

    def _parse_tx_waveform_sarkit(self, xml) -> Optional[CPHDTxWaveform]:
        """Extract transmit waveform parameters from sarkit XML."""
        elem = xml.find('{*}TxRcv/{*}TxWFParameters')
        if elem is None:
            return None
        lfm_text = elem.findtext('{*}LFMRate')
        pulse_text = elem.findtext('{*}PulseLength')
        wf_id = elem.findtext('{*}Identifier')
        return CPHDTxWaveform(
            lfm_rate=abs(float(lfm_text)) if lfm_text else None,
            pulse_length=float(pulse_text) if pulse_text else None,
            identifier=wf_id,
        )

    def _parse_rcv_params_sarkit(self, xml) -> Optional[CPHDRcvParameters]:
        """Extract receive parameters from sarkit XML."""
        elem = xml.find('{*}TxRcv/{*}RcvParameters')
        if elem is None:
            return None
        win_text = elem.findtext('{*}WindowLength')
        rate_text = elem.findtext('{*}SampleRate')
        rcv_id = elem.findtext('{*}Identifier')
        return CPHDRcvParameters(
            window_length=float(win_text) if win_text else None,
            sample_rate=float(rate_text) if rate_text else None,
            identifier=rcv_id,
        )

    def _load_pvp_sarkit(self, channel_id: str) -> CPHDPVP:
        """Load per-vector parameters from sarkit for a channel."""
        pvp_data = self._reader.read_pvps(channel_id)

        def _get_field(name: str) -> Optional[np.ndarray]:
            """Extract a PVP field by name, returning None if absent."""
            try:
                arr = pvp_data[name]
                return np.asarray(arr)
            except (KeyError, IndexError):
                return None

        return CPHDPVP(
            tx_time=_get_field('TxTime'),
            tx_pos=_get_field('TxPos'),
            tx_vel=_get_field('TxVel'),
            rcv_time=_get_field('RcvTime'),
            rcv_pos=_get_field('RcvPos'),
            rcv_vel=_get_field('RcvVel'),
            srp_pos=_get_field('SRPPos'),
            fx1=_get_field('FX1'),
            fx2=_get_field('FX2'),
            sc0=_get_field('SC0'),
            scss=_get_field('SCSS'),
            signal=_get_field('SIGNAL'),
            a_fdop=_get_field('aFDOP'),
            a_frr1=_get_field('aFRR1'),
            a_frr2=_get_field('aFRR2'),
            amp_sf=_get_field('AmpSF'),
            toa1=_get_field('TOA1'),
            toa2=_get_field('TOA2'),
        )

    # ------------------------------------------------------------------
    # Antenna pattern extraction (sarkit)
    # ------------------------------------------------------------------

    def _parse_antenna_sarkit(self, xml) -> Optional[CPHDAntennaPattern]:
        """Extract antenna pattern from sarkit XML."""
        ant_pat = xml.find('{*}Antenna/{*}AntPattern')
        if ant_pat is None:
            return None

        freq_text = ant_pat.findtext('{*}FreqZero')
        gain_text = ant_pat.findtext('{*}GainZero')

        # 2D gain polynomial
        gain_poly = None
        gp_elem = ant_pat.find('{*}Array/{*}GainPoly')
        if gp_elem is not None:
            gain_poly = self._parse_poly2d(gp_elem)

        # Electrical boresight polynomials
        eb_dcx = None
        eb_dcy = None
        eb_elem = ant_pat.find('{*}EB')
        if eb_elem is not None:
            dcx_elem = eb_elem.find('{*}DCXPoly')
            dcy_elem = eb_elem.find('{*}DCYPoly')
            if dcx_elem is not None:
                eb_dcx = self._parse_poly1d(dcx_elem)
            if dcy_elem is not None:
                eb_dcy = self._parse_poly1d(dcy_elem)

        # ACF (Antenna Coordinate Frame) polynomials
        acf_x = None
        acf_y = None
        acf_elem = xml.find('{*}Antenna/{*}AntCoordFrame')
        if acf_elem is not None:
            xp = acf_elem.find('{*}XAxisPoly')
            yp = acf_elem.find('{*}YAxisPoly')
            if xp is not None:
                acf_x = self._parse_xyz_poly(xp)
            if yp is not None:
                acf_y = self._parse_xyz_poly(yp)

        # APC offset
        apc_offset = None
        apc_elem = xml.find('{*}Antenna/{*}AntPhaseCenter')
        if apc_elem is not None:
            acf_x_text = apc_elem.findtext('{*}ACFId')
            offset_elem = apc_elem.find('{*}APCXYZ')
            if offset_elem is not None:
                x = float(offset_elem.findtext('{*}X') or '0')
                y = float(offset_elem.findtext('{*}Y') or '0')
                z = float(offset_elem.findtext('{*}Z') or '0')
                apc_offset = np.array([x, y, z])

        return CPHDAntennaPattern(
            freq_zero=float(freq_text) if freq_text else None,
            gain_zero=float(gain_text) if gain_text else None,
            gain_poly=gain_poly,
            eb_dcx_poly=eb_dcx,
            eb_dcy_poly=eb_dcy,
            acf_x_poly=acf_x,
            acf_y_poly=acf_y,
            apc_offset=apc_offset,
        )

    # ------------------------------------------------------------------
    # Scene coordinates extraction (sarkit)
    # ------------------------------------------------------------------

    def _parse_scene_coordinates_sarkit(
        self, xml,
    ) -> Optional[CPHDSceneCoordinates]:
        """Extract scene coordinates from sarkit XML."""
        sc = xml.find('{*}SceneCoordinates')
        if sc is None:
            return None

        earth_model = sc.findtext('{*}EarthModel')

        # IARP ECF
        iarp_ecf = None
        iarp_ecf_elem = sc.find('{*}IARP/{*}ECF')
        if iarp_ecf_elem is not None:
            x = float(iarp_ecf_elem.findtext('{*}X') or '0')
            y = float(iarp_ecf_elem.findtext('{*}Y') or '0')
            z = float(iarp_ecf_elem.findtext('{*}Z') or '0')
            iarp_ecf = np.array([x, y, z])

        # IARP LLH
        iarp_llh = None
        iarp_llh_elem = sc.find('{*}IARP/{*}LLH')
        if iarp_llh_elem is not None:
            lat = float(iarp_llh_elem.findtext('{*}Lat') or '0')
            lon = float(iarp_llh_elem.findtext('{*}Lon') or '0')
            hae = float(iarp_llh_elem.findtext('{*}HAE') or '0')
            iarp_llh = np.array([lat, lon, hae])

        # Image area extent
        image_area_x = None
        image_area_y = None
        ia_elem = sc.find('{*}ImageAreaCornerPoints')
        # Also try ImageArea for X1Y1/X2Y2 style
        ia_xy = sc.find('{*}ImageArea')
        if ia_xy is not None:
            x1y1 = ia_xy.find('{*}X1Y1')
            x2y2 = ia_xy.find('{*}X2Y2')
            if x1y1 is not None and x2y2 is not None:
                x1 = float(x1y1.findtext('{*}X') or '0')
                y1 = float(x1y1.findtext('{*}Y') or '0')
                x2 = float(x2y2.findtext('{*}X') or '0')
                y2 = float(x2y2.findtext('{*}Y') or '0')
                image_area_x = (x1, x2)
                image_area_y = (y1, y2)

        # Corner points
        corner_points = None
        iacp = sc.find('{*}ImageAreaCornerPoints')
        if iacp is not None:
            pts = []
            for cp in iacp.findall('{*}IACP'):
                lat = float(cp.findtext('{*}Lat') or '0')
                lon = float(cp.findtext('{*}Lon') or '0')
                pts.append([lat, lon])
            if pts:
                corner_points = np.array(pts)

        return CPHDSceneCoordinates(
            earth_model=earth_model,
            iarp_ecf=iarp_ecf,
            iarp_llh=iarp_llh,
            image_area_x=image_area_x,
            image_area_y=image_area_y,
            corner_points=corner_points,
        )

    # ------------------------------------------------------------------
    # Reference geometry extraction (sarkit)
    # ------------------------------------------------------------------

    def _parse_reference_geometry_sarkit(
        self, xml,
    ) -> Optional[CPHDReferenceGeometry]:
        """Extract reference geometry from sarkit XML."""
        rg = xml.find('{*}ReferenceGeometry')
        if rg is None:
            return None

        ref_time_text = rg.findtext('{*}ReferenceTime')

        # SRP
        srp_ecf = None
        srp_llh = None
        srp_ecf_elem = rg.find('{*}SRP/{*}ECF')
        if srp_ecf_elem is not None:
            x = float(srp_ecf_elem.findtext('{*}X') or '0')
            y = float(srp_ecf_elem.findtext('{*}Y') or '0')
            z = float(srp_ecf_elem.findtext('{*}Z') or '0')
            srp_ecf = np.array([x, y, z])
        srp_llh_elem = rg.find('{*}SRP/{*}IAC')
        if srp_llh_elem is None:
            srp_llh_elem = rg.find('{*}SRP/{*}LLH')
        if srp_llh_elem is not None:
            lat = float(srp_llh_elem.findtext('{*}Lat') or '0')
            lon = float(srp_llh_elem.findtext('{*}Lon') or '0')
            hae = float(srp_llh_elem.findtext('{*}HAE') or '0')
            srp_llh = np.array([lat, lon, hae])

        # Monostatic geometry
        mono = rg.find('{*}Monostatic')
        side = None
        graze = None
        azim = None
        twist = None
        slope = None
        layover = None
        if mono is not None:
            side = mono.findtext('{*}SideOfTrack')
            graze_text = mono.findtext('{*}GrazeAng')
            azim_text = mono.findtext('{*}AzimuthAngle')
            twist_text = mono.findtext('{*}TwistAngle')
            slope_text = mono.findtext('{*}SlopeAngle')
            layover_text = mono.findtext('{*}LayoverAngle')
            graze = float(graze_text) if graze_text else None
            azim = float(azim_text) if azim_text else None
            twist = float(twist_text) if twist_text else None
            slope = float(slope_text) if slope_text else None
            layover = float(layover_text) if layover_text else None

        return CPHDReferenceGeometry(
            ref_time=(
                float(ref_time_text) if ref_time_text else None
            ),
            srp_ecf=srp_ecf,
            srp_llh=srp_llh,
            side_of_track=side,
            graze_angle_deg=graze,
            azimuth_angle_deg=azim,
            twist_angle_deg=twist,
            slope_angle_deg=slope,
            layover_angle_deg=layover,
        )

    # ------------------------------------------------------------------
    # Dwell polynomial extraction (sarkit)
    # ------------------------------------------------------------------

    def _parse_dwell_sarkit(self, xml) -> Optional[CPHDDwellPolynomial]:
        """Extract dwell polynomials from sarkit XML."""
        dwell = xml.find('{*}Dwell')
        if dwell is None:
            return None

        cod_poly = None
        dwell_poly = None

        cod_elem = dwell.find('{*}CODPolynomial')
        if cod_elem is None:
            cod_elem = dwell.find('{*}CODTime/{*}CODTimePoly')
        if cod_elem is not None:
            cod_poly = self._parse_poly1d(cod_elem)

        dwell_elem = dwell.find('{*}DwellTimePolynomial')
        if dwell_elem is None:
            dwell_elem = dwell.find('{*}DwellTime/{*}DwellTimePoly')
        if dwell_elem is not None:
            dwell_poly = self._parse_poly1d(dwell_elem)

        if cod_poly is None and dwell_poly is None:
            return None

        return CPHDDwellPolynomial(
            cod_time_poly=cod_poly,
            dwell_time_poly=dwell_poly,
        )

    # ------------------------------------------------------------------
    # XML polynomial parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_poly1d(elem) -> Optional[np.ndarray]:
        """Parse a 1D polynomial from XML ``Coef`` elements.

        Returns
        -------
        np.ndarray or None
            Coefficient array ordered by exponent, or None.
        """
        coefs = elem.findall('{*}Coef')
        if not coefs:
            # Try text content (single scalar polynomial)
            text = elem.text
            if text is not None:
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

    @staticmethod
    def _parse_poly2d(elem) -> Optional[np.ndarray]:
        """Parse a 2D polynomial from XML ``Coef`` elements.

        Returns
        -------
        np.ndarray or None
            2D coefficient array, shape ``(M, N)``, or None.
        """
        coefs = elem.findall('{*}Coef')
        if not coefs:
            return None

        max_e1 = max(
            int(c.get('exponent1', '0')) for c in coefs
        )
        max_e2 = max(
            int(c.get('exponent2', '0')) for c in coefs
        )
        result = np.zeros((max_e1 + 1, max_e2 + 1))
        for c in coefs:
            e1 = int(c.get('exponent1', '0'))
            e2 = int(c.get('exponent2', '0'))
            result[e1, e2] = float(c.text)
        return result

    @staticmethod
    def _parse_xyz_poly(elem) -> Optional[np.ndarray]:
        """Parse an XYZ polynomial (3 component polys) from XML.

        Returns
        -------
        np.ndarray or None
            Shape ``(K, 3)`` where K is the polynomial order + 1.
        """
        x_elem = elem.find('{*}X')
        y_elem = elem.find('{*}Y')
        z_elem = elem.find('{*}Z')
        if x_elem is None:
            return None

        x_coefs = x_elem.findall('{*}Coef')
        y_coefs = y_elem.findall('{*}Coef') if y_elem is not None else []
        z_coefs = z_elem.findall('{*}Coef') if z_elem is not None else []

        max_order = max(
            len(x_coefs), len(y_coefs), len(z_coefs),
        )
        if max_order == 0:
            return None

        result = np.zeros((max_order, 3))
        for i, c in enumerate(x_coefs):
            exp = int(c.get('exponent1', c.get('exponent', i)))
            result[exp, 0] = float(c.text)
        for i, c in enumerate(y_coefs):
            exp = int(c.get('exponent1', c.get('exponent', i)))
            result[exp, 1] = float(c.text)
        for i, c in enumerate(z_coefs):
            exp = int(c.get('exponent1', c.get('exponent', i)))
            result[exp, 2] = float(c.text)
        return result

    def _load_metadata_sarpy(self) -> None:
        """Load metadata via sarpy (fallback)."""
        from sarpy.io.phase_history.converter import open_phase_history

        try:
            self._reader = open_phase_history(str(self.filepath))
            self._sarpy_meta = self._reader.cphd_meta

            # Channels
            channel_list: list[CPHDChannel] = []
            for ch in self._sarpy_meta.Data.Channels:
                channel_list.append(CPHDChannel(
                    identifier=ch.Identifier or '',
                    num_vectors=ch.NumVectors,
                    num_samples=ch.NumSamples,
                ))

            # Collection info
            ci = self._sarpy_meta.CollectionInfo
            radar_mode_obj = getattr(ci, 'RadarMode', None)
            collection_info = CPHDCollectionInfo(
                collector_name=getattr(ci, 'CollectorName', None),
                core_name=getattr(ci, 'CoreName', None),
                classification=getattr(ci, 'Classification', None),
                collect_type=getattr(ci, 'CollectType', None),
                radar_mode=(
                    getattr(radar_mode_obj, 'ModeType', None)
                    if radar_mode_obj else None
                ),
                radar_mode_id=(
                    getattr(radar_mode_obj, 'ModeID', None)
                    if radar_mode_obj else None
                ),
            )

            # Global parameters
            global_obj = self._sarpy_meta.Global
            sgn_val = getattr(global_obj, 'PhaseSGN', None)
            if sgn_val is None:
                sgn_val = getattr(global_obj, 'SGN', None)
            phase_sgn = int(sgn_val) if sgn_val is not None else -1
            fx_band = getattr(global_obj, 'FxBand', None)
            toa_swath = getattr(global_obj, 'TOASwath', None)
            global_params = CPHDGlobal(
                domain_type=getattr(global_obj, 'DomainType', None),
                phase_sgn=phase_sgn,
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
            )

            # TxRcv parameters
            tx_waveform = self._parse_tx_waveform_sarpy()
            rcv_parameters = self._parse_rcv_params_sarpy()

            # PVP for first channel
            pvp = self._load_pvp_sarpy(channel=0)

            # Use first channel dimensions as rows/cols
            first_ch = channel_list[0] if channel_list else CPHDChannel()
            self.metadata = CPHDMetadata(
                format='CPHD',
                rows=first_ch.num_vectors,
                cols=first_ch.num_samples,
                dtype='complex64',
                channels=channel_list,
                pvp=pvp,
                global_params=global_params,
                collection_info=collection_info,
                tx_waveform=tx_waveform,
                rcv_parameters=rcv_parameters,
                # sarpy backend: antenna/scene/refgeo/dwell
                # extraction deferred (sarkit preferred)
                antenna_pattern=None,
                scene_coordinates=None,
                reference_geometry=None,
                dwell=None,
                num_channels=(
                    self._sarpy_meta.Data.NumCPHDChannels
                ),
                extras={'backend': 'sarpy'},
            )

        except Exception as e:
            raise ValueError(f"Failed to load CPHD metadata: {e}") from e

    def _parse_tx_waveform_sarpy(self) -> Optional[CPHDTxWaveform]:
        """Extract transmit waveform parameters from sarpy metadata."""
        txrcv = getattr(self._sarpy_meta, 'TxRcv', None)
        if txrcv is None:
            return None
        tx_params = getattr(txrcv, 'TxWFParameters', None)
        if tx_params is None or len(tx_params) == 0:
            return None
        wf = tx_params[0]
        return CPHDTxWaveform(
            lfm_rate=(
                abs(float(wf.LFMRate))
                if getattr(wf, 'LFMRate', None) is not None else None
            ),
            pulse_length=(
                float(wf.PulseLength)
                if getattr(wf, 'PulseLength', None) is not None
                else None
            ),
            identifier=getattr(wf, 'Identifier', None),
        )

    def _parse_rcv_params_sarpy(self) -> Optional[CPHDRcvParameters]:
        """Extract receive parameters from sarpy metadata."""
        txrcv = getattr(self._sarpy_meta, 'TxRcv', None)
        if txrcv is None:
            return None
        rcv_params = getattr(txrcv, 'RcvParameters', None)
        if rcv_params is None or len(rcv_params) == 0:
            return None
        rcv = rcv_params[0]
        return CPHDRcvParameters(
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
            identifier=getattr(rcv, 'Identifier', None),
        )

    def _load_pvp_sarpy(self, channel: int = 0) -> CPHDPVP:
        """Load per-vector parameters from sarpy for a channel."""
        pvp_raw = self._reader.read_pvp_array(channel)
        meta_pvp = self._sarpy_meta.PVP
        pvp_dict = meta_pvp.to_dict() if meta_pvp else {}
        pvp_keys = set(pvp_dict.keys())

        def _get_field(name: str) -> Optional[np.ndarray]:
            """Extract a PVP field by searching pvp_dict keys."""
            if name not in pvp_keys:
                return None
            items = list(pvp_dict.items())
            idx = [i for i, (k, _) in enumerate(items) if k == name]
            if not idx:
                return None
            return np.array([val[idx[0]] for val in pvp_raw])

        return CPHDPVP(
            tx_time=_get_field('TxTime'),
            tx_pos=_get_field('TxPos'),
            tx_vel=_get_field('TxVel'),
            rcv_time=_get_field('RcvTime'),
            rcv_pos=_get_field('RcvPos'),
            rcv_vel=_get_field('RcvVel'),
            srp_pos=_get_field('SRPPos'),
            fx1=_get_field('FX1'),
            fx2=_get_field('FX2'),
            sc0=_get_field('SC0'),
            scss=_get_field('SCSS'),
            signal=_get_field('SIGNAL'),
            a_fdop=_get_field('aFDOP'),
            a_frr1=_get_field('aFRR1'),
            a_frr2=_get_field('aFRR2'),
            amp_sf=_get_field('AmpSF'),
            toa1=_get_field('TOA1'),
            toa2=_get_field('TOA2'),
        )

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
            return self._reader.read_signal(
                ch_id,
                start_vector=row_start,
                stop_vector=row_end,
            )[:, col_start:col_end]
        else:
            return self._reader.read_chip(
                index=channel,
                dim1_range=(row_start, row_end),
                dim2_range=(col_start, col_end),
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
            return self._reader.read_signal(ch_id)
        else:
            return self._reader.read(index=channel)

    def get_shape(self) -> Tuple[int, int]:
        """Get phase history dimensions for first channel.

        Returns
        -------
        Tuple[int, int]
            ``(num_vectors, num_samples)`` for the first channel.
        """
        first_ch = self.metadata.channels[0]
        return (first_ch.num_vectors, first_ch.num_samples)

    def get_dtype(self) -> np.dtype:
        """Get data type.

        Returns
        -------
        np.dtype
            ``complex64`` for CPHD data.
        """
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
