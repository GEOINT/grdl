"""
Antenna Polarization Computation from SICD Metadata
"""

from pathlib import Path
from typing import Union, Optional, Tuple, Dict, List
import numpy as np

def get_antenna_polarization(
    meta: dict,
    ap_perceent: float = 50.0,
    ang_tol: float = 5.0,
    save_units_path: Optional[str] = None,
) -> Tuple[Union[str, List[str]], Dict[str, Union[float, List[float]]]]: 
    """
    Compute the antenna polarization from SICD metadata.

    Parameters
    ----------
    meta : dict
        SICD metadata dictionary.
    ap_perceent : float, optional
        Antenna polarization percentage threshold, by default 50.0.
    ang_tol : float, optional
        Angular tolerance in degrees, by default 5.0.
    save_units_path : Optional[str], optional
        Path to save computed units, by default None.

    Returns
    -------
    Tuple[Union[str, List[str]], Dict[str, Union[float, List[float]]]]
        A tuple containing the computed antenna polarization(s) and a dictionary
        of associated metrics.
    """

    # Import here to avoid circular dependencies if needed
    from sarpy.geometry.geocoords import ecf_to_geodetic
    from grdl.geolocation.ecf_sez import ecf_to_sez

    # Validate metadata structure
    _validate_metadata(meta)

    # TODO: Implement the actual computation of antenna polarization based on the mapped transmit and receive polarizations.
    # This requires sarpy.geometry.geoccords import ecf_to_geodetic and ...geolocation.ecf_sez import ecf_to_sez
    # See high side implmentation
    # Temporarily hard code x_ang and y_ang
    # x_ang = 0.0
    # y_ang = 90.0 + x_ang

    # Get antenna unit vector polynomials
    x_coefs, y_coefs = _extract_antenna_polynomials(meta)

    # Compute collect time based on aperture percentage
    timeline = meta.timeline if hasattr(meta, 'timeline') else meta['timeline']
    collect_duration = timeline.collect_duration if hasattr(timeline, 'collect_duration') else timeline['collect_duration']
    curr_time = collect_duration * (ap_perceent / 100.0)

    # Evaluate polynomials at the computed collect time to get antenna unit vectors (ECF)
    x_unit_ecf = _eval_vector_poly(x_coefs, curr_time)
    y_unit_ecf = _eval_vector_poly(y_coefs, curr_time)

    # Get the scene center position (SCP) at current time
    scp_ecf = _get_scp_ecf(meta, curr_time)

    # Get the antenna reference point (ARP) at the current time
    arp_ecf = _get_arp_ecf(meta, curr_time)

    # Convert SCP to geodetic coordinates
    scp_lla = ecf_to_geodetic(scp_ecf)

    # Roate antenna unit vectors to SEZ frame
    # ecf_to_geodetic already returns degrees so use directly
    x_unit_sez = ecf_to_sez(x_unit_ecf, scp_lla)
    y_unit_sez = ecf_to_sez(y_unit_ecf, scp_lla)

    # Normalize to unit vectors (rotation preserves direction but may change magnitude due to numerical issues)
    x_unit_sez /= np.linalg.norm(x_unit_sez)
    y_unit_sez /= np.linalg.norm(y_unit_sez)
    z_unit_sez = np.cross(x_unit_sez, y_unit_sez)
    z_unit_sez /= np.linalg.norm(z_unit_sez)

    # Compute polarization angles based on SEZ unit vectors
    # Angle from horizontal = arcsin(zenith component of SEZ unit vector)
    x_ang = np.degrees(np.arcsin(z_unit_sez[2]))
    y_ang = 90.0 + x_ang

    # Map transmit polarization
    radar_collection = meta.radar_collection if hasattr(meta, 'radar_collection') else meta['radar_collection']
    tx_pol_raw = radar_collection.tx_polarization if hasattr(radar_collection, 'tx_polarization') else radar_collection['tx_polarization']
    tx_pol, tx_pol_ang = _map_polarization(tx_pol_raw, x_ang, y_ang, ang_tol)

    # Map receive polarization(s)
    rx_pol_raw = _resolve_rx_pol_raw(meta)

    if isinstance(rx_pol_raw, list):
        # Mulitple receive channels
        rx_pol: List[str] = []
        rx_pol_ang: List[float] = []
        for pol_str in rx_pol_raw:
            mapped_pol, mapped_ang = _map_polarization(_rx_char(pol_str), x_ang, y_ang, ang_tol)
            # Build full polarization string
            full_pol = tx_pol + mapped_pol
            rx_pol.append(full_pol)
            rx_pol_ang.append(mapped_ang)

        pol_str = rx_pol
        pol_angles: Dict[str, Union[float, List[float]]] = {
            'tx': tx_pol_ang,
            'rx': rx_pol_ang,
        }

        # Handle slant polarization adjustment
        if tx_pol == 'S':
            pol_angles['tx'] = _adjust_slant_angle(pol_angles['tx'])
        for i, ps in enumerate(pol_str):
            if ps[-1] == 'S':
                pol_angles['rx'][i] = _adjust_slant_angle(pol_angles['rx'][i])

    else:
        # Single receive channel
        rx_pol, rx_pol_ang = _map_polarization(_rx_char(rx_pol_raw), x_ang, y_ang, ang_tol)

        pol_str = tx_pol + rx_pol
        pol_angles: Dict[str, Union[float, List[float]]] = {
            'tx': tx_pol_ang,
            'rx': rx_pol_ang,
        }

        # Handle slant polarization adjustment
        if tx_pol == 'S':
            pol_angles['tx'] = _adjust_slant_angle(pol_angles['tx'])
        if pol_str[-1] == 'S':
            pol_angles['rx'] = _adjust_slant_angle(pol_angles['rx'])

    # Save unit vectors if requested
    if save_units_path is not None:
        # TODO: Implement saving of unit vectors
        print("Saving unit vectors not yet implemented")
        pass

    return pol_str, pol_angles


def get_antenna_polarization_from_file(
    filepath: Union[str, Path],
    ap_perceent: float = 50.0,
    ang_tol: float = 5.0,
    save_units_path: Optional[str] = None,
) -> Tuple[
    Union[str, List[str]],
    Dict[str, Union[float, List[float]]],
    Union[Optional[float], List[Optional[float]]],
]:
    """
    Compute the antenna polarization of a SICD file.

    Reads the metadata from ``filepath`` and calls
    :func:`get_antenna_polarization`.

    Parameters
    ----------
    filepath : str or Path
        Path to a SICD file.
    ap_perceent : float, optional
        Antenna polarization percentage threshold, by default 50.0.
    ang_tol : float, optional
        Angular tolerance in degrees, by default 5.0.
    save_units_path : Optional[str], optional
        Path to save computed units, by default None.

    Returns
    -------
    Tuple[Union[str, List[str]], Dict[str, Union[float, List[float]]], Union[Optional[float], List[Optional[float]]]]
        The polarization(s) and angles from :func:`get_antenna_polarization`,
        and the rotation angle in degrees. The rotation angle is
        ``rx_ang - 0.0`` when the Rx character of ``TxRcvPolarization``
        (format "TX:RX") is 'X', ``rx_ang - 90.0`` when it is 'Y', and None
        otherwise. For multiple receive channels it is a list with one entry
        per channel.
    """
    from grdl.IO.sar.sicd import SICDReader

    with SICDReader(filepath) as reader:
        meta = reader.metadata

    pol_str, pol_angles = get_antenna_polarization(
        meta, ap_perceent, ang_tol, save_units_path
    )

    rx_pol_raw = _resolve_rx_pol_raw(meta)
    if isinstance(rx_pol_raw, list):
        rotation_angle = [
            _rotation_angle(raw, ang)
            for raw, ang in zip(rx_pol_raw, pol_angles['rx'])
        ]
    else:
        rotation_angle = _rotation_angle(rx_pol_raw, pol_angles['rx'])

    return pol_str, pol_angles, rotation_angle


def _resolve_rx_pol_raw(meta: dict) -> Union[str, List[str]]:
    """Return the raw "TX:RX" polarization string(s) of the processed image."""
    radar_collection = meta.radar_collection if hasattr(meta, 'radar_collection') else meta['radar_collection']

    # Prefer ImageFormation/TxRcvPolarizationProc (processed polarzaiton over channel polarization
    image_formation = meta.image_formation if hasattr(meta, "image_formation") else meta["image_formation"]

    # Try to get TxRcvPolarizationProc from metadata (GRDL exploes this as of version 0.7.3+)
    rx_pol_raw = None
    if hasattr(image_formation, 'tx_rcv_polarization_proc'):
        rx_pol_raw = image_formation.tx_rcv_polarization_proc

    # Fall back to channel-based polarization if TxRcvPolarizationProc is not available
    if rx_pol_raw is None:
        rcv_chan_proc = (
            image_formation.rcv_chan_proc
            if hasattr(image_formation, "rcv_chan_proc")
            else image_formation["rcv_chan_proc"]
        )
        chan_indices = (
            rcv_chan_proc.chan_indices
            if hasattr(rcv_chan_proc, "chan_indices")
            else rcv_chan_proc["chan_indices"]
        )

        # Get receive channels from radaqr_collection
        rcv_channels = (
            radar_collection.rcv_channels
            if hasattr(radar_collection, "rcv_channels")
            else radar_collection["rcv_channels"]
        )

        # Extract polarization(s) based on channel indices
        if chan_indices and len(chan_indices) > 1:
            # Multiple channels (e.g., quad-pol)
            rx_pol_raw = []
            for idx in chan_indices:
                chan = rcv_channels[idx - 1]  # Channel indices are 1-based
                pol = chan.tx_rcv_polarization if hasattr(chan, "tx_rcv_polarization") else chan["tx_rcv_polarization"]
                rx_pol_raw.append(pol)
        else:
            # Single channel
            idx = chan_indices[0] if chan_indices else 1  # Default to the first channel if chan_indices is empty
            chan = rcv_channels[idx - 1]  # Channel indices are 1-based
            rx_pol_raw = (
                chan.tx_rcv_polarization if hasattr(chan, "tx_rcv_polarization") else chan["tx_rcv_polarization"]
            )

    return rx_pol_raw


def _rx_char(pol_str: str) -> str:
    """Return the Rx part of a "TX:RX" polarization string."""
    if ':' in pol_str:
        return pol_str.split(':')[1]
    return pol_str[-1] # Fallback to last character


def _rotation_angle(rx_pol_raw: str, rx_ang: float) -> Optional[float]:
    """Rotation of an 'X' or 'Y' receive polarization from its nominal axis."""
    rx_char = _rx_char(rx_pol_raw).upper()
    if rx_char == 'X':
        return rx_ang - 0.0
    if rx_char == 'Y':
        return rx_ang - 90.0
    return None

def _validate_metadata(meta: dict) -> None:
    """Validate required metadata fields in the given dictionary.

    Raises:
        ValueError: If any required metadata field is missing.
    """
    required_paths = [
        ('timeline', 'collect_duration'),
        ('position', 'arp_poly'),
        ('geo_data', 'scp', 'ecf'),
        ('radar_collection', 'tx_polarization'),
        ('radar_collection', 'rcv_channels'),
        ('image_formation', 'rcv_chan_proc'),
    ]

    # Check for antenna data - handle both dict and object attribute access
    try:
        antenna = meta['antenna'] if 'antenna' in meta else getattr(meta, 'antenna', None)
        if antenna is None:
            raise ValueError("Missing required metadata field: 'antenna'")

        has_tx = hasattr(antenna, 'tx') and antenna.tx is not None
        has_two_way = hasattr(antenna, 'two_way') and antenna.two_way is not None

        if not (has_tx or has_two_way):
            raise ValueError("Missing required antenna subfields: 'tx' or 'two_way'")
        
    except (KeyError, AttributeError) as e:
        raise ValueError("Error accessing antenna.tx or antenna.two_way metadata field") from e

    # Check required paths
    for path in required_paths:
        current = meta
        for key in path:
            if isinstance(current, dict):
                if key not in current:
                    raise ValueError(f"Missing required metadata field: {'.'.join(path)}")
                current = current[key]
            else:
                if not hasattr(current, key):
                    raise ValueError(f"Missing required metadata field: {'.'.join(path)}")
                current = getattr(current, key)


def _extract_antenna_polynomials(meta: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Extract the antenna unit vector polynomials for the x and y axes.

    Args:
        meta (dict): Metadata dictionary containing antenna information.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Coefficients for the x and y axis polynomials.
    """

    # Get antenna object (dict or attribute access)
    if hasattr(meta, '__getitem__'):
        antenna_obj = getattr(meta, 'antenna', None)
    else:
        antenna_obj = meta.antenna

    # Try Tx first, then TwoWay
    if hasattr(antenna_obj, 'tx') and antenna_obj.tx is not None:
        antenna = antenna_obj.tx
    elif hasattr(antenna_obj, 'two_way') and antenna_obj.two_way is not None:
        antenna = antenna_obj.two_way
    else:
        raise ValueError("Missing required antenna subfields: 'tx' or 'two_way'")
    
    def _get_poly_coefs(poly_component):
        """Extract the coefficients from the Poly1D object or array
        
        Args:
            poly_components: Poly1D object or array containing polynomial coefficients.

        Returns:
            np.ndarray: Array of polynomial coefficients in ascending order.
        """
        if hasattr(poly_component, 'coefs'):
            # GRDL Poly1D object - coefs in ascending order, reverse for np.polyval
            return np.array(poly_component.coefs)[::-1]
        elif hasattr(poly_component, '__iter__'):
            # Array-like object, assume already in ascending order
            return np.asarray(poly_component)
        else:
            raise ValueError("Unsupported polynomial component type")

    # Extract X-axis polynomial
    # In GRDL: antenna.x_asis_poly is XYZPoly with x, y, z as Poly1D objects
    # Poly1D.coefs is already in ascending order (matches numpy convenction)
    x_axis = antenna.x_axis_poly if hasattr(antenna, 'x_axis_poly') else antenna['x_axis_poly']
    x_coefs = np.array([
        _get_poly_coefs(x_axis.x if hasattr(x_axis, 'x') else x_axis['x']),
        _get_poly_coefs(x_axis.y if hasattr(x_axis, 'y') else x_axis['y']),
        _get_poly_coefs(x_axis.z if hasattr(x_axis, 'z') else x_axis['z'])
    ])

    # Extract Y-axis polynomial
    y_axis = antenna.y_axis_poly if hasattr(antenna, 'y_axis_poly') else antenna['y_axis_poly']
    y_coefs = np.array([
        _get_poly_coefs(y_axis.x if hasattr(y_axis, 'x') else y_axis['x']),
        _get_poly_coefs(y_axis.y if hasattr(y_axis, 'y') else y_axis['y']),
        _get_poly_coefs(y_axis.z if hasattr(y_axis, 'z') else y_axis['z'])
    ])

    return x_coefs, y_coefs


def _eval_vector_poly(coefs: np.ndarray, time: float) -> np.ndarray:
    """Evaluate the vector polynomial at a given time.

    Args:
        coefs (np.ndarray): Array of polynomial coefficients for x, y, z components.
        time (float): Time at which to evaluate the polynomial.

    Returns:
        np.ndarray: Evaluated vector at the given time.
    """
    return np.array([
        np.polyval(coefs[0][::-1], time),
        np.polyval(coefs[1][::-1], time),
        np.polyval(coefs[2][::-1], time)
    ])


def _get_scp_position(meta: dict, time: float) -> np.ndarray:
    """
    Get the scene center position at a given time.
    Tries position.grp_poly first, falls back to geo_data.scp.ecf.
    """
    # Get position object
    position = meta.position if hasattr(meta, 'position') else meta['position']

    # Check for GRP polynomial
    has_grp = hasattr(position, 'grp_poly') and position.grp_poly is not None
    if not has_grp and hasattr(position, '__getitem__'):
        has_grp = 'grp_poly' in position

    if has_grp:
        # Evaluate the GRP polynomial
        grp_poly = position.grp_poly if hasattr(position, 'grp_poly') else position['grp_poly']

        # Extract coefficients and reverse for np.polyval (expects descending order)
        x_poly = grp_poly.x if hasattr(grp_poly, 'x') else grp_poly['x']
        y_poly = grp_poly.y if hasattr(grp_poly, 'y') else grp_poly['y']
        z_poly = grp_poly.z if hasattr(grp_poly, 'z') else grp_poly['z']

        x_coefs = x_poly.coefs[::-1] if hasattr(x_poly, 'coefs') else _get_poly_coefs(x_poly)
        y_coefs = y_poly.coefs[::-1] if hasattr(y_poly, 'coefs') else _get_poly_coefs(y_poly)
        z_coefs = z_poly.coefs[::-1] if hasattr(z_poly, 'coefs') else _get_poly_coefs(z_poly)

        return np.array([
            np.polyval(x_coefs, time),
            np.polyval(y_coefs, time),
            np.polyval(z_coefs, time)])
    else:
        # Fallback to static SCP
        geo_data = meta.geo_data if hasattr(meta, 'geo_data') else meta['geo_data']
        scp = geo_data.scp if hasattr(geo_data, 'scp') else geo_data['scp']
        ecf = scp.ecf if hasattr(scp, 'ecf') else scp['ecf']
        return np.array([
            ecf.x if hasattr(ecf, 'x') else ecf['x'],
            ecf.y if hasattr(ecf, 'y') else ecf['y'],
            ecf.z if hasattr(ecf, 'z') else ecf['z']
        ])


def _get_arp_position(meta: dict, time: float) -> np.ndarray:
    """
    Get the antenna reference point (ARP) position at a given time.
    
    """
    # Get position object
    position = meta.position if hasattr(meta, 'position') else meta['position']
    arp_poly = position.arp_poly if hasattr(position, 'arp_poly') else position['arp_poly']

    # Extract coefficients and reverse for np.polyval (expects descending order)
    x_poly = arp_poly.x if hasattr(arp_poly, 'x') else arp_poly['x']
    y_poly = arp_poly.y if hasattr(arp_poly, 'y') else arp_poly['y']
    z_poly = arp_poly.z if hasattr(arp_poly, 'z') else arp_poly['z']

    x_coefs = x_poly.coefs[::-1] if hasattr(x_poly, 'coefs') else np.array(x_poly)
    y_coefs = y_poly.coefs[::-1] if hasattr(y_poly, 'coefs') else np.array(y_poly)
    z_coefs = z_poly.coefs[::-1] if hasattr(z_poly, 'coefs') else np.array(z_poly)

    return np.array([
        np.polyval(x_coefs, time),
        np.polyval(y_coefs, time),
        np.polyval(z_coefs, time)
    ])


def _map_polarization(
        pol_char: str,
        x_ang: float,
        y_ang: float,
        ang_tol: float) -> Tuple[str, float]:
    """Map a polarization character to its corresponding polarization string and angle.

    Args:
        pol_char (str): Polarization character ('H', 'V', 'X', 'Y', etc.).
        x_ang (float): X-axis angle.
        y_ang (float): Y-axis angle.
        ang_tol (float): Angle tolerance.

    Returns:
        tuple[str, float]: Polarization string and corresponding polarization angle in degrees.
    """
    pol_char = pol_char.upper()


    if pol_char == 'H':
        # H maps to x-axis (0 from horizontal)
        return 'H', x_ang
    
    elif pol_char == 'V':
        # V maps to y-axis (0 from vertical)
        return 'V', y_ang
    
    elif pol_char == 'X':
        # Check if within tolerance of cardinal angles
        if abs(x_ang) < ang_tol:
            return 'H', x_ang
        elif abs(90 - x_ang) < ang_tol:
            return 'V', x_ang
        elif abs(45 - x_ang) < ang_tol:
            return 'S', x_ang
        return 'X', x_ang
    
    elif pol_char == 'Y':
        # Check if within tolerance of cardinal angles
        if abs(y_ang) < ang_tol:
            return 'H', y_ang
        elif abs(90 - y_ang) < ang_tol:
            return 'V', y_ang
        elif abs(45 - y_ang) < ang_tol:
            return 'S', y_ang
        return 'Y', y_ang
    else:
        # Unknown polarization character, return as-is
        return pol_char, x_ang

def _adjust_slant_angle(angle: float) -> float:
    """Adjust the slant angle by +/- 45 degrees.

    Args:
        angle (float): Input slant angle in degrees.

    Returns:
        float: Adjusted slant angle in degrees.
    """
    if angle > 45:
        angle -= 45
    else:
        angle += 45
    return angle
