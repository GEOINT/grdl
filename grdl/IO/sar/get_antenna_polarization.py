"""
Antenna Polarization Computation from SICD Metadata
"""

from pathlib import Path
from typing import Any, cast

import numpy as np

from grdl.IO.models.sicd import SICDMetadata


def get_antenna_polarization(
    meta: SICDMetadata | dict[str, Any],
    ap_perceent: float = 50.0,
    ang_tol: float = 5.0,
    save_units_path: str | None = None,
) -> tuple[str | list[str], dict[str, float | list[float]]]:
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
    timeline = _metadata_field(meta, "timeline")
    collect_duration = _metadata_field(timeline, "collect_duration")
    curr_time = collect_duration * (ap_perceent / 100.0)

    # Evaluate polynomials at the computed collect time to get antenna unit vectors (ECF)
    x_unit_ecf = _eval_vector_poly(x_coefs, curr_time)
    y_unit_ecf = _eval_vector_poly(y_coefs, curr_time)

    # Get the scene center position (SCP) at current time
    scp_ecf = _get_scp_position(meta, curr_time)

    # Convert SCP to geodetic coordinates
    scp_lla = ecf_to_geodetic(scp_ecf)

    # Roate antenna unit vectors to SEZ frame
    # ecf_to_geodetic already returns degrees so use directly
    x_unit_sez = ecf_to_sez(x_unit_ecf, scp_lla[0], scp_lla[1])
    y_unit_sez = ecf_to_sez(y_unit_ecf, scp_lla[0], scp_lla[1])

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
    radar_collection = _metadata_field(meta, "radar_collection")
    tx_pol_raw = _metadata_field(radar_collection, "tx_polarization")
    tx_pol, tx_pol_ang = _map_polarization(tx_pol_raw, x_ang, y_ang, ang_tol)

    # Map receive polarization(s)
    rx_pol_raw = _resolve_rx_pol_raw(meta)
    pol_str: str | list[str]
    pol_angles: dict[str, float | list[float]]
    if tx_pol == "S":
        tx_pol_ang = _adjust_slant_angle(tx_pol_ang)

    if isinstance(rx_pol_raw, list):
        # Mulitple receive channels
        rx_pol: list[str] = []
        rx_pol_ang: list[float] = []
        for raw_polarization in rx_pol_raw:
            mapped_pol, mapped_ang = _map_polarization(
                _rx_char(raw_polarization), x_ang, y_ang, ang_tol
            )
            if mapped_pol == "S":
                mapped_ang = _adjust_slant_angle(mapped_ang)
            # Build full polarization string
            full_pol = tx_pol + mapped_pol
            rx_pol.append(full_pol)
            rx_pol_ang.append(mapped_ang)

        pol_str = rx_pol
        pol_angles = {
            "tx": tx_pol_ang,
            "rx": rx_pol_ang,
        }

    else:
        # Single receive channel
        rx_label, rx_angle = _map_polarization(_rx_char(rx_pol_raw), x_ang, y_ang, ang_tol)
        if rx_label == "S":
            rx_angle = _adjust_slant_angle(rx_angle)

        pol_str = tx_pol + rx_label
        pol_angles = {
            "tx": tx_pol_ang,
            "rx": rx_angle,
        }

    # Save unit vectors if requested
    if save_units_path is not None:
        # TODO: Implement saving of unit vectors
        print("Saving unit vectors not yet implemented")
        pass

    return pol_str, pol_angles


def get_antenna_polarization_from_file(
    filepath: str | Path,
    ap_perceent: float = 50.0,
    ang_tol: float = 5.0,
    save_units_path: str | None = None,
) -> tuple[
    str | list[str],
    dict[str, float | list[float]],
    float | None | list[float | None],
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
        meta = cast(SICDMetadata, reader.metadata)

    pol_str, pol_angles = get_antenna_polarization(meta, ap_perceent, ang_tol, save_units_path)

    rx_pol_raw = _resolve_rx_pol_raw(meta)
    rx_angles = pol_angles["rx"]
    rotation_angle: float | None | list[float | None]
    if isinstance(rx_pol_raw, list):
        if not isinstance(rx_angles, list):
            raise ValueError("Multiple receive channels require a list of angles")
        rotation_angle = [
            _rotation_angle(raw, ang) for raw, ang in zip(rx_pol_raw, rx_angles, strict=True)
        ]
    else:
        if isinstance(rx_angles, list):
            raise ValueError("A single receive channel requires a scalar angle")
        rotation_angle = _rotation_angle(rx_pol_raw, rx_angles)

    return pol_str, pol_angles, rotation_angle


def _resolve_rx_pol_raw(meta: SICDMetadata | dict[str, Any]) -> str | list[str]:
    """Return the raw "TX:RX" polarization string(s) of the processed image."""
    image_formation = _metadata_field(meta, "image_formation")
    processed = _metadata_field(image_formation, "tx_rcv_polarization_proc", required=False)
    if processed is not None:
        if not isinstance(processed, str) or not processed:
            raise ValueError("Processed polarization must be a nonempty string")
        return processed

    radar_collection = _metadata_field(meta, "radar_collection")
    rcv_channels = _metadata_field(radar_collection, "rcv_channels")
    rcv_chan_proc = _metadata_field(image_formation, "rcv_chan_proc")
    chan_indices = _metadata_field(rcv_chan_proc, "chan_indices", required=False) or [1]
    polarizations = []
    for index in chan_indices:
        if not 1 <= index <= len(rcv_channels):
            raise ValueError(f"Receive channel index out of range: {index}")
        polarization = _metadata_field(rcv_channels[index - 1], "tx_rcv_polarization")
        if not isinstance(polarization, str) or not polarization:
            raise ValueError("Receive polarization must be a nonempty string")
        polarizations.append(polarization)
    return polarizations if len(polarizations) > 1 else polarizations[0]


def _rx_char(pol_str: str) -> str:
    """Return the Rx part of a "TX:RX" polarization string."""
    if ":" in pol_str:
        return pol_str.split(":")[1]
    return pol_str[-1]  # Fallback to last character


def _rotation_angle(rx_pol_raw: str, rx_ang: float) -> float | None:
    """Rotation of an 'X' or 'Y' receive polarization from its nominal axis."""
    rx_char = _rx_char(rx_pol_raw).upper()
    if rx_char == "X":
        return rx_ang - 0.0
    if rx_char == "Y":
        return rx_ang - 90.0
    return None


def _metadata_field(value: object, name: str, *, required: bool = True) -> Any:
    """Read a metadata field from a dictionary or dataclass, validating required values."""
    field = value.get(name) if isinstance(value, dict) else getattr(value, name, None)
    if required and field is None:
        raise ValueError(f"Missing required metadata field: {name}")
    return field


def _validate_metadata(meta: SICDMetadata | dict[str, Any]) -> None:
    """Validate required metadata fields in the given dictionary.

    Raises:
        ValueError: If any required metadata field is missing.
    """
    required_paths = [
        ("timeline", "collect_duration"),
        ("position", "arp_poly"),
        ("geo_data", "scp", "ecf"),
        ("radar_collection", "tx_polarization"),
        ("radar_collection", "rcv_channels"),
        ("image_formation", "rcv_chan_proc"),
    ]

    antenna = _metadata_field(meta, "antenna")
    if (
        _metadata_field(antenna, "tx", required=False) is None
        and _metadata_field(antenna, "two_way", required=False) is None
    ):
        raise ValueError("Missing required antenna subfields: 'tx' or 'two_way'")

    # Check required paths
    for path in required_paths:
        current: Any = meta
        for key in path:
            current = _metadata_field(current, key)


def _get_poly_coefs(poly_component: Any) -> np.ndarray:
    """Return polynomial coefficients in ascending order."""
    coefficients = _metadata_field(poly_component, "coefs", required=False)
    if coefficients is not None:
        return np.asarray(coefficients)
    if hasattr(poly_component, "__iter__"):
        return np.asarray(poly_component)
    raise ValueError("Unsupported polynomial component type")


def _extract_antenna_polynomials(
    meta: SICDMetadata | dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the antenna unit vector polynomials for the x and y axes.

    Args:
        meta (dict): Metadata dictionary containing antenna information.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Coefficients for the x and y axis polynomials.
    """

    # Get antenna object (dict or attribute access)
    antenna_obj = _metadata_field(meta, "antenna")

    # Try Tx first, then TwoWay
    antenna = _metadata_field(antenna_obj, "tx", required=False)
    if antenna is None:
        antenna = _metadata_field(antenna_obj, "two_way")

    # Extract X-axis polynomial
    # In GRDL: antenna.x_asis_poly is XYZPoly with x, y, z as Poly1D objects
    # Poly1D.coefs is already in ascending order (matches numpy convenction)
    x_axis = _metadata_field(antenna, "x_axis_poly")
    x_coefs = np.array(
        [
            _get_poly_coefs(_metadata_field(x_axis, "x")),
            _get_poly_coefs(_metadata_field(x_axis, "y")),
            _get_poly_coefs(_metadata_field(x_axis, "z")),
        ]
    )

    # Extract Y-axis polynomial
    y_axis = _metadata_field(antenna, "y_axis_poly")
    y_coefs = np.array(
        [
            _get_poly_coefs(_metadata_field(y_axis, "x")),
            _get_poly_coefs(_metadata_field(y_axis, "y")),
            _get_poly_coefs(_metadata_field(y_axis, "z")),
        ]
    )

    return x_coefs, y_coefs


def _eval_vector_poly(coefs: np.ndarray, time: float) -> np.ndarray:
    """Evaluate the vector polynomial at a given time.

    Args:
        coefs (np.ndarray): Array of polynomial coefficients for x, y, z components.
        time (float): Time at which to evaluate the polynomial.

    Returns:
        np.ndarray: Evaluated vector at the given time.
    """
    return np.array(
        [
            np.polyval(coefs[0][::-1], time),
            np.polyval(coefs[1][::-1], time),
            np.polyval(coefs[2][::-1], time),
        ]
    )


def _get_scp_position(meta: SICDMetadata | dict[str, Any], time: float) -> np.ndarray:
    """
    Get the scene center position at a given time.
    Tries position.grp_poly first, falls back to geo_data.scp.ecf.
    """
    # Get position object
    position = _metadata_field(meta, "position")
    grp_poly = _metadata_field(position, "grp_poly", required=False)

    if grp_poly is not None:
        # Evaluate the GRP polynomial
        # Extract coefficients and reverse for np.polyval (expects descending order)
        x_poly = _metadata_field(grp_poly, "x")
        y_poly = _metadata_field(grp_poly, "y")
        z_poly = _metadata_field(grp_poly, "z")

        x_coefs = _get_poly_coefs(x_poly)[::-1]
        y_coefs = _get_poly_coefs(y_poly)[::-1]
        z_coefs = _get_poly_coefs(z_poly)[::-1]

        return np.array(
            [np.polyval(x_coefs, time), np.polyval(y_coefs, time), np.polyval(z_coefs, time)]
        )
    else:
        # Fallback to static SCP
        geo_data = _metadata_field(meta, "geo_data")
        scp = _metadata_field(geo_data, "scp")
        ecf = _metadata_field(scp, "ecf")
        return np.array(
            [
                _metadata_field(ecf, "x"),
                _metadata_field(ecf, "y"),
                _metadata_field(ecf, "z"),
            ]
        )


def _get_arp_position(meta: SICDMetadata | dict[str, Any], time: float) -> np.ndarray:
    """
    Get the antenna reference point (ARP) position at a given time.

    """
    # Get position object
    position = _metadata_field(meta, "position")
    arp_poly = _metadata_field(position, "arp_poly")

    # Extract coefficients and reverse for np.polyval (expects descending order)
    x_poly = _metadata_field(arp_poly, "x")
    y_poly = _metadata_field(arp_poly, "y")
    z_poly = _metadata_field(arp_poly, "z")

    x_coefs = _get_poly_coefs(x_poly)[::-1]
    y_coefs = _get_poly_coefs(y_poly)[::-1]
    z_coefs = _get_poly_coefs(z_poly)[::-1]

    return np.array(
        [np.polyval(x_coefs, time), np.polyval(y_coefs, time), np.polyval(z_coefs, time)]
    )


def _map_polarization(
    pol_char: str, x_ang: float, y_ang: float, ang_tol: float
) -> tuple[str, float]:
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

    if pol_char == "H":
        # H maps to x-axis (0 from horizontal)
        return "H", x_ang

    elif pol_char == "V":
        # V maps to y-axis (0 from vertical)
        return "V", y_ang

    elif pol_char == "X":
        # Check if within tolerance of cardinal angles
        if abs(x_ang) < ang_tol:
            return "H", x_ang
        elif abs(90 - x_ang) < ang_tol:
            return "V", x_ang
        elif abs(45 - x_ang) < ang_tol:
            return "S", x_ang
        return "X", x_ang

    elif pol_char == "Y":
        # Check if within tolerance of cardinal angles
        if abs(y_ang) < ang_tol:
            return "H", y_ang
        elif abs(90 - y_ang) < ang_tol:
            return "V", y_ang
        elif abs(45 - y_ang) < ang_tol:
            return "S", y_ang
        return "Y", y_ang
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
