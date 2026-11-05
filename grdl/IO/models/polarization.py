"""Pure polarization descriptor, channel-index and cube-layout resolution.

Indices address the current cube, not ancestral ``source_indices``. Unlabelled
raw cubes use only the canonical 2/3/4-band orders. Packed C2 is a distinct
representation and is never interpreted as raw scattering.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from numbers import Integral

from grdl.IO.models.base import ChannelMetadata

_BASES = ("HV", "XY", "RL")
_SLOTS = ("co_pol_a", "cross_pol", "cross_pol_2", "co_pol_b")
_PACKED = ("C11", "C12_real", "C12_imag", "C22")


def channel_axis(shape: tuple[int, ...], metadata: object) -> int:
    """Return 0 (CYX) or 2 (YXC), requiring unique metadata consistency.

    Explicit layout still validates bands, descriptor count and spatial shape.
    Consumers with explicit channel arrays should bypass this function.
    """
    if len(shape) != 3:
        raise ValueError(f"Expected 3D channel cube, got shape {shape}")
    order = getattr(metadata, "axis_order", None)
    if order not in (None, "", "CYX", "YXC"):
        raise ValueError(f"Channel input requires CYX or YXC, got {order!r}")
    descriptors = getattr(metadata, "channel_metadata", None)
    counts = [getattr(metadata, "bands", None)]
    if descriptors:
        counts.append(len(descriptors))
    rows, cols = getattr(metadata, "rows", None), getattr(metadata, "cols", None)
    candidates = []
    for axis in (0, 2):
        spatial = shape[1:] if axis == 0 else shape[:2]
        if order and axis != (0 if order == "CYX" else 2):
            continue
        if any(count is not None and shape[axis] != count for count in counts):
            continue
        if rows is not None and spatial[0] != rows:
            continue
        if cols is not None and spatial[1] != cols:
            continue
        candidates.append(axis)
    if len(candidates) != 1:
        raise ValueError(f"Ambiguous or inconsistent channel layout for shape {shape}")
    return candidates[0]


def _single(value: str | None) -> str | None:
    if not value:
        return None
    token = value.upper().strip()
    return {"RHC": "R", "LHC": "L"}.get(token, token)


def _token(value: str | None) -> str | None:
    if not value:
        return None
    text = value.upper().replace("RHC", "R").replace("LHC", "L")
    compact = re.sub(r"[^A-Z0-9]", "", text)
    tokens = {first + second for basis in _BASES for first in basis for second in basis}
    if compact in tokens or compact in tuple("HVXYRL"):
        return compact
    if compact.startswith("S") and compact[1:] in tokens:
        return compact[1:]
    matches = set(re.findall(r"(?<![A-Z])([HVXYRL]{2})(?![A-Z])", text)) & tokens
    if len(matches) > 1:
        raise ValueError(f"Conflicting polarization tokens in {value!r}")
    return next(iter(matches), None)


def _role(value: str | None) -> str | None:
    token = re.sub(r"[^a-z0-9]", "", (value or "").lower())
    return {
        "copola": "co_pol_a",
        "copolb": "co_pol_b",
        "crosspol2": "cross_pol_2",
        "copol": "co",
        "parallelpol": "co",
        "crosspol": "cross_pol",
        "xpol": "cross_pol",
    }.get(token)


def _packed_token(value: str | None) -> str | None:
    token = re.sub(r"[^a-z0-9]", "", (value or "").lower())
    return {
        "c11": "C11",
        "c22": "C22",
        "c12real": "C12_real",
        "rec12": "C12_real",
        "realc12": "C12_real",
        "c12imag": "C12_imag",
        "imc12": "C12_imag",
        "imagc12": "C12_imag",
    }.get(token)


def _descriptor(channel: ChannelMetadata) -> tuple[str | None, str | None]:
    extras = channel.extras or {}
    matrix_name = re.match(r"^[CT][1-4][1-4](?:$|[^0-9])", channel.name.upper())
    if (
        _packed_token(channel.name)
        or matrix_name
        or channel.role in ("covariance", "coherency")
        or extras.get("matrix_element") is not None
    ):
        raise ValueError("Packed matrix descriptors are not raw scattering channels")
    explicit = _token(channel.polarization)
    if channel.polarization and explicit is None:
        raise ValueError(f"Unsupported polarization {channel.polarization!r}")
    tx, rx = _single(channel.tx_polarization), _single(channel.rcv_polarization)
    if bool(tx) != bool(rx):
        raise ValueError("Incomplete transmit/receive polarization descriptor")
    txrx = _token(tx + rx) if tx and rx else None
    if tx and rx and txrx is None:
        raise ValueError(f"Unsupported transmit/receive polarization {tx!r}/{rx!r}")
    name = _token(channel.name)
    evidence = {value for value in (explicit, txrx, name) if value}
    if len(evidence) > 1:
        raise ValueError(f"Conflicting polarization descriptors for {channel.name!r}")
    token = explicit or txrx or name
    roles = {
        role
        for value in (
            channel.role,
            extras.get("polarimetric_role"),
            extras.get("pol_role"),
            extras.get("role"),
        )
        if (role := _role(value)) is not None
    }
    specific = roles - {"co"}
    if len(specific) > 1 or ("co" in roles and any("cross" in role for role in roles)):
        raise ValueError(f"Conflicting polarization roles for {channel.name!r}")
    role = next(iter(specific or roles), None)
    if token and role:
        if len(token) != 2:
            raise ValueError("Single-field polarization cannot describe a co/cross role")
        is_co = token[0] == token[1]
        if is_co != (role in ("co", "co_pol_a", "co_pol_b")):
            raise ValueError(f"Conflicting polarization role for {channel.name!r}")
    return token, role


def _descriptors(count: int, metadata: object) -> Sequence[ChannelMetadata]:
    channels = getattr(metadata, "channel_metadata", None) or []
    if channels and len(channels) != count:
        raise ValueError("Channel descriptor count does not match cube bands")
    return channels


def resolve_raw_indices(
    count: int,
    metadata: object,
    requirement: str = "matrix",
) -> tuple[int, ...]:
    """Resolve ordered raw indices for matrix, dual, full, or stokes.

    Matrix returns [co_a,cross] / [co_a,cross,co_b] /
    [co_a,cross,cross_2,co_b]. Full requires four actual channels. Dual
    selects co/cross; Stokes selects co/co on >=3 bands, else its two fields.
    Labelled surplus bands may be ignored, but meaningful duplicates,
    conflicts and mixed bases are rejected, including unused descriptors.
    """
    if requirement not in ("matrix", "dual", "full", "stokes"):
        raise ValueError(f"Unknown raw channel requirement {requirement!r}")
    descriptors = [_descriptor(channel) for channel in _descriptors(count, metadata)]
    if not any(token or role for token, role in descriptors):
        if count not in (2, 3, 4) or (requirement == "full" and count != 4):
            raise ValueError("Unlabelled raw channels require canonical 2/3/4 band count")
        if requirement == "dual":
            return (0, 1)
        if requirement == "stokes":
            return (0, count - 1) if count >= 3 else (0, 1)
        return tuple(range(count))
    tokens = [token for token, _ in descriptors if token]
    if len(set(tokens)) != len(tokens):
        raise ValueError("Duplicate polarization descriptors")
    bases = {basis for token in tokens for basis in _BASES if set(token) <= set(basis)}
    if len(bases) > 1:
        raise ValueError("Mixed polarization bases")
    basis = next(iter(bases), None)
    if any(len(token) == 1 for token in tokens):
        if (
            requirement != "stokes"
            or count != 2
            or len(tokens) != 2
            or any(len(token) != 1 for token in tokens)
            or basis is None
        ):
            raise ValueError("Single-field polarizations require a two-field Stokes pair")
        return tuple(
            index
            for letter in basis
            for index, (token, _) in enumerate(descriptors)
            if letter == token
        )
    slots: dict[str, int] = {}
    generic_co = []
    for index, (token, role) in enumerate(descriptors):
        slot = None
        if token and basis:
            slot = _SLOTS[
                (basis[0] + basis[0], basis, basis[::-1], basis[1] + basis[1]).index(token)
            ]
        if role in _SLOTS:
            if slot and slot != role:
                raise ValueError("Conflicting canonical polarization and named role")
            slot = role
        elif role == "co" and slot is None:
            generic_co.append(index)
        if slot:
            if slot in slots:
                raise ValueError(f"Duplicate polarization role {slot}")
            slots[slot] = index
    for index in generic_co:
        slot = next((name for name in ("co_pol_a", "co_pol_b") if name not in slots), None)
        if slot is None:
            raise ValueError("Duplicate co-polarization roles")
        slots[slot] = index
    co = [slots[name] for name in ("co_pol_a", "co_pol_b") if name in slots]
    cross = [slots[name] for name in ("cross_pol", "cross_pol_2") if name in slots]
    if requirement == "full":
        required: tuple[str, ...] = _SLOTS
    elif requirement == "dual":
        if co and cross:
            return (co[0], cross[0])
        required = ("co_pol_a", "cross_pol")
    elif requirement == "stokes":
        if count >= 3:
            required = ("co_pol_a", "co_pol_b")
        elif len(tokens) == 2 and basis is not None and not any(role for _, role in descriptors):
            order = (basis[0] * 2, basis, basis[::-1], basis[1] * 2)
            return tuple(
                index
                for token in order
                for index, (value, _) in enumerate(descriptors)
                if token == value
            )
        elif co and cross:
            return (co[0], cross[0])
        else:
            required = ("co_pol_a", "co_pol_b")
    else:
        if len(co) == 2 and cross:
            return (co[0], *cross, co[1])
        if len(co) == 1 and len(cross) == 1:
            return (co[0], cross[0])
        required = ("co_pol_a", "cross_pol", "co_pol_b")
    if any(name not in slots for name in required):
        raise ValueError(f"Missing raw channels for {requirement}: required {required}")
    return tuple(slots[name] for name in required)


def resolve_packed_c2_indices(count: int, metadata: object) -> tuple[int, ...]:
    """Resolve compact-pol C2 band order, not matrix values or matrix basis.

    Compact-pol decompositions consume C11, Re(C12), Im(C12), C22 rather
    than raw scattering channels. Metadata-based ordering keeps shuffled
    covariance bands meaningful and prevents interpreting HH/HV/VH/VV as
    packed C2 merely because both representations have four bands.

    This resolver is C2-specific because that is the compact-pol input
    contract. Generic pack_matrix/unpack_matrix already handle canonical
    C2/T2, C3/T3 and C4/T4 storage; they do not resolve metadata labels.
    If other matrix consumers need labelled-band reordering, generalize
    this resolver with an explicit matrix_type rather than duplicating
    it for each C/T dimension. No covariance/coherency conversion occurs.

    Returns indices in C11, Re(C12), Im(C12), C22 order. Unlabelled input
    must have exactly four bands in that canonical order.
    """
    channels = _descriptors(count, metadata)
    mapping = {}
    meaningful = False
    for index, channel in enumerate(channels):
        token = _packed_token(channel.name)
        if channel.role == "coherency" or (
            token is None and re.match(r"^[CT][1-4][1-4](?:$|[^0-9])", channel.name.upper())
        ):
            raise ValueError(f"Unsupported packed C2 descriptor {channel.name!r}")
        if channel.polarization or channel.tx_polarization or channel.rcv_polarization:
            raise ValueError("Raw scattering descriptors are not packed C2")
        if _token(channel.name) or _role(channel.role):
            raise ValueError("Raw scattering descriptors are not packed C2")
        meaningful |= token is not None
        if token:
            if token in mapping:
                raise ValueError(f"Duplicate packed C2 descriptor {token}")
            mapping[token] = index
    if not meaningful:
        if count != 4:
            raise ValueError("Unlabelled packed C2 requires exactly four bands")
        return (0, 1, 2, 3)
    if any(name not in mapping for name in _PACKED):
        raise ValueError("Missing packed C2 descriptors")
    return tuple(mapping[name] for name in _PACKED)


def resolve_quad_pol_mapping(
    count: int,
    metadata: object,
    overrides: Mapping[str, int | None] | None = None,
) -> dict[str, int]:
    """Return exact HH/HV/VH/VV indices; validated 1-based overrides win.

    A complete override bypasses descriptor interpretation, not layout checks.
    Partial overrides never fill an incomplete labelled mapping positionally.
    """
    keys = ("HH", "HV", "VH", "VV")
    selected: dict[str, int] = {}
    for key, value in (overrides or {}).items():
        if key not in keys:
            raise ValueError(f"Unknown polarization override {key!r}")
        if value is not None:
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or not 1 <= value <= count
            ):
                raise ValueError(f"{key} band override out of range or not an integer: {value}")
            selected[key] = int(value) - 1
    if len(selected) != 4:
        channels = _descriptors(count, metadata)
        mapping = {}
        meaningful = False
        for index, channel in enumerate(channels):
            if index in selected.values():
                continue
            token, role = _descriptor(channel)
            meaningful |= bool(token or role)
            if token and token not in selected:
                if token in mapping:
                    raise ValueError(f"Duplicate polarization descriptor {token}")
                mapping[token] = index
        bases = {basis for token in mapping for basis in _BASES if set(token) <= set(basis)}
        if len(bases) > 1:
            raise ValueError("Mixed polarization bases")
        if not meaningful and not any(_descriptor(channel) != (None, None) for channel in channels):
            if count != 4:
                raise ValueError("Unlabelled quad-pol requires exactly four bands")
            mapping = dict(zip(keys, range(4), strict=True))
        selected = {**mapping, **selected}
    if any(key not in selected for key in keys):
        raise ValueError("Missing exact HH/HV/VH/VV channels; no labelled positional fallback")
    result = {key: selected[key] for key in keys}
    if len(set(result.values())) != 4:
        raise ValueError(f"Duplicate band indices in mapping: {result}")
    return result
