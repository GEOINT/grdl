# -*- coding: utf-8 -*-
"""
Remote Utilities - Shared credential loading, token acquisition, and
streaming download infrastructure for GRDL catalog remote data acquisition.

Provides reusable functions for authenticating with ESA Copernicus Data
Space Ecosystem (CDSE) and NASA Earthdata, and for downloading products
with progress logging and ZIP extraction.

Dependencies
------------
requests

Author
------
Ava Courtney
courtney-ava@zai.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-03-12

Modified
--------
2026-03-12
"""

# Standard library
from pathlib import Path
from typing import Any, Dict, Optional, Union
import json
import logging
import os
import warnings

# Third-party
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    warnings.warn(
        "Requests not available for remote queries. "
        "Install with: pip install requests",
        ImportWarning,
    )

# GRDL internal
from grdl.exceptions import DependencyError, ProcessorError

logger = logging.getLogger(__name__)

# Default credentials file path (repo-agnostic, shared across projects)
_CREDENTIALS_PATH = Path.home() / ".config" / "geoint" / "credentials.json"

# ── CDSE (Copernicus Data Space Ecosystem) ──────────────────────────────
CDSE_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE"
    "/protocol/openid-connect/token"
)
CDSE_CLIENT_ID = "cdse-public"

# ── NASA Earthdata ──────────────────────────────────────────────────────
EARTHDATA_TOKEN_URL = (
    "https://urs.earthdata.nasa.gov/api/users/token"
)


# ── Credential Management ──────────────────────────────────────────────


def load_credentials(
    provider: str = "esa_maap",
    credentials_file: Optional[Union[str, Path]] = None,
) -> Dict[str, str]:
    """Load credentials from the shared geoint config file.

    Reads the provider block from ``~/.config/geoint/credentials.json``
    (or a user-specified path).  Falls back to environment variables if
    the file entry is missing or empty.

    Supported providers and their env-var fallbacks:

    ============== ======================================
    Provider       Environment variables
    ============== ======================================
    esa_maap       ``ESA_MAAP_OFFLINE_TOKEN``
    esa_copernicus ``COPERNICUS_USERNAME``, ``COPERNICUS_PASSWORD``
    nasa_earthdata ``EARTHDATA_USERNAME``, ``EARTHDATA_PASSWORD``
    ============== ======================================

    Parameters
    ----------
    provider : str, default="esa_maap"
        Key in the credentials JSON for the provider block.
    credentials_file : Optional[Union[str, Path]], default=None
        Path to credentials JSON. If None, uses
        ``~/.config/geoint/credentials.json``.

    Returns
    -------
    Dict[str, str]
        Credential fields for the requested provider.

    Raises
    ------
    ValueError
        If credentials are empty or missing for the requested provider.
    """
    cred_path = (
        Path(credentials_file) if credentials_file else _CREDENTIALS_PATH
    )

    if cred_path.exists():
        with open(cred_path, 'r') as f:
            creds = json.load(f)

        block = creds.get(provider, {})

        # Check for non-empty values
        if block and all(v for v in block.values()):
            return dict(block)

    # Fallback to environment variables
    env_map: Dict[str, Dict[str, str]] = {
        "esa_maap": {
            "offline_token": "ESA_MAAP_OFFLINE_TOKEN",
        },
        "esa_copernicus": {
            "username": "COPERNICUS_USERNAME",
            "password": "COPERNICUS_PASSWORD",
        },
        "nasa_earthdata": {
            "username": "EARTHDATA_USERNAME",
            "password": "EARTHDATA_PASSWORD",
        },
    }

    if provider in env_map:
        result: Dict[str, str] = {}
        for key, env_var in env_map[provider].items():
            val = os.environ.get(env_var, "")
            if val:
                result[key] = val
        if result and len(result) == len(env_map[provider]):
            return result

    raise ValueError(
        f"No credentials found for '{provider}'. "
        f"Set them in {_CREDENTIALS_PATH} or via environment variables."
    )


# ── Token Acquisition ──────────────────────────────────────────────────


def _require_requests() -> None:
    """Raise DependencyError if requests is not installed."""
    if not REQUESTS_AVAILABLE:
        raise DependencyError(
            "Requests library required for remote operations. "
            "Install with: pip install requests"
        )


def get_cdse_token(
    credentials_file: Optional[Union[str, Path]] = None,
) -> str:
    """Acquire a short-lived access token from the Copernicus Data Space
    Ecosystem (CDSE) identity provider.

    Uses the ``esa_copernicus`` credential block (username/password).

    Parameters
    ----------
    credentials_file : Optional[Union[str, Path]], default=None
        Path to credentials JSON.

    Returns
    -------
    str
        CDSE bearer access token.

    Raises
    ------
    DependencyError
        If ``requests`` is not installed.
    ProcessorError
        If token exchange fails.
    """
    _require_requests()

    creds = load_credentials("esa_copernicus", credentials_file)

    response = requests.post(
        CDSE_TOKEN_URL,
        data={
            "grant_type": "password",
            "username": creds["username"],
            "password": creds["password"],
            "client_id": CDSE_CLIENT_ID,
        },
        timeout=30,
    )

    if response.status_code != 200:
        raise ProcessorError(
            f"CDSE token exchange failed ({response.status_code}): "
            f"{response.text[:300]}"
        )

    return response.json()["access_token"]


def get_earthdata_token(
    credentials_file: Optional[Union[str, Path]] = None,
) -> str:
    """Acquire a bearer token from NASA Earthdata Login.

    Uses the ``nasa_earthdata`` credential block (username/password) for
    HTTP Basic authentication against the Earthdata token endpoint.

    Parameters
    ----------
    credentials_file : Optional[Union[str, Path]], default=None
        Path to credentials JSON.

    Returns
    -------
    str
        Earthdata bearer token.

    Raises
    ------
    DependencyError
        If ``requests`` is not installed.
    ProcessorError
        If token request fails.
    """
    _require_requests()

    creds = load_credentials("nasa_earthdata", credentials_file)

    response = requests.post(
        EARTHDATA_TOKEN_URL,
        auth=(creds["username"], creds["password"]),
        timeout=30,
    )

    if response.status_code != 200:
        raise ProcessorError(
            f"Earthdata token request failed ({response.status_code}): "
            f"{response.text[:300]}"
        )

    data = response.json()

    # Earthdata returns either a token object or a list of tokens
    if isinstance(data, list):
        if not data:
            raise ProcessorError("Earthdata returned empty token list")
        return data[0]["access_token"]

    return data["access_token"]


# ── Streaming Download ─────────────────────────────────────────────────


def download_file(
    url: str,
    destination: Path,
    filename: str,
    headers: Optional[Dict[str, str]] = None,
    auth: Optional[Any] = None,
    extract: bool = True,
    timeout: int = 600,
) -> Path:
    """Stream-download a file with progress logging and optional extraction.

    Parameters
    ----------
    url : str
        Remote file URL.
    destination : Path
        Directory to save the downloaded file.
    filename : str
        Name for the downloaded file (e.g., ``product_id.zip``).
    headers : Optional[Dict[str, str]], default=None
        HTTP headers (e.g., ``{"Authorization": "Bearer ..."}``.
    auth : Optional[Any], default=None
        ``requests``-compatible auth (tuple, AuthBase, etc.).
    extract : bool, default=True
        If True and the file is a valid ZIP, extract and remove the ZIP.
    timeout : int, default=600
        Request timeout in seconds.

    Returns
    -------
    Path
        Path to the downloaded file (or extracted directory if extracted).

    Raises
    ------
    DependencyError
        If ``requests`` is not installed.
    ProcessorError
        If download fails. Partial files are cleaned up.
    """
    import zipfile

    _require_requests()

    destination.mkdir(parents=True, exist_ok=True)
    file_path = destination / filename

    try:
        logger.info("Downloading %s", filename)

        session = requests.Session()
        if headers:
            session.headers.update(headers)

        response = session.get(
            url, stream=True, auth=auth, timeout=timeout,
        )
        response.raise_for_status()

        total = int(response.headers.get("content-length", 0))
        downloaded = 0
        next_threshold = 25
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    if pct >= next_threshold:
                        logger.debug(
                            "Download progress: %.1f / %.1f MB (%.0f%%)",
                            downloaded / 1e6, total / 1e6, pct,
                        )
                        next_threshold += 25

        logger.info("Downloaded %s", filename)

        # Extract ZIP if applicable
        result_path = file_path
        if extract and zipfile.is_zipfile(file_path):
            extract_dir = destination / file_path.stem
            with zipfile.ZipFile(file_path, "r") as zf:
                zf.extractall(extract_dir)
            file_path.unlink()
            result_path = extract_dir
            logger.info("Extracted %s to %s", filename, extract_dir.name)

        return result_path

    except requests.RequestException as e:
        if file_path.exists():
            file_path.unlink()
        raise ProcessorError(f"Download failed: {e}") from e
