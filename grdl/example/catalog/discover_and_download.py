# -*- coding: utf-8 -*-
"""
BIOMASS Data Discovery and Download Example.

Demonstrates how to use BIOMASSCatalog to search the ESA MAAP STAC catalog
for BIOMASS L1A products, inspect results, and download selected products.

This example searches for imagery over New Norcia, Australia -- the site of
ESA's Biomass Calibration Transponder (BCT). The BCT is a GPS-surveyed
precision radar target used for geolocation validation.

Prerequisites
-------------
1. Create an EO Sign In account at https://eoiam-idp.eo.esa.int/
2. Generate an offline token from the MAAP identity service
   at https://portal.maap.eo.esa.int/
3. Store the token in ~/.config/geoint/credentials.json:

   {
       "esa_copernicus": {
           "username": "",
           "password": ""
       },
       "esa_maap": {
           "offline_token": "<your-offline-token>"
       }
   }

Dependencies
------------
requests

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
2026-01-30

Modified
--------
2026-01-30
"""

# Standard library
import sys
from pathlib import Path

# GRDL
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from grdl.IO import BIOMASSCatalog


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Where to store downloads and the catalog database
DATA_DIR = Path("/Volumes/PRO-G40/SAR_DATA/BIOMASS")

# New Norcia, Australia -- ESA BCT calibration transponder site
# The BCT is at approximately (-31.05, 116.19)
# BIOMASS frames covering this region pass east of the transponder
BBOX_NEW_NORCIA = (115.5, -31.5, 116.8, -30.5)  # (min_lon, min_lat, max_lon, max_lat)

# Product type: S3_SCS__1S = Single-look Complex Slant (single pol processing)
#               S3_SCS__1M = Single-look Complex Slant (multi-pol processing)
# S3 = Swath 3 (tomographic phase); S1 = Swath 1 (interferometric phase)
PRODUCT_TYPE = "S3_SCS__1S"


# ---------------------------------------------------------------------------
# Step 1: Search the MAAP STAC catalog
# ---------------------------------------------------------------------------

def search_products():
    """Search for BIOMASS products near New Norcia."""
    catalog = BIOMASSCatalog(str(DATA_DIR))

    print("=" * 70)
    print("Searching ESA MAAP STAC catalog for BIOMASS L1A products")
    print(f"  Region: New Norcia, Australia")
    print(f"  Bbox:   {BBOX_NEW_NORCIA}")
    print(f"  Type:   {PRODUCT_TYPE}")
    print("=" * 70)
    print()

    products = catalog.query_esa(
        bbox=BBOX_NEW_NORCIA,
        max_results=20,
        product_type=PRODUCT_TYPE,
    )

    print(f"Found {len(products)} products\n")

    # Display results
    for i, p in enumerate(products):
        props = p.get("properties", {})
        geom = p.get("geometry", {})
        coords = geom.get("coordinates", [[]])[0] if geom.get("coordinates") else []

        print(f"[{i}] {p['id']}")
        print(f"    Date:  {props.get('datetime', '?')}")
        print(f"    Orbit: {props.get('sat:absolute_orbit', '?')}")
        print(f"    Type:  {props.get('product:type', '?')}")
        print(f"    Pols:  {props.get('sar:polarizations', '?')}")

        if coords:
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            print(f"    Bbox:  lon [{min(lons):.2f}, {max(lons):.2f}] "
                  f"lat [{min(lats):.2f}, {max(lats):.2f}]")

        print()

    catalog.close()
    return products


# ---------------------------------------------------------------------------
# Step 2: Download a product
# ---------------------------------------------------------------------------

def download_product(product_id: str):
    """Download a specific product by ID.

    Parameters
    ----------
    product_id : str
        STAC item ID from search results.
    """
    catalog = BIOMASSCatalog(str(DATA_DIR))

    # Ensure the product is in the local database (re-query if needed)
    catalog.query_esa(bbox=BBOX_NEW_NORCIA, max_results=20)

    print("=" * 70)
    print(f"Downloading: {product_id}")
    print(f"Destination: {DATA_DIR}")
    print("=" * 70)
    print()

    product_path = catalog.download_product(
        product_id,
        destination=DATA_DIR,
        extract=True,
    )

    print(f"\nProduct available at: {product_path}")
    catalog.close()
    return product_path


# ---------------------------------------------------------------------------
# Step 3: Discover local products
# ---------------------------------------------------------------------------

def discover_local():
    """Scan the data directory for existing BIOMASS products."""
    catalog = BIOMASSCatalog(str(DATA_DIR))

    print("=" * 70)
    print(f"Scanning for local BIOMASS products in: {DATA_DIR}")
    print("=" * 70)
    print()

    local = catalog.discover_local()
    print(f"Found {len(local)} local products:\n")
    for path in local:
        print(f"  {path.name}")

    catalog.close()
    return local


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--download":
        # Download a specific product: python discover_and_download.py --download <product_id>
        if len(sys.argv) < 3:
            print("Usage: python discover_and_download.py --download <product_id>")
            sys.exit(1)
        download_product(sys.argv[2])

    elif len(sys.argv) > 1 and sys.argv[1] == "--local":
        # Discover local products
        discover_local()

    elif len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("BIOMASS Data Discovery and Download")
        print()
        print("Usage:")
        print("  python discover_and_download.py              # Search MAAP catalog (default)")
        print("  python discover_and_download.py --download ID # Download product by ID")
        print("  python discover_and_download.py --local       # List local products")
        print()
        print("Before using, configure credentials:")
        print("  ~/.config/geoint/credentials.json")

    else:
        # Default: search and display results
        products = search_products()

        if products:
            # Filter to 1S products and show the recommended download
            single_look = [
                p for p in products
                if "SCS__1S" in p.get("properties", {}).get("product:type", "")
            ]

            if single_look:
                latest = single_look[0]
                print("-" * 70)
                print("To download the most recent product, run:")
                print(f"  python {Path(__file__).name} --download {latest['id']}")
                print("-" * 70)
