# Test data

## n33_w082_1arc_v3.dt2.gz

Gzip-compressed SRTM DTED Level 2 tile covering 33-34 N, 82-81 W
(3601 x 3601 int16 posts, 1 arc-second). Decompressed size is about 26 MB.

- Source: USGS EarthExplorer, SRTM 1 Arc-Second Global, DTED format.
- Producer `USCNIMA`, edition 02, vertical datum EGM96, void-filled from NED.
- The DSI record carries `SecurityHandling = PUBLIC SALE/NO RESTRICTION`
  and classification `U`. The DTED format has no separate licence field.
  Confirm the current USGS data-use terms before redistributing elsewhere.

Used by `tests/test_geolocation_dted_srtm_real.py`, which decompresses it into
a temporary directory.
