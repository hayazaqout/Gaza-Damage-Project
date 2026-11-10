"""
Download the UNOSAT Gaza Strip Comprehensive Building & Housing Unit Damage
Assessment (29 Feb 2024 edition -- the edition cited in the paper as ref [18],
UNOSAT product 3804).

unosat.org's own product page is a JS-rendered app and isn't scrapeable with a
plain HTTP client, but this product is mirrored on the Humanitarian Data
Exchange (HDX). HDX's CKAN API gives a direct, anonymous download link to the
underlying file, served from unosat.org's static file storage. The result is an
Esri File Geodatabase (.gdb, a directory of binary table files, not a single
file), zipped for distribution.

Reading the .gdb afterwards requires GDAL's built-in OpenFileGDB driver (read
support is included in a stock GDAL -- no proprietary ESRI SDK needed), e.g. via
`geopandas.read_file(path, layer="Damage_Sites")`. This script only downloads
and extracts the data; it does not assume exact field names, since those could
not be verified without GDAL available in the download environment. The actual
analysis notebook discovers layers/columns at runtime instead of hardcoding them.

Usage:
    python3 scripts/download_unosat.py
"""
import json
import subprocess
import zipfile
from pathlib import Path

HDX_PACKAGE_API = (
    "https://data.humdata.org/api/3/action/package_show"
    "?id=unosat-gaza-strip-comprehensive-building-housing-unit-damage-assessment-march-2024"
)
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "unosat"


def fetch(url: str) -> bytes:
    result = subprocess.run(
        ["curl", "-sL", "-A", "Mozilla/5.0", "--max-time", "120", url],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    )
    return result.stdout


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Querying HDX CKAN API for dataset metadata ...")
    meta = json.loads(fetch(HDX_PACKAGE_API))
    if not meta.get("success"):
        raise RuntimeError(f"HDX API call failed: {meta}")

    resources = meta["result"]["resources"]
    gdb_resources = [r for r in resources if r["url"].lower().endswith(".zip")]
    if not gdb_resources:
        raise RuntimeError(f"No .zip (geodatabase) resource found; resources were: {resources}")
    resource = gdb_resources[0]

    print(f"Downloading {resource['name']} ({resource.get('size', '?')} bytes) ...")
    zip_bytes = fetch(resource["url"])
    zip_path = OUTPUT_DIR / resource["name"]
    zip_path.write_bytes(zip_bytes)
    print(f"  saved to {zip_path} ({len(zip_bytes)/1e6:.1f} MB)")

    print("Extracting geodatabase ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(OUTPUT_DIR)

    gdb_dirs = sorted(OUTPUT_DIR.glob("*.gdb"))
    print(f"Done: geodatabase available at {gdb_dirs[0] if gdb_dirs else '(not found)'}")
    print("Note: reading this layer requires geopandas/fiona with GDAL's OpenFileGDB "
          "driver (standard in Colab after `pip install geopandas`).")


if __name__ == "__main__":
    main()
