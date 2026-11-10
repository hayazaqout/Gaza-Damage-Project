"""
Download Microsoft's open building footprints for the Gaza Strip.

Source: https://github.com/microsoft/GlobalMLBuildingFootprints (CDLA Permissive
2.0 license, no authentication required). The master link list partitions the
~1.4 billion worldwide building polygons by region/quadkey; the Gaza Strip is
listed under the region name "GazaStrip" and currently fits in 3 quadkey tiles
(~23MB gzipped total). Each tile is line-delimited GeoJSON (one Feature per
line: a polygon plus "height"/"confidence" properties), gzip-compressed with a
".csv.gz" extension.

Usage:
    python3 scripts/download_building_footprints.py
"""
import gzip
import subprocess
from pathlib import Path

DATASET_LINKS_URL = "https://bfppub.blob.core.windows.net/$web/2026-08-13/dataset-links.csv"
REGION_NAME = "GazaStrip"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "building_footprints"


def fetch(url: str) -> bytes:
    result = subprocess.run(
        ["curl", "-sL", "-A", "Mozilla/5.0", "--max-time", "120", url],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    )
    return result.stdout


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Fetching dataset link list from {DATASET_LINKS_URL} ...")
    csv_bytes = fetch(DATASET_LINKS_URL)
    lines = csv_bytes.decode().splitlines()
    header = lines[0].split(",")
    rows = [dict(zip(header, line.split(","))) for line in lines[1:]]

    tiles = [r for r in rows if r["Location"] == REGION_NAME]
    if not tiles:
        raise RuntimeError(
            f"No tiles found for region {REGION_NAME!r}; Microsoft may have renamed "
            "the region or restructured the dataset-links.csv file."
        )
    print(f"Found {len(tiles)} quadkey tile(s) for {REGION_NAME}.")

    out_path = OUTPUT_DIR / "gaza_buildings.geojsonl"
    n_features = 0
    with open(out_path, "w") as out_f:
        for i, tile in enumerate(tiles, 1):
            url = tile["Url"]
            print(f"  [{i}/{len(tiles)}] downloading quadkey={tile['QuadKey']} ({tile['Size']}) ...")
            gz_bytes = fetch(url)
            text = gzip.decompress(gz_bytes).decode()
            lines = [l for l in text.splitlines() if l.strip()]
            out_f.write("\n".join(lines) + "\n")
            n_features += len(lines)

    print(f"Wrote {n_features} building footprints to {out_path} "
          f"({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
