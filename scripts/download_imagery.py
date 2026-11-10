"""
Download before/after satellite imagery for a chosen subset of sections from the
Kaggle "Gaza Before and After" dataset (abdoomoh/gaza-before-and-after-2).

The full archive is ~6GB (34 sections x ~115 weekly images each), which does not
fit comfortably on disks with limited free space. This script avoids downloading
the whole archive: it reads the remote ZIP's central directory over HTTP Range
requests (the archive is served from Google Cloud Storage, which supports byte
ranges), then downloads only the specific entries we need via further Range
requests. No Kaggle account or API key is required -- Kaggle's public dataset
download endpoint redirects anonymous GET requests to a signed, public GCS URL.

For each requested section we download:
  - the most recent "before" image dated strictly before the war's start
    (2023-10-07), skipping placeholder/empty tiles
  - the most recent "after" image available, skipping placeholders
  - the matching metadata JSON for both dates (contains the WGS84 bbox)

Usage:
    python3 scripts/download_imagery.py
    python3 scripts/download_imagery.py --sections 4 5 10 0 1 2
"""
import argparse
import json
import re
import struct
import subprocess
import zlib
from datetime import date
from pathlib import Path

KAGGLE_DOWNLOAD_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/abdoomoh/gaza-before-and-after-2"
)
WAR_START = date(2023, 10, 7)
PLACEHOLDER_MAX_BYTES = 10_000  # empty/cloud-masked tiles are ~6KB; real tiles are ~2-3MB
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "imagery"

# Sections fixed by the paper's text (Sec. 3.1): test = Sections 4, 5; validation =
# Section 10. The remaining training sections are our own reasonable choice (lowest
# section indices not already used for val/test) -- the paper's exact 12-section
# training curation after overlap-auditing is not reconstructable from the paper text
# alone, so this is a documented approximation, not a claim of an exact match.
DEFAULT_TEST_SECTIONS = [4, 5]
DEFAULT_VAL_SECTIONS = [10]
DEFAULT_TRAIN_SECTIONS = [0, 1, 2, 3, 6, 7, 8, 9, 11, 12, 13, 14]


def resolve_signed_url() -> str:
    """Follow Kaggle's redirect to get the actual (time-limited) GCS download URL.

    Uses a 1-byte ranged GET rather than HEAD: Kaggle's API endpoint returns 404
    for HEAD requests but 302-redirects a ranged GET to a signed, anonymously
    readable GCS URL.
    """
    result = subprocess.run(
        ["curl", "-sL", "-A", "Mozilla/5.0", "--max-time", "30", "-r", "0-0",
         "-o", "/dev/null", "-w", "%{url_effective}", KAGGLE_DOWNLOAD_URL],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    )
    url = result.stdout.decode().strip()
    if "storage.googleapis.com" not in url:
        raise RuntimeError(f"Unexpected redirect target, Kaggle's API may have changed: {url}")
    return url


def curl_range(url: str, start: int, end: int) -> bytes:
    result = subprocess.run(
        ["curl", "-s", "-L", "-A", "Mozilla/5.0", "--max-time", "60",
         "-H", f"Range: bytes={start}-{end}", url],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    )
    return result.stdout


def remote_file_size(url: str) -> int:
    result = subprocess.run(
        ["curl", "-sL", "-A", "Mozilla/5.0", "--max-time", "30",
         "-D", "-", "-o", "/dev/null", "-r", "0-0", url],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    )
    headers = result.stdout.decode(errors="ignore")
    m = re.search(r"Content-Range:\s*bytes\s+\d+-\d+/(\d+)", headers, re.IGNORECASE)
    if not m:
        raise RuntimeError("Could not determine remote file size from Content-Range header")
    return int(m.group(1))


class RemoteZipReader:
    """Minimal seek/read file-like object backed by HTTP Range requests, so the
    stdlib `zipfile` module can parse the central directory of a multi-GB remote
    ZIP without ever downloading the whole thing."""

    def __init__(self, url: str, size: int):
        self.url = url
        self.size = size
        self.pos = 0

    def seek(self, offset, whence=0):
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        elif whence == 2:
            self.pos = self.size + offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        end = (self.size - 1) if (n is None or n < 0) else min(self.pos + n, self.size) - 1
        if end < self.pos:
            return b""
        data = curl_range(self.url, self.pos, end)
        self.pos += len(data)
        return data


def list_zip_index(url: str, size: int):
    import zipfile

    zf = zipfile.ZipFile(RemoteZipReader(url, size))
    return {
        zi.filename: (zi.header_offset, zi.compress_size, zi.file_size, zi.compress_type)
        for zi in zf.infolist()
    }


def extract_entry(url: str, index: dict, name: str) -> bytes:
    off, csize, _usize, ctype = index[name]
    # Local file header is >= 30 bytes, plus filename and extra-field lengths we
    # don't know up front -- fetch generously, then slice using the real lengths.
    raw = curl_range(url, off, off + 30 + len(name.encode()) + 1024 + csize)
    if raw[:4] != b"PK\x03\x04":
        raise RuntimeError(f"Bad local file header for {name!r}")
    name_len = struct.unpack("<H", raw[26:28])[0]
    extra_len = struct.unpack("<H", raw[28:30])[0]
    data_start = 30 + name_len + extra_len
    comp_data = raw[data_start:data_start + csize]
    return comp_data if ctype == 0 else zlib.decompress(comp_data, -15)


def pick_before_after_dates(index: dict, section: str):
    dated = {}
    pattern = re.compile(rf"Gaza Before and After/images/{section}/(\d{{4}}-\d{{2}}-\d{{2}})\.png$")
    for name, (_off, _csize, usize, _ctype) in index.items():
        m = pattern.match(name)
        if m and usize > PLACEHOLDER_MAX_BYTES:
            dated[m.group(1)] = name
    before_dates = sorted(d for d in dated if date.fromisoformat(d) < WAR_START)
    after_dates = sorted(d for d in dated if date.fromisoformat(d) >= WAR_START)
    if not before_dates or not after_dates:
        return None
    return before_dates[-1], after_dates[-1]


def download_section(url: str, index: dict, section_num: int, out_dir: Path):
    section = f"section_{section_num}"
    picked = pick_before_after_dates(index, section)
    if picked is None:
        print(f"  [skip] {section}: no usable before/after pair found")
        return False
    before_date, after_date = picked
    sec_dir = out_dir / section
    sec_dir.mkdir(parents=True, exist_ok=True)

    for role, d in (("before", before_date), ("after", after_date)):
        img_name = f"Gaza Before and After/images/{section}/{d}.png"
        meta_name = f"Gaza Before and After/metadata/{section}_{d}.json"
        img_bytes = extract_entry(url, index, img_name)
        (sec_dir / f"{role}_{d}.png").write_bytes(img_bytes)
        if meta_name in index:
            meta_bytes = extract_entry(url, index, meta_name)
            (sec_dir / f"{role}_{d}.json").write_bytes(meta_bytes)
        print(f"  [ok] {section}/{role}_{d}.png ({len(img_bytes)/1e6:.1f} MB)")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sections", type=int, nargs="+",
        default=DEFAULT_TEST_SECTIONS + DEFAULT_VAL_SECTIONS + DEFAULT_TRAIN_SECTIONS,
        help="Section numbers to download (default: paper's test(4,5)+val(10) "
             "plus 12 chosen training sections, 15 total).",
    )
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    print("Resolving signed Kaggle/GCS URL...")
    url = resolve_signed_url()
    size = remote_file_size(url)
    print(f"Remote archive size: {size/1e9:.2f} GB (reading central directory only)")

    index = list_zip_index(url, size)
    print(f"Archive contains {len(index)} entries across 34 sections.")

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {"test": DEFAULT_TEST_SECTIONS, "val": DEFAULT_VAL_SECTIONS,
                "requested": sorted(args.sections)}
    ok = 0
    for sec_num in sorted(set(args.sections)):
        print(f"Downloading section_{sec_num} ...")
        if download_section(url, index, sec_num, args.out):
            ok += 1
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Done: {ok}/{len(set(args.sections))} sections downloaded to {args.out}")


if __name__ == "__main__":
    main()
