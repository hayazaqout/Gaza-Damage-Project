"""Geospatial plumbing: section metadata, affine transforms, CRS handling."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from pyproj import Transformer
from rasterio.transform import from_bounds
from rasterio import features as rio_features

from . import config


@dataclass
class SectionImage:
    section: int
    role: str  # "before" or "after"
    date: str
    bbox_wgs84: tuple  # (minx, miny, maxx, maxy) in EPSG:4326
    image: np.ndarray  # H x W x 3 uint8, RGB


def _find_pair(section_dir: Path, role: str):
    matches = sorted(section_dir.glob(f"{role}_*.png"))
    if not matches:
        raise FileNotFoundError(f"No {role!r} image found in {section_dir}")
    return matches[0]


def load_section_image(section: int, role: str, imagery_dir: Path = config.IMAGERY_DIR) -> SectionImage:
    """Load one before/after PNG + its bbox metadata for a section, as downloaded
    by scripts/download_imagery.py."""
    section_dir = imagery_dir / f"section_{section}"
    png_path = _find_pair(section_dir, role)
    date = png_path.stem.split("_", 1)[1]
    json_path = section_dir / f"{role}_{date}.json"
    meta = json.loads(json_path.read_text())
    # Metadata schema is not fully uniform across the Kaggle archive: most
    # entries have a top-level "bbox", but some nest it under "metadata.bbox".
    bbox = meta.get("bbox") or meta.get("metadata", {}).get("bbox")
    if bbox is None:
        raise KeyError(f"No bbox found (checked top-level and metadata.bbox) in {json_path}")
    img = np.array(Image.open(png_path).convert("RGB"))
    return SectionImage(
        section=section, role=role, date=date,
        bbox_wgs84=tuple(bbox), image=img,
    )


def reproject_bbox(bbox_wgs84, dst_crs=config.WORKING_CRS):
    """Reproject a (minx, miny, maxx, maxy) WGS84 bbox to the working CRS."""
    transformer = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
    minx, miny, maxx, maxy = bbox_wgs84
    xs, ys = transformer.transform([minx, maxx], [miny, maxy])
    return (min(xs), min(ys), max(xs), max(ys))


def section_affine_transform(bbox_wgs84, width: int, height: int, dst_crs=config.WORKING_CRS):
    """Affine transform mapping pixel (col,row) -> working-CRS (x,y), matching
    how the section's bbox+PNG tile the ground (north-up, row 0 = top/north)."""
    minx, miny, maxx, maxy = reproject_bbox(bbox_wgs84, dst_crs)
    return from_bounds(minx, miny, maxx, maxy, width, height)


def rasterize_building_ids(buildings_gdf, bbox_wgs84, width: int, height: int, dst_crs=config.WORKING_CRS):
    """Burn every building's row index (1-based; 0 = no building) into a single
    (height, width) int32 raster aligned with a section's image grid. Shared by
    weak-label change-statistic computation and building-level score aggregation
    so both use exactly the same per-building pixel footprint."""
    transform = section_affine_transform(bbox_wgs84, width, height, dst_crs)
    reprojected = buildings_gdf.to_crs(dst_crs)
    shapes = [
        (geom, i + 1)
        for i, geom in enumerate(reprojected.geometry)
        if geom is not None and not geom.is_empty
    ]
    if not shapes:
        return np.zeros((height, width), dtype=np.int32)
    return rio_features.rasterize(
        shapes, out_shape=(height, width), transform=transform,
        fill=0, dtype=np.int32, all_touched=True,
    )


def rasterize_buildings(buildings_gdf, bbox_wgs84, width: int, height: int,
                         value_field: str, dst_crs=config.WORKING_CRS, fill=config.LABEL_IGNORE):
    """Burn a GeoDataFrame of building polygons (with an integer `value_field`)
    into a (height, width) uint8 raster aligned with a section's image grid."""
    transform = section_affine_transform(bbox_wgs84, width, height, dst_crs)
    reprojected = buildings_gdf.to_crs(dst_crs)
    shapes = [
        (geom, int(val))
        for geom, val in zip(reprojected.geometry, reprojected[value_field])
        if geom is not None and not geom.is_empty
    ]
    if not shapes:
        return np.full((height, width), fill, dtype=np.uint8)
    return rio_features.rasterize(
        shapes, out_shape=(height, width), transform=transform,
        fill=fill, dtype=np.uint8, all_touched=False,
    )
