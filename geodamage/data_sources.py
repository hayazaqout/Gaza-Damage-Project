"""Loaders for the three raw data sources (see data/README.md for provenance)."""
import json

import geopandas as gpd
import pyogrio
from shapely.geometry import box

from . import config


def load_unosat_damage_sites(gdb_path=config.UNOSAT_GDB) -> gpd.GeoDataFrame:
    """Load UNOSAT's damage-site points.

    Rather than hardcoding a guessed layer name, we discover it at runtime: the
    geodatabase ships several "Template_FC" placeholder layers alongside the
    real, dated damage-sites layer (e.g. "Damage_Sites_GazaStrip_20240229").
    We pick the point layer with the most features.
    """
    layers = pyogrio.list_layers(gdb_path)
    point_layers = [name for name, geomtype in layers if geomtype and "Point" in geomtype]
    if not point_layers:
        raise RuntimeError(f"No point layers found in {gdb_path}; layers were: {layers}")

    best_name, best_gdf = None, None
    for name in point_layers:
        gdf = gpd.read_file(gdb_path, layer=name)
        if best_gdf is None or len(gdf) > len(best_gdf):
            best_name, best_gdf = name, gdf

    gdf = best_gdf.set_geometry(best_gdf.geometry.force_2d())  # drop EGM96 Z, keep lon/lat
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


def load_building_footprints(path=config.FOOTPRINTS_PATH, bbox_wgs84=None) -> gpd.GeoDataFrame:
    """Load Microsoft building footprints (line-delimited GeoJSON), optionally
    restricted to a WGS84 bounding box for memory efficiency (the full Gaza
    Strip file has ~280k polygons)."""
    aoi = box(*bbox_wgs84) if bbox_wgs84 else None
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            feat = json.loads(line)
            if aoi is not None:
                minx = min(c[0] for ring in feat["geometry"]["coordinates"] for c in ring)
                maxx = max(c[0] for ring in feat["geometry"]["coordinates"] for c in ring)
                miny = min(c[1] for ring in feat["geometry"]["coordinates"] for c in ring)
                maxy = max(c[1] for ring in feat["geometry"]["coordinates"] for c in ring)
                if maxx < aoi.bounds[0] or minx > aoi.bounds[2] or maxy < aoi.bounds[1] or miny > aoi.bounds[3]:
                    continue
            records.append(feat)
    gdf = gpd.GeoDataFrame.from_features(records, crs="EPSG:4326")
    gdf["building_id"] = range(len(gdf))
    return gdf


def list_downloaded_sections(imagery_dir=config.IMAGERY_DIR):
    return sorted(
        int(p.name.split("_")[1])
        for p in imagery_dir.glob("section_*")
        if p.is_dir()
    )
