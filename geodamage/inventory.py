"""Final building-level inventory (paper Sec. 3.5, 4.4-4.5).

Takes per-section pixel damage-probability rasters (produced by running the
trained model -- this module doesn't run inference itself, so it has no GPU
dependency) and aggregates them to a building-level inventory with AI damage
scores, confidence tiers, and a UNOSAT audit column, then exports GIS layers.

The paper scopes its final inventory to "all 2,986 Al-Shati buildings" found
within the Al-Shati Camp boundary. We fetch that boundary the same way the
repo's existing prototype notebook already does (`osmnx`), intersect it with
Microsoft's building footprints, and score whichever of those buildings fall
inside sections we actually have model predictions for. Our own section
coverage differs from the paper's, so our resulting building count will not
match 2,986 -- this module reports whatever it actually covers rather than
claiming an exact reproduction.
"""
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
from scipy import ndimage
from shapely.geometry import box, Point

from . import calibration, config, geo_utils


def fetch_al_shati_point(query=config.AL_SHATI_OSM_QUERY) -> Point:
    """Geocode Al-Shati Camp via OpenStreetMap/Nominatim (same data source the
    repo's prototype notebook already uses for building footprints).

    Note: OSM only has a single Point node for "Al-Shati Camp" (no mapped area
    polygon/relation exists at the time of writing -- confirmed by querying
    Nominatim directly), so `ox.geocode_to_gdf` cannot be used here (it raises
    when the result isn't a polygon). `al_shati_area_from_sections` below is
    the actually-used study-area definition; this point is kept only as an
    independent sanity check that it's in the right place.
    """
    import osmnx as ox
    lat, lon = ox.geocode(query)
    return Point(lon, lat)


def al_shati_area_from_sections(sections=config.TEST_SECTIONS, imagery_dir=config.IMAGERY_DIR) -> gpd.GeoDataFrame:
    """Define the Al-Shati study area as the union of the paper's own "Al-Shati
    test" section bboxes (Sections 4, 5) -- this is the paper's explicit
    definition of the Al-Shati deployment area, and doesn't depend on an OSM
    boundary polygon that doesn't exist (see `fetch_al_shati_point`)."""
    from . import geo_utils

    boxes = []
    for sec in sections:
        img = geo_utils.load_section_image(sec, "before", imagery_dir)
        boxes.append(box(*img.bbox_wgs84))
    union = boxes[0]
    for b in boxes[1:]:
        union = union.union(b)
    return gpd.GeoDataFrame({"name": ["Al-Shati study area"]}, geometry=[union], crs="EPSG:4326")


@dataclass
class SectionPrediction:
    section: int
    bbox_wgs84: tuple
    width: int
    height: int
    pixel_probs: np.ndarray  # (H, W) float in [0,1], model sigmoid output


def score_buildings(buildings_gdf: gpd.GeoDataFrame, predictions: list) -> gpd.GeoDataFrame:
    """Add an `ai_damage_score` column (Eq. 1: mean pixel probability within
    footprint) to `buildings_gdf`, using whichever `SectionPrediction` each
    building's centroid falls inside. Buildings covered by no section are left
    with NaN (reported, not silently dropped)."""
    buildings_gdf = buildings_gdf.reset_index(drop=True).copy()
    scores = np.full(len(buildings_gdf), np.nan)
    covered_by = np.full(len(buildings_gdf), -1, dtype=int)

    for pred in predictions:
        id_raster = geo_utils.rasterize_building_ids(
            buildings_gdf, pred.bbox_wgs84, pred.width, pred.height,
        )
        ids = np.arange(1, len(buildings_gdf) + 1)
        present = np.isin(ids, id_raster)
        sum_probs = ndimage.sum_labels(pred.pixel_probs, id_raster, index=ids)
        count_px = ndimage.sum_labels(np.ones_like(pred.pixel_probs), id_raster, index=ids)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_probs = sum_probs / count_px
        newly_covered = present & (count_px > 0) & np.isnan(scores)
        scores[newly_covered] = mean_probs[newly_covered]
        covered_by[newly_covered] = pred.section

    buildings_gdf["ai_damage_score"] = scores
    buildings_gdf["scored_from_section"] = covered_by
    return buildings_gdf


def build_inventory(buildings_gdf: gpd.GeoDataFrame, predictions: list, thresholds: dict,
                     unosat_gdf: gpd.GeoDataFrame = None,
                     unosat_buffer_m=config.DAMAGE_PROXIMITY_BUFFER_M) -> gpd.GeoDataFrame:
    """Score buildings, assign confidence tiers, and (optionally) attach a
    UNOSAT audit column -- the full building-level product described in the
    paper's Sec. 4.4-4.5, minus the exact building count (see module docstring)."""
    scored = score_buildings(buildings_gdf, predictions)
    scored["ai_tier"] = [calibration.assign_tier(s, thresholds) for s in scored["ai_damage_score"]]

    if unosat_gdf is not None:
        matched, distances = _match_for_audit(scored, unosat_gdf, unosat_buffer_m)
        scored["unosat_matched"] = matched
        scored["distance_to_unosat_m"] = distances

    return scored


def _match_for_audit(buildings_gdf, unosat_gdf, buffer_m):
    from . import weak_labels
    return weak_labels.match_unosat_damage(buildings_gdf, unosat_gdf, buffer_m)


def export_inventory(inventory_gdf: gpd.GeoDataFrame, out_dir=config.DATA_OUTPUTS,
                      basename="al_shati_building_inventory"):
    """Write the inventory as both GeoPackage (primary, preserves dtypes well)
    and GeoJSON (easy to consume from the web app / other tools)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    export_gdf = inventory_gdf.copy()
    export_gdf["scored_from_section"] = export_gdf["scored_from_section"].astype(int)

    gpkg_path = out_dir / f"{basename}.gpkg"
    geojson_path = out_dir / f"{basename}.geojson"
    export_gdf.to_file(gpkg_path, driver="GPKG")
    export_gdf.to_file(geojson_path, driver="GeoJSON")
    return gpkg_path, geojson_path


def tier_summary(inventory_gdf: gpd.GeoDataFrame) -> dict:
    counts = inventory_gdf["ai_tier"].value_counts().to_dict()
    total = len(inventory_gdf)
    return {tier: {"count": n, "percent": round(100 * n / total, 2)} for tier, n in counts.items()}
