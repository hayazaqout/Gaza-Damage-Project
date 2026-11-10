"""Weak-label construction (paper Sec. 3.2).

Policy, mirroring the paper's deliberately asymmetric rule:
  - "damaged" (positive): building footprint has a matched UNOSAT damage
    reference within DAMAGE_PROXIMITY_BUFFER_M.
  - "likely-intact": building additionally satisfies *all* of:
      (a) valid image quality (no no-data/near-black pixels under its footprint)
      (b) sufficiently far from any UNOSAT damage reference
      (c) a before/after RGB-change statistic below a Q20 percentile, computed
          on the training building pool only, then frozen
  - everything else -> ignore (LABEL_IGNORE), excluded from loss/metrics.

The paper reports its own calibrated Q20 value (0.055273) from its own training
run; we recompute our own Q20 from our own training pool rather than hardcoding
theirs, since the number is a property of the specific training data, not a
universal constant.

Performance note: a per-building Python loop over tens of thousands of
buildings per section (Gaza is dense -- ~50k+ footprints per section) is too
slow to be practically usable, even on Colab. `compute_section_change_stats`
instead rasterizes every footprint *once* into a single building-id raster and
computes all per-building statistics in one vectorized `scipy.ndimage` pass.
"""
import geopandas as gpd
import numpy as np
from scipy import ndimage

from . import config, geo_utils


def compute_section_change_stats(buildings_gdf: gpd.GeoDataFrame, before_img: np.ndarray,
                                  after_img: np.ndarray, bbox_wgs84, width: int, height: int,
                                  near_black_threshold=5, near_black_fraction_max=0.5):
    """Vectorized per-building change statistic + image-quality flag for every
    building in `buildings_gdf`, for one section.

    Returns a DataFrame-aligned pair of arrays (change_stat, valid_quality),
    same length/order as `buildings_gdf`. change_stat is NaN where the building
    has no pixels in this section or quality is invalid.
    """
    n = len(buildings_gdf)
    id_raster = geo_utils.rasterize_building_ids(buildings_gdf, bbox_wgs84, width, height)
    ids = np.arange(1, n + 1)

    present = np.isin(ids, id_raster)  # buildings that actually land in this section

    before_f = before_img.astype(np.float32)
    after_f = after_img.astype(np.float32)
    dist_px = np.linalg.norm(after_f - before_f, axis=-1) / (255.0 * np.sqrt(3))  # (H, W) in [0,1]

    near_black_px = (before_f.max(axis=-1) < near_black_threshold) | (after_f.max(axis=-1) < near_black_threshold)

    sum_dist = ndimage.sum_labels(dist_px, id_raster, index=ids)
    count_px = ndimage.sum_labels(np.ones_like(dist_px), id_raster, index=ids)
    count_black = ndimage.sum_labels(near_black_px.astype(np.float32), id_raster, index=ids)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_dist = sum_dist / count_px
        black_fraction = count_black / count_px

    valid_quality = present & (count_px > 0) & (black_fraction <= near_black_fraction_max)
    change_stat = np.where(valid_quality, mean_dist, np.nan)
    return change_stat, valid_quality


def match_unosat_damage(buildings_gdf: gpd.GeoDataFrame, unosat_gdf: gpd.GeoDataFrame,
                         buffer_m=config.DAMAGE_PROXIMITY_BUFFER_M):
    """Return (matched: bool array, distance_m: float array) per building --
    whether a UNOSAT damage point falls within `buffer_m` of the footprint, and
    the actual nearest-point distance (used later for the min-distance rule).

    Uses a spatial-index nearest join (geopandas `sjoin_nearest`) rather than a
    dense distance matrix, so it stays fast with tens of thousands of points.
    """
    buildings_m = buildings_gdf.to_crs(config.WORKING_CRS).reset_index(drop=True)
    unosat_m = unosat_gdf.to_crs(config.WORKING_CRS)

    nearest = gpd.sjoin_nearest(
        buildings_m[["geometry"]], unosat_m[["geometry"]],
        distance_col="distance_to_unosat_m", how="left",
    )
    nearest = nearest.groupby(level=0)["distance_to_unosat_m"].min()
    distances = nearest.reindex(range(len(buildings_m))).to_numpy()
    matched = distances <= buffer_m
    return matched, distances


def assign_weak_labels(buildings_gdf, before_img, after_img, bbox_wgs84, width, height,
                        unosat_gdf, q20_threshold=None,
                        min_intact_distance_m=config.DAMAGE_PROXIMITY_BUFFER_M):
    """Compute per-building weak labels for one section (vectorized).

    Returns buildings_gdf with added columns: `change_stat`, `valid_quality`,
    `unosat_matched`, `distance_to_unosat_m`, `weak_label`
    (config.LABEL_DAMAGED / LABEL_LIKELY_INTACT / LABEL_IGNORE).

    If `q20_threshold` is None, only damaged/ignore are assigned (used for the
    training-pool pass that first computes the change statistic, before Q20 can
    be calibrated from it).
    """
    buildings_gdf = buildings_gdf.reset_index(drop=True).copy()

    matched, distances = match_unosat_damage(buildings_gdf, unosat_gdf)
    buildings_gdf["unosat_matched"] = matched
    buildings_gdf["distance_to_unosat_m"] = distances

    change_stat, valid_quality = compute_section_change_stats(
        buildings_gdf, before_img, after_img, bbox_wgs84, width, height,
    )
    buildings_gdf["change_stat"] = change_stat
    buildings_gdf["valid_quality"] = valid_quality

    labels = np.full(len(buildings_gdf), config.LABEL_IGNORE, dtype=np.uint8)
    labels[buildings_gdf["unosat_matched"].values] = config.LABEL_DAMAGED

    if q20_threshold is not None:
        likely_intact = (
            (~buildings_gdf["unosat_matched"].values)
            & buildings_gdf["valid_quality"].values
            & (buildings_gdf["distance_to_unosat_m"].values >= min_intact_distance_m)
            & (change_stat <= q20_threshold)
        )
        labels[likely_intact] = config.LABEL_LIKELY_INTACT

    buildings_gdf["weak_label"] = labels
    return buildings_gdf


def calibrate_q20(training_change_stats, percentile=config.LOW_CHANGE_PERCENTILE):
    """Compute the low-change threshold as a percentile of the training pool's
    change statistic, mirroring the paper's `Q20` procedure (their reported
    value 0.055273 is specific to their own training run/data)."""
    valid_stats = np.asarray(training_change_stats, dtype=np.float64)
    valid_stats = valid_stats[~np.isnan(valid_stats)]
    if valid_stats.size == 0:
        raise ValueError("No valid change statistics to calibrate Q20 from")
    return float(np.percentile(valid_stats, percentile))
