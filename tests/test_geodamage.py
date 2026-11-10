"""Tests for the geodamage package.

Two tiers:
  - Synthetic tests (model/losses/calibration/dataset): always run, no data
    dependency, check core logic in isolation.
  - Data-backed tests (geo_utils/data_sources/weak_labels/inventory): skipped
    automatically if `data/raw/` hasn't been populated via `scripts/download_*.py`,
    otherwise run against the real downloaded imagery/footprints/UNOSAT data.

Run with: pytest tests/ -v
"""
import numpy as np
import pytest

from geodamage import calibration, config, dataset, losses, model

torch = pytest.importorskip("torch")

DATA_AVAILABLE = config.IMAGERY_DIR.exists() and any(config.IMAGERY_DIR.glob("section_*"))
requires_data = pytest.mark.skipif(not DATA_AVAILABLE, reason="run scripts/download_*.py first")


# ---------------------------------------------------------------------------
# Synthetic tests
# ---------------------------------------------------------------------------

def test_model_forward_shape_and_param_count():
    net = model.SixChannelUNet()
    x = torch.randn(2, 6, 128, 128)
    y = net(x)
    assert y.shape == (2, 1, 128, 128)
    n_params = sum(p.numel() for p in net.parameters())
    # Reconstructed from the paper's description; paper reports 3,315,345.
    assert abs(n_params - config.PAPER_REPORTED_PARAM_COUNT) / config.PAPER_REPORTED_PARAM_COUNT < 0.01


def test_loss_masks_ignore_pixels_and_backprops():
    net = model.SixChannelUNet()
    x = torch.randn(1, 6, 128, 128)
    logits = net(x)
    label = torch.zeros(1, 128, 128)
    label[:, :64, :] = config.LABEL_DAMAGED
    label[:, :10, :10] = config.LABEL_IGNORE

    loss_fn = losses.MaskedWeightedBCEDiceLoss()
    loss = loss_fn(logits, label)
    assert torch.isfinite(loss)
    loss.backward()
    assert net.enc0.block[0].weight.grad is not None


def test_compute_class_weight():
    masks = [np.array([[1, 1, 0], [config.LABEL_IGNORE, 0, 1]], dtype=np.uint8)]
    assert losses.compute_class_weight(masks) == pytest.approx(1.5)  # 3 damaged / 2 intact


def test_extract_patches_covers_whole_padded_image():
    arr = np.random.rand(130, 200, 6).astype(np.float32)
    patches = dataset.extract_patches(arr, patch_size=128)
    # padded to 256x256 (ceil to next multiple of 128 on each axis) -> 2x2 grid
    assert len(patches) == 4
    for p in patches:
        assert p.shape == (128, 128, 6)


def test_pixel_threshold_sweep_picks_reasonable_threshold():
    rng = np.random.default_rng(0)
    n = 4000
    labels = rng.choice(
        [config.LABEL_DAMAGED, config.LABEL_LIKELY_INTACT, config.LABEL_IGNORE],
        size=n, p=[0.6, 0.1, 0.3],
    )
    probs = np.where(labels == config.LABEL_DAMAGED, rng.beta(5, 2, n), rng.beta(2, 5, n))
    _, best = calibration.sweep_pixel_threshold(probs, labels)
    assert 0.0 < best.threshold < 1.0
    assert best.macro_f1 > 0.5  # well-separated synthetic distributions


def test_assign_tier_ordering():
    th = {"screening": 0.3, "spec_90": 0.6, "spec_95": 0.8}
    assert calibration.assign_tier(0.1, th) == config.TIER_LOW
    assert calibration.assign_tier(0.4, th) == config.TIER_SCREENING
    assert calibration.assign_tier(0.7, th) == config.TIER_HIGH_CONF
    assert calibration.assign_tier(0.9, th) == config.TIER_VERY_HIGH_CONF
    assert calibration.assign_tier(float("nan"), th) == "unscored"


# ---------------------------------------------------------------------------
# Data-backed tests
# ---------------------------------------------------------------------------

@requires_data
def test_load_section_image_handles_both_metadata_schemas():
    from geodamage import geo_utils
    # section_4 uses the top-level "bbox" schema; section_10 nests it under
    # "metadata.bbox" -- both must work.
    for sec in (4, 10):
        img = geo_utils.load_section_image(sec, "before")
        assert img.image.ndim == 3 and img.image.shape[2] == 3
        assert len(img.bbox_wgs84) == 4


@requires_data
def test_weak_label_pipeline_runs_on_real_section():
    from geodamage import data_sources, geo_utils, weak_labels

    before = geo_utils.load_section_image(10, "before")
    after = geo_utils.load_section_image(10, "after")
    h, w = before.image.shape[:2]
    buildings = data_sources.load_building_footprints(bbox_wgs84=before.bbox_wgs84)
    unosat = data_sources.load_unosat_damage_sites()

    labeled = weak_labels.assign_weak_labels(
        buildings, before.image, after.image, before.bbox_wgs84, w, h, unosat, q20_threshold=0.0687,
    )
    assert set(labeled["weak_label"].unique()) <= {config.LABEL_DAMAGED, config.LABEL_LIKELY_INTACT, config.LABEL_IGNORE}
    assert (labeled["weak_label"] == config.LABEL_DAMAGED).sum() > 0

    mask = geo_utils.rasterize_buildings(labeled, before.bbox_wgs84, w, h, value_field="weak_label")
    assert mask.shape == (h, w)
    assert set(np.unique(mask)) <= {0, 1, 255}


@requires_data
def test_inventory_scoring_and_export(tmp_path):
    from geodamage import data_sources, geo_utils, inventory

    area = inventory.al_shati_area_from_sections()
    minx, miny, maxx, maxy = area.total_bounds
    buildings = data_sources.load_building_footprints(bbox_wgs84=(minx, miny, maxx, maxy))
    buildings = buildings[buildings.intersects(area.geometry.iloc[0])].reset_index(drop=True)
    assert len(buildings) > 0

    rng = np.random.default_rng(0)
    preds = []
    for sec in config.TEST_SECTIONS:
        img = geo_utils.load_section_image(sec, "before")
        hh, ww = img.image.shape[:2]
        preds.append(inventory.SectionPrediction(
            sec, img.bbox_wgs84, ww, hh, rng.random((hh, ww)).astype(np.float32),
        ))

    thresholds = {"screening": 0.3, "spec_90": 0.6, "spec_95": 0.8}
    inv = inventory.build_inventory(buildings, preds, thresholds)
    assert "ai_damage_score" in inv.columns
    assert inv["ai_tier"].isin(
        [config.TIER_SCREENING, config.TIER_HIGH_CONF, config.TIER_VERY_HIGH_CONF, config.TIER_LOW, "unscored"]
    ).all()

    gpkg_path, geojson_path = inventory.export_inventory(inv, out_dir=tmp_path)
    assert gpkg_path.exists() and geojson_path.exists()
