"""Shared constants for the GeoDamage AI pipeline.

Values here either come directly from the paper's text (cited inline) or are
our own documented choices where the paper doesn't give enough detail to
reconstruct its exact original setup (e.g. the precise 12 training sections,
which the paper selected via a manual overlap-auditing process over 34
candidate sections that we don't have access to -- see Fig. 1 discussion).
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_OUTPUTS = REPO_ROOT / "data" / "outputs"

IMAGERY_DIR = DATA_RAW / "imagery"
FOOTPRINTS_PATH = DATA_RAW / "building_footprints" / "gaza_buildings.geojsonl"
UNOSAT_GDB = DATA_RAW / "unosat" / "UNOSAT_GazaStrip_CDA_29Feb2024.gdb"
UNOSAT_LAYER = "Damage_Sites_GazaStrip_20240229"  # confirmed via pyogrio.list_layers()

# Paper Sec. 3.1: "Twelve sections were used for training, Section 10 was reserved
# exclusively for validation and threshold calibration, and Sections 4 and 5 ...
# were reserved as the final Al-Shati test."
TEST_SECTIONS = [4, 5]
VAL_SECTIONS = [10]
# Our own choice of 12 training sections (lowest section indices not used for
# val/test) -- NOT a reconstruction of the paper's original overlap-audited set.
TRAIN_SECTIONS = [0, 1, 2, 3, 6, 7, 8, 9, 11, 12, 13, 14]
ALL_SECTIONS = sorted(TRAIN_SECTIONS + VAL_SECTIONS + TEST_SECTIONS)

# Paper Sec. 3.1: "standardized to EPSG:32636 at 8m resolution"
WORKING_CRS = "EPSG:32636"
PIXEL_SIZE_M = 8.0

# Paper Sec. 3.3: 128x128x6 patches
PATCH_SIZE = 128

# Label values used when rasterizing building-level weak labels to per-pixel masks.
LABEL_DAMAGED = 1
LABEL_LIKELY_INTACT = 0
LABEL_IGNORE = 255  # paper Sec. 3.2: "an explicit ignore class (mask value 255)"

# Paper Sec. 3.2: likely-intact buildings must be far enough from any damage
# reference. The paper doesn't give the exact buffer distance, so this is our
# own conservative, documented choice (roughly 2 Al-Shati building footprints).
DAMAGE_PROXIMITY_BUFFER_M = 30.0

# Paper Sec. 3.2: low-change threshold calibrated as a percentile (Q20) of the
# training building pool's before/after RGB-change statistic. The *value*
# (paper reports 0.055273) is a result of their training run on their data; we
# recompute our own Q20 from our own training pool rather than hardcoding theirs.
LOW_CHANGE_PERCENTILE = 20

# Paper Sec. 3.3-3.4: architecture / training hyperparameters
MODEL_IN_CHANNELS = 6
MODEL_ENCODER_FILTERS = (24, 48, 96, 192)
MODEL_BOTTLENECK_FILTERS = 256
MODEL_DROPOUT = 0.1
PAPER_REPORTED_PARAM_COUNT = 3_315_345  # for sanity-checking our reconstruction

LOSS_DICE_WEIGHT = 0.25
TRAIN_LR = 3e-4
TRAIN_BATCH_SIZE = 8
TRAIN_MAX_EPOCHS = 40

# Paper Sec. 4.3-4.4: building-level tiers
TIER_SCREENING = "screening_candidate"
TIER_HIGH_CONF = "high_confidence_candidate"
TIER_VERY_HIGH_CONF = "very_high_confidence_candidate"
TIER_LOW = "lower_ai_score"

AL_SHATI_OSM_QUERY = "Al-Shati Camp, Gaza"
