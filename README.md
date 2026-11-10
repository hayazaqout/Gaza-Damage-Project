# GeoDamage AI -- Gaza Damage Assessment

A weakly supervised, six-channel U-Net pipeline for building-level damage
screening in Al-Shati Camp, Gaza, from freely available before/after satellite
imagery -- implementing the methodology described in *"GeoDamage AI: A weakly
supervised six-channel U-Net pipeline for building-level damage screening in
Al-Shati Camp, Gaza."*

**Read this before trusting any numbers this repo produces:** the paper's
exact reported results (pixel threshold 0.23, positive-class F1 0.982,
building thresholds 0.570/0.918/0.955, a 2,986-building inventory, etc.) are
empirical outputs of the authors' own training run on their own manually
curated 15-section split and original code, which aren't included here. This
repo **reimplements the paper's procedure** faithfully (verified against real
downloaded data wherever that didn't require a GPU -- see
`tests/test_geodamage.py`), but running it produces its own numbers on its own
data split, not the paper's exact figures. Every notebook states explicitly,
inline, where it's following the paper exactly vs. making a documented,
necessary assumption.

## What's in this repo

This branch (`paper-aligned-pipeline`) contains two generations of the
pipeline, kept side by side on purpose:

| | Prototype (Appendix A) | Final pipeline (Sec. 3-4) |
|---|---|---|
| Notebook | `notebooks/01_prototype_pipeline.ipynb` | `notebooks/02_geodamage_ai_pipeline.ipynb` |
| Buildings from | OpenStreetMap only | Microsoft GlobalMLBuildingFootprints |
| Damage labels | Heuristic pixel-difference pseudo-label | UNOSAT weak labels + asymmetric likely-intact rule + ignore class |
| Architecture | Two-stage: ResNet34 U-Net (buildings) -> EfficientNet-B4 6ch U-Net (damage) | Single native six-channel U-Net, early fusion |
| Evaluation | Random 80/20 patch split | Spatially isolated train/val/test, validation-only threshold calibration |
| Output | Overlay image + damage % | Calibrated building-level inventory, confidence tiers, GIS export, web app |

The prototype is kept, translated to English and cleaned up, because it's the
real, working development history the paper's Appendix A describes as
superseded -- not because it's the recommended pipeline. Use notebook 02 for
anything resembling the paper's actual results.

## Repo layout

```
geodamage/              Tested Python package backing both the final-pipeline
                         notebook and the web app: data loading, weak-label
                         construction, the six-channel U-Net, loss, threshold
                         calibration, and building-level inventory/GIS export.
tests/test_geodamage.py Pytest suite -- synthetic tests for model/loss/
                         calibration logic, plus tests that run against the
                         real downloaded data once scripts/download_*.py has
                         been run (9/9 passing locally at time of writing).
scripts/                download_imagery.py, download_building_footprints.py,
                         download_unosat.py -- see data/README.md.
notebooks/
  01_prototype_pipeline.ipynb      Appendix A prototype, English, consolidated
                                    from the repo's original four notebooks.
  02_geodamage_ai_pipeline.ipynb   The paper's final methodology, end to end.
  03_web_app_demo.ipynb            Colab launcher for app/streamlit_app.py.
app/streamlit_app.py    Building-inventory map + live inference.
data/README.md          What each data source is, its license, and how to
                         re-download it (nothing is vendored in git).
```

## Quickstart

```bash
pip install -r requirements.txt
python3 scripts/download_imagery.py               # ~60MB: 15 sections, before/after
python3 scripts/download_building_footprints.py   # ~100MB: Gaza Strip building footprints
python3 scripts/download_unosat.py                 # ~30MB: UNOSAT damage reference
pytest tests/ -v                                    # exercises the non-GPU parts against real data
```

Then open `notebooks/02_geodamage_ai_pipeline.ipynb` on Colab (needs a GPU for
the training cell) to run the full methodology end to end, and
`app/streamlit_app.py` / `notebooks/03_web_app_demo.ipynb` once it has
produced a checkpoint and an inventory export.

## Data sources

Kaggle "Gaza Before and After" satellite imagery (Sentinel-2, ~8m resolution),
Microsoft's GlobalMLBuildingFootprints, and UNOSAT's Gaza Strip Comprehensive
Building & Housing Unit Damage Assessment -- all three are downloaded
anonymously (no API keys/credentials needed) by the scripts in `scripts/`.
See `data/README.md` for exact provenance, licensing, and scope caveats (in
particular: we download a 15-section subset rather than the full ~6GB
imagery archive, and our 12 training sections are our own choice, not a
reconstruction of the paper's original overlap-audited selection).

## Known deviations from the paper (by design, documented inline)

- **Training-section selection**: the paper's test (Sections 4, 5) and
  validation (Section 10) sections are fixed by its text; our 12 training
  sections are our own choice (see `geodamage/config.py`).
- **Al-Shati Camp boundary**: OpenStreetMap has no polygon for Al-Shati Camp
  (confirmed by direct Nominatim query -- only a point node exists), so the
  final inventory's study area is the union of the test sections' bboxes
  instead, which is much larger than the paper's tight camp boundary. The
  final building count will not match the paper's 2,986.
  (`geodamage/inventory.py`)
- **Weak-label hyperparameters** (UNOSAT proximity buffer, the Q20 low-change
  percentile's resulting value, likely-intact minimum distance) are
  recomputed from our own data rather than hardcoded to the paper's reported
  values, since those are properties of the authors' specific training run.
- **Pixel-threshold selection rule**: the paper doesn't fully specify how it
  picks its threshold from the calibration sweep; we use max macro-F1,
  stated explicitly in the notebook.

## Original prototype author's note

The original (pre-rewrite) prototype was developed as an academic deep
learning project for satellite-based damage assessment, focused on Al-Shati
Camp, Gaza Strip, Palestine.
