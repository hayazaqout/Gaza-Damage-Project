"""GeoDamage AI pipeline library.

Backs `notebooks/02_geodamage_ai_pipeline.ipynb` and `app/streamlit_app.py`.
See each submodule's docstring for how it maps to a specific section of the
paper, and what's a faithful reconstruction vs. a documented assumption where
the paper doesn't give enough detail to reproduce exactly.
"""
from . import (
    calibration,
    config,
    data_sources,
    dataset,
    geo_utils,
    inventory,
    losses,
    model,
    weak_labels,
)

__all__ = [
    "calibration", "config", "data_sources", "dataset",
    "geo_utils", "inventory", "losses", "model", "weak_labels",
]
