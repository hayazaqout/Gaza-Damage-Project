"""Interactive web app: building-inventory map + live inference.

Mirrors the paper's description of delivery through "GIS exports and an
interactive web application with live inference" (Sec. 4.4-4.5). Two panels:

  1. Map view of the final building inventory (from
     `notebooks/02_geodamage_ai_pipeline.ipynb`'s GIS export), colored by
     confidence tier, with per-building popups.
  2. Live inference: pick a downloaded section (or upload a before/after image
     pair) and run the trained six-channel U-Net checkpoint to produce a
     damage-probability overlay on the spot.

Requires, from running notebook 02 first:
  - `data/outputs/al_shati_building_inventory.gpkg` (the inventory)
  - `data/outputs/six_channel_unet_best.pt` (the trained checkpoint)
Neither is faked or mocked if missing -- the relevant panel just explains
what's needed and how to produce it, rather than showing placeholder data.

Run locally:   streamlit run app/streamlit_app.py
Run on Colab:  see notebooks/03_web_app_demo.ipynb
"""
import sys
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from geodamage import config, data_sources, geo_utils  # noqa: E402

st.set_page_config(page_title="GeoDamage AI -- Al-Shati Camp", layout="wide")
st.title("GeoDamage AI -- Al-Shati Camp Building Damage Screening")
st.caption(
    "AI-estimated damage screening candidates derived from low-resolution imagery and weak labels. "
    "**Not field-confirmed damage counts.** See the paper's Sec. 5 for limitations."
)

INVENTORY_PATH = config.DATA_OUTPUTS / "al_shati_building_inventory.gpkg"
CHECKPOINT_PATH = config.DATA_OUTPUTS / "six_channel_unet_best.pt"

tab_map, tab_inference = st.tabs(["Building Inventory Map", "Live Inference"])

# ---------------------------------------------------------------------------
# Tab 1: inventory map
# ---------------------------------------------------------------------------
with tab_map:
    if not INVENTORY_PATH.exists():
        st.warning(
            f"No inventory found at `{INVENTORY_PATH}`. Run "
            "`notebooks/02_geodamage_ai_pipeline.ipynb` through the "
            "'Final inventory' section first -- it writes this file."
        )
    else:
        import geopandas as gpd
        import folium
        from streamlit_folium import st_folium

        inventory = gpd.read_file(INVENTORY_PATH)

        tier_colors = {
            config.TIER_VERY_HIGH_CONF: "#7f1d1d",
            config.TIER_HIGH_CONF: "#dc2626",
            config.TIER_SCREENING: "#f59e0b",
            config.TIER_LOW: "#16a34a",
            "unscored": "#9ca3af",
        }

        col1, col2 = st.columns([1, 3])
        with col1:
            st.subheader("Reporting tiers")
            counts = inventory["ai_tier"].value_counts()
            for tier, color in tier_colors.items():
                n = int(counts.get(tier, 0))
                st.markdown(
                    f"<span style='color:{color}'>●</span> {tier.replace('_', ' ')}: **{n}**",
                    unsafe_allow_html=True,
                )
            st.metric("Total buildings", len(inventory))

        with col2:
            bounds = inventory.total_bounds
            center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
            fmap = folium.Map(location=center, zoom_start=15, tiles="CartoDB positron")

            for _, row in inventory.iterrows():
                if row.geometry is None or row.geometry.is_empty:
                    continue
                color = tier_colors.get(row["ai_tier"], "#9ca3af")
                score = row.get("ai_damage_score", float("nan"))
                popup = (
                    f"Score: {score:.3f}<br>Tier: {row['ai_tier']}<br>"
                    f"UNOSAT matched: {row.get('unosat_matched', 'n/a')}"
                )
                folium.GeoJson(
                    row.geometry.__geo_interface__,
                    style_function=lambda _f, c=color: {"fillColor": c, "color": c, "weight": 1, "fillOpacity": 0.6},
                    tooltip=popup,
                ).add_to(fmap)

            st_folium(fmap, width=900, height=600)

# ---------------------------------------------------------------------------
# Tab 2: live inference
# ---------------------------------------------------------------------------
with tab_inference:
    if not CHECKPOINT_PATH.exists():
        st.warning(
            f"No trained checkpoint found at `{CHECKPOINT_PATH}`. Run "
            "`notebooks/02_geodamage_ai_pipeline.ipynb`'s training section on "
            "Colab (GPU required) first -- it saves this file."
        )
    else:
        import torch
        from geodamage import dataset, model

        @st.cache_resource
        def load_model():
            net = model.SixChannelUNet()
            net.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
            net.eval()
            return net

        net = load_model()
        st.success(f"Loaded checkpoint from `{CHECKPOINT_PATH}`.")

        available_sections = data_sources.list_downloaded_sections()
        section = st.selectbox("Section to run inference on", available_sections)

        if st.button("Run inference"):
            with st.spinner("Running six-channel U-Net inference..."):
                before = geo_utils.load_section_image(section, "before")
                after = geo_utils.load_section_image(section, "after")
                img6ch = dataset.stack_before_after(before.image, after.image)
                h, w = img6ch.shape[:2]

                probs = np.zeros((h, w), dtype=np.float32)
                ps = config.PATCH_SIZE
                with torch.no_grad():
                    for row in range(0, h, ps):
                        for col in range(0, w, ps):
                            r1, c1 = min(row + ps, h), min(col + ps, w)
                            patch = np.zeros((ps, ps, 6), dtype=np.float32)
                            patch[:r1 - row, :c1 - col] = img6ch[row:r1, col:c1]
                            x = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).float()
                            p = torch.sigmoid(net(x))[0, 0].numpy()
                            probs[row:r1, col:c1] = p[:r1 - row, :c1 - col]

            col1, col2, col3 = st.columns(3)
            col1.image(before.image, caption=f"Before ({before.date})", use_container_width=True)
            col2.image(after.image, caption=f"After ({after.date})", use_container_width=True)
            col3.image(probs, caption="Predicted damage probability", clamp=True, use_container_width=True)
