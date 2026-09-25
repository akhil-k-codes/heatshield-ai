
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
import folium

from folium.plugins import HeatMap
from streamlit_folium import st_folium

from sklearn.ensemble import RandomForestRegressor


# =============================
# LOAD DATA
# =============================

df = pd.read_csv("Kochi_Heat_AI_Data.csv")

df["longitude"] = df[".geo"].apply(
    lambda x: json.loads(x)["coordinates"][0]
)

df["latitude"] = df[".geo"].apply(
    lambda x: json.loads(x)["coordinates"][1]
)


# =============================
# KOCHI AREA
# =============================

# Approximate Kochi study area
MIN_LAT = 9.85
MAX_LAT = 10.10
MIN_LON = 76.15
MAX_LON = 76.40

# Keep only Kochi-area points
df_kochi = df[
    (df["latitude"] >= MIN_LAT) &
    (df["latitude"] <= MAX_LAT) &
    (df["longitude"] >= MIN_LON) &
    (df["longitude"] <= MAX_LON)
].copy()


# =============================
# ML MODEL
# =============================

X = df[["NDVI"]]
y = df["temperature"]

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)


# =============================
# PAGE
# =============================

st.set_page_config(
    page_title="HeatShield AI",
    page_icon="🌡️",
    layout="wide"
)
st.markdown("""
<style>

.main {
    background-color: #0b1220;
}

h1 {
    font-size: 42px !important;
}

h2 {
    margin-top: 25px;
}

[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    padding: 18px;
    border-radius: 15px;
    border: 1px solid rgba(255,255,255,0.1);
}

</style>
""", unsafe_allow_html=True)
st.title("🌡️ HeatShield AI")

st.subheader(
    "Urban Heat Monitoring & Cooling Planner — Kochi"
)

st.write(
    "AI/ML-based analysis of urban surface temperature "
    "and vegetation patterns using satellite data."
)


# =============================
# METRICS
# =============================

col1, col2, col3 = st.columns(3)

col1.metric(
    "Average Surface Temperature",
    f"{df_kochi['temperature'].mean():.1f} °C"
)

col2.metric(
    "Maximum Surface Temperature",
    f"{df_kochi['temperature'].max():.1f} °C"
)

col3.metric(
    "Average NDVI",
    f"{df_kochi['NDVI'].mean():.2f}"
)


# =============================
# HEAT MAP
# =============================

st.header("🔥 Kochi Urban Heat Map")

m = folium.Map(
    location=[9.97, 76.28],
    zoom_start=12,
    min_zoom=11,
    max_zoom=14,
    max_bounds=True,
    control_scale=True
)

# Temperature values
heat_data = [
    [
        row["latitude"],
        row["longitude"],
        row["temperature"]
    ]
    for _, row in df_kochi.iterrows()
]

HeatMap(
    heat_data,
    radius=18,
    blur=22,
    min_opacity=0.35,
    max_zoom=14,
    gradient={
        0.2: "green",
        0.45: "yellow",
        0.7: "orange",
        1.0: "red"
    }
).add_to(m)

# Lock map to Kochi
m.fit_bounds([
    [MIN_LAT, MIN_LON],
    [MAX_LAT, MAX_LON]
])

st_folium(
    m,
    width=1200,
    height=550
)


# =============================
# TEMPERATURE DISTRIBUTION
# =============================

st.header("🌡️ Temperature Distribution")

col1, col2 = st.columns([1, 2])

with col1:
    st.write("### Heat Statistics")
    st.write(f"Average: **{df_kochi['temperature'].mean():.1f} °C**")
    st.write(f"Maximum: **{df_kochi['temperature'].max():.1f} °C**")

with col2:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(df_kochi["temperature"], bins=25)
    ax.set_xlabel("Surface Temperature (°C)")
    ax.set_ylabel("Locations")
    ax.set_title("Temperature Distribution")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)

# =============================
# NDVI VS TEMPERATURE
# =============================

st.header("🌳 Vegetation Impact")

col1, col2 = st.columns([1, 2])

with col1:
    st.write("### What we found")
    st.write(
        "Vegetation level (NDVI) is analyzed against "
        "surface temperature to identify cooling patterns."
    )

with col2:
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    ax2.scatter(
        df_kochi["NDVI"],
        df_kochi["temperature"],
        s=4,
        alpha=0.5
    )
    ax2.set_xlabel("NDVI")
    ax2.set_ylabel("Surface Temperature (°C)")
    ax2.set_title("Vegetation vs Temperature")
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=False)


# =============================
# COOLING SIMULATOR
# =============================

st.header("🌱 Cooling Intervention Simulator")

vegetation_increase = st.slider(
    "Increase vegetation (%)",
    0,
    50,
    10
)

current_ndvi = df_kochi["NDVI"].mean()

new_ndvi = min(
    current_ndvi + vegetation_increase / 100,
    0.8
)

predicted_before = model.predict(
    [[current_ndvi]]
)[0]

predicted_after = model.predict(
    [[new_ndvi]]
)[0]

temperature_change = predicted_after - predicted_before

col1, col2 = st.columns(2)

col1.metric(
    "Baseline Estimated Temperature",
    f"{predicted_before:.2f} °C"
)

col2.metric(
    "Scenario Estimated Temperature",
    f"{predicted_after:.2f} °C",
    f"{temperature_change:.2f} °C"
)

st.info(
    "Scenario results are model-based estimates and should "
    "not be interpreted as guaranteed real-world temperature reductions."
)


# =============================
# AI RECOMMENDATION
# =============================

st.header("🤖 AI Recommendation")

if df_kochi["NDVI"].mean() < 0.3:

    st.warning(
        "Low vegetation detected. Prioritize urban greening, "
        "tree cover and vegetation-based cooling interventions."
    )

else:

    st.success(
        "Vegetation levels are relatively higher. "
        "Focus interventions on remaining high-temperature areas."
    )


st.caption(
    "HeatShield AI | Landsat-derived surface temperature + NDVI"
)
