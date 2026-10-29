"""Solar India: how much solar power will your rooftop make?
Run locally with:  streamlit run app.py
The app never trains anything. It loads the models saved by notebook 02 and feeds them
tomorrow's weather forecast from Open-Meteo (free, no key)."""

import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st

ROOT = Path(__file__).parent
MODEL_DIR, DATA_DIR = ROOT / "models", ROOT / "data"

st.set_page_config(page_title="Solar India", page_icon="☀️", layout="wide")

# ---------------------------------------------------------------- look and feel
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@600&family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Fraunces', serif !important; letter-spacing: -0.01em; }
.block-container { padding-top: 2rem; max-width: 1200px; }
.hero { background: linear-gradient(120deg, #1d1a3a 0%, #3b2a5c 55%, #e07a2f 130%); color: #fff7ec;
        border-radius: 18px; padding: 28px 32px; margin-bottom: 18px; }
.hero h1 { color: #fff7ec; margin: 0 0 6px 0; font-size: 2.4rem; }
.hero p { margin: 0; opacity: .85; font-size: 1.02rem; }
.card { background: #fffaf3; border: 1px solid #f0dcc4; border-radius: 16px; padding: 18px 20px; height: 100%; }
.card .label { font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; color: #8a6a4a; }
.card .value { font-family: 'Fraunces', serif; font-size: 2.1rem; color: #2a1d12; line-height: 1.15; }
.card .sub { font-size: .85rem; color: #7a6450; }
.pill { display: inline-block; padding: 4px 12px; border-radius: 999px; font-weight: 600; font-size: .9rem; }
.pill.good { background: #ffe2b3; color: #8a4b00; }
.pill.average { background: #e7e2f5; color: #4a3d7a; }
.pill.poor { background: #dfe5ec; color: #34495e; }
</style>
""", unsafe_allow_html=True)

COLOURS = {"good": [244, 162, 54], "average": [150, 130, 210], "poor": [110, 130, 150]}


# ---------------------------------------------------------------- loading
@st.cache_resource
def load_models():
    import tensorflow as tf                                     # loaded once, only here
    meta = json.loads((MODEL_DIR / "meta.json").read_text())
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    svm = joblib.load(MODEL_DIR / "svm.joblib")
    nn = tf.keras.models.load_model(MODEL_DIR / "nn_regression.keras")
    return meta, scaler, svm, nn

@st.cache_data
def load_tables():
    cities = pd.read_csv(MODEL_DIR / "cities.csv")
    typical = pd.read_csv(MODEL_DIR / "clear_sky_typical.csv")
    history = pd.read_csv(DATA_DIR / "solar.csv", parse_dates=["date"]) if (DATA_DIR / "solar.csv").exists() else None
    return cities, typical, history

meta, scaler, svm, nn = load_models()
cities, typical, history = load_tables()
FEATURES, CLASS_NAMES = meta["features"], meta["class_names"]


# ---------------------------------------------------------------- weather forecast -> model features
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_forecast(lats, lons):
    """7-day hourly forecast for every city in one request (Open-Meteo)."""
    params = {"latitude": ",".join(map(str, lats)), "longitude": ",".join(map(str, lons)),
              "hourly": "temperature_2m,dew_point_2m,relative_humidity_2m,cloud_cover,precipitation,surface_pressure,wind_speed_10m",
              "wind_speed_unit": "ms", "forecast_days": 7, "timezone": "Asia/Kolkata"}
    r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=30)
    r.raise_for_status()
    out = r.json()
    return out if isinstance(out, list) else [out]

def daily_features(hourly, city_row):
    """Turn hourly forecast values into the same daily features NASA POWER gives (notebook 01 units)."""
    h = pd.DataFrame(hourly)
    h["date"] = pd.to_datetime(h["time"]).dt.normalize()
    d = h.groupby("date").agg(temp=("temperature_2m", "mean"), temp_max=("temperature_2m", "max"),
                              temp_min=("temperature_2m", "min"), dew_point=("dew_point_2m", "mean"),
                              humidity=("relative_humidity_2m", "mean"), cloud=("cloud_cover", "mean"),
                              rain=("precipitation", "sum"), pressure=("surface_pressure", "mean"),
                              wind10=("wind_speed_10m", "mean")).reset_index()
    d["temp_range"] = d["temp_max"] - d["temp_min"]
    d["pressure"] = d["pressure"] / 10                                  # hPa -> kPa
    d["wind"] = d["wind10"] * 0.748                                     # 10 m wind -> 2 m wind (FAO-56 rule)
    e = 6.112 * np.exp(17.67 * d["dew_point"] / (d["dew_point"] + 243.5))     # vapour pressure (hPa)
    d["specific_humidity"] = 1000 * 0.622 * e / (d["pressure"] * 10 - 0.378 * e)   # g/kg
    d["lat"], d["lon"], d["city"] = city_row["lat"], city_row["lon"], city_row["city"]
    d["day_of_year"] = d["date"].dt.day_of_year
    d = d.merge(typical, on=["city", "day_of_year"], how="left")
    return d

def predict(df):
    """Solar energy (neural network) and day class (SVM) for rows of the 12 features."""
    X = scaler.transform(df[FEATURES])
    df = df.copy()
    df["solar_pred"] = np.clip(nn.predict(X, verbose=0).ravel(), 0, None)
    df["day_class"] = [CLASS_NAMES[c] for c in svm.predict(X)]
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def all_city_forecasts():
    raw = fetch_forecast(tuple(cities["lat"]), tuple(cities["lon"]))
    frames = [daily_features(r["hourly"], row) for r, (_, row) in zip(raw, cities.iterrows())]
    days = pd.concat(frames, ignore_index=True).dropna(subset=FEATURES)
    return predict(days)


# ---------------------------------------------------------------- helpers
def card(label, value, sub=""):
    st.markdown(f"<div class='card'><div class='label'>{label}</div><div class='value'>{value}</div>"
                f"<div class='sub'>{sub}</div></div>", unsafe_allow_html=True)

def india_map(df):
    df = df.assign(colour=df["day_class"].map(COLOURS), radius=df["solar_pred"] * 9000,
                   kwh=df["solar_pred"].round(2))
    layer = pdk.Layer("ScatterplotLayer", df, get_position="[lon, lat]", get_fill_color="colour",
                      get_radius="radius", pickable=True, opacity=0.85, stroked=True,
                      get_line_color=[255, 255, 255], line_width_min_pixels=1)
    view = pdk.ViewState(latitude=22.5, longitude=80, zoom=3.6)
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, map_style="light",
                             tooltip={"text": "{city}\n{kwh} kWh/m² · {day_class} day"}), height=520)


# ---------------------------------------------------------------- page
st.markdown("<div class='hero'><h1>☀️ Solar India</h1><p>How much electricity will a rooftop solar panel make tomorrow? "
            "Forecasts for 22 Indian cities, from a model trained on 11 years of NASA satellite data.</p></div>",
            unsafe_allow_html=True)

tab_forecast, tab_history = st.tabs(["Tomorrow's forecast", "Past days"])

with tab_forecast:
    try:
        with st.spinner("Getting the weather forecast for 22 cities..."):
            fc = all_city_forecasts()
        online = True
    except Exception as error:
        online = False
        st.warning(f"The weather forecast service can't be reached right now ({type(error).__name__}). "
                   "Try again in a minute, or use the **Past days** tab, which works offline.")

    if online:
        tomorrow = pd.Timestamp.now(tz="Asia/Kolkata").normalize().tz_localize(None) + pd.Timedelta(days=1)
        tmr = fc[fc["date"] == tomorrow]
        if tmr.empty:
            tmr = fc[fc["date"] == fc["date"].min()]

        left, right = st.columns([1.25, 1])
        with left:
            st.subheader(f"Sunlight across India, {tmr['date'].iloc[0]:%A %d %B}")
            india_map(tmr)
            st.caption("Bigger circle = more solar energy. Orange = good day, purple = average, grey = poor.")
        with right:
            city = st.selectbox("Your city", sorted(cities["city"]), index=sorted(cities["city"]).index("Mumbai"))
            size = st.slider("Rooftop system size (kW)", 1.0, 10.0, 3.0, 0.5)
            tariff = st.number_input("Electricity price (₹ per kWh)", 3.0, 15.0, 8.0, 0.5)

            row = tmr[tmr["city"] == city].iloc[0]
            energy = size * row["solar_pred"] * 0.75            # 75% performance ratio: heat, wiring, inverter losses
            st.markdown(f"<span class='pill {row['day_class']}'>{row['day_class'].title()} solar day</span>", unsafe_allow_html=True)
            st.write("")
            c1, c2 = st.columns(2)
            with c1:
                card("Your panels will make", f"{energy:.1f} kWh", f"from a {size:g} kW system")
            with c2:
                card("Worth about", f"₹{energy * tariff:.0f}", f"at ₹{tariff:g} per kWh")
            st.write("")
            c3, c4, c5 = st.columns(3)
            c3.metric("Cloud cover", f"{row['cloud']:.0f}%")
            c4.metric("Humidity", f"{row['humidity']:.0f}%")
            c5.metric("Rain", f"{row['rain']:.1f} mm")

        st.subheader(f"Next 7 days in {city}")
        week = fc[fc["city"] == city].set_index("date")
        week_kwh = (size * week["solar_pred"] * 0.75).rename("kWh from your panels")
        chart_df = pd.DataFrame({"day": week.index.strftime("%a %d"), "kWh": week_kwh.values, "type": week["day_class"].values})
        st.altair_chart(alt.Chart(chart_df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
            x=alt.X("day", sort=None, title=None, axis=alt.Axis(labelAngle=0)), y=alt.Y("kWh", title="kWh from your panels"),
            color=alt.Color("type", scale=alt.Scale(domain=CLASS_NAMES, range=["#6e8296", "#9682d2", "#f4a236"]), title="Day type"),
            tooltip=["day", alt.Tooltip("kWh", format=".1f"), "type"]).properties(height=280), width="stretch")
        st.dataframe(pd.DataFrame({"Day": week.index.strftime("%a %d %b"), "Solar (kWh/m²)": week["solar_pred"].round(2).values,
                                   "Your panels (kWh)": week_kwh.round(1).values, "Day type": week["day_class"].values,
                                   "Cloud (%)": week["cloud"].round(0).values}), hide_index=True, width="stretch")

with tab_history:
    if history is None:
        st.info("Add data/solar.csv to the app folder to explore past days.")
    else:
        st.write("Pick a city and a month: the model's estimate against what NASA actually measured.")
        c1, c2 = st.columns(2)
        h_city = c1.selectbox("City", sorted(cities["city"]), key="h_city", index=sorted(cities["city"]).index("Jodhpur"))
        months = history["date"].dt.to_period("M").drop_duplicates().sort_values()
        h_month = c2.selectbox("Month", months.astype(str).tolist()[::-1], key="h_month")

        past = history[(history["city"] == h_city) & (history["date"].dt.to_period("M").astype(str) == h_month)].copy()
        past["temp_range"] = past["temp_max"] - past["temp_min"]
        past = past.merge(typical, on=["city", "day_of_year"], how="left")
        past["clear_sky_typical"] = past["clear_sky_typical"].fillna(past["clear_sky"])
        past = predict(past)

        st.line_chart(past.set_index("date")[["solar", "solar_pred"]].rename(
            columns={"solar": "measured by NASA", "solar_pred": "model estimate"}))
        err = (past["solar"] - past["solar_pred"]).abs().mean()
        st.caption(f"Average error this month: {err:.2f} kWh/m² per day. Days are from the "
                   f"{', '.join(past['split'].unique())} set.")
