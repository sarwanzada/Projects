import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import joblib
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(page_title="Flood Risk Forecasting — Indus Basin", layout="wide")

# ---------------------------------------------------------------------------
# Load the saved model package (model + threshold + feature list + location)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("flood_model_final.pkl")

model_package = load_model()
model = model_package["model"]
THRESHOLD = model_package["threshold"]
FEATURE_COLUMNS = model_package["feature_columns"]
FORECAST_HORIZON_DAYS = model_package["forecast_horizon_days"]

# ---------------------------------------------------------------------------
# Sidebar — location selector
# ---------------------------------------------------------------------------
st.sidebar.title("📍 Location")

LOCATIONS = {
    "Tarbela (Indus River)": (34.09, 72.70),
    "Mangla (Jhelum River)": (33.13, 73.64),
    "Kalabagh (Indus River)": (32.96, 71.55),
}

location_name = st.sidebar.selectbox("Choose a gauge location", list(LOCATIONS.keys()))
LATITUDE, LONGITUDE = LOCATIONS[location_name]
st.sidebar.write(f"Lat: {LATITUDE}, Lon: {LONGITUDE}")

DAYS_OF_HISTORY = st.sidebar.slider("Days of history to pull", 30, 90, 45)

# ---------------------------------------------------------------------------
# Fetch fresh data (last N days) — same two free APIs used in training
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)  # refresh once per hour
def fetch_weather(lat, lon, days):
    end = datetime.utcnow().date()
    start = end - timedelta(days=days)
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters=PRECTOTCORR,T2M,RH2M,WS10M"
        f"&community=AG&longitude={lon}&latitude={lat}"
        f"&start={start.strftime('%Y%m%d')}&end={end.strftime('%Y%m%d')}&format=CSV"
    )
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    raw_lines = response.text.splitlines()
    header_idx = next(i for i, line in enumerate(raw_lines) if line.startswith("YEAR"))
    csv_data = "\n".join(raw_lines[header_idx:])
    weather_df = pd.read_csv(io.StringIO(csv_data))
    weather_df["date"] = pd.to_datetime(
        weather_df["YEAR"].astype(str) + weather_df["DOY"].astype(str).str.zfill(3),
        format="%Y%j"
    )
    weather_df = weather_df.rename(columns={
        "PRECTOTCORR": "rainfall_mm", "T2M": "temperature_c",
        "RH2M": "humidity_pct", "WS10M": "wind_speed_ms"
    })[["date", "rainfall_mm", "temperature_c", "humidity_pct", "wind_speed_ms"]]
    return weather_df


@st.cache_data(ttl=3600)
def fetch_discharge(lat, lon, days):
    end = datetime.utcnow().date()
    start = end - timedelta(days=days)
    url = (
        "https://flood-api.open-meteo.com/v1/flood"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start.isoformat()}&end_date={end.isoformat()}"
        f"&daily=river_discharge"
    )
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()
    return pd.DataFrame({
        "date": pd.to_datetime(data["daily"]["time"]),
        "river_discharge": data["daily"]["river_discharge"]
    })


with st.spinner("Pulling latest rainfall, weather, and river discharge data..."):
    weather_df = fetch_weather(LATITUDE, LONGITUDE, DAYS_OF_HISTORY)
    discharge_df = fetch_discharge(LATITUDE, LONGITUDE, DAYS_OF_HISTORY)

df = weather_df.merge(discharge_df, on="date", how="left").sort_values("date").reset_index(drop=True)
df = df.set_index("date").interpolate(method="time", limit=3).reset_index()
df = df.dropna().reset_index(drop=True)

# ---------------------------------------------------------------------------
# Rebuild the exact same features used in training
# ---------------------------------------------------------------------------
for lag in [1, 3, 7]:
    df[f"rainfall_lag_{lag}"] = df["rainfall_mm"].shift(lag)

for window in [3, 7, 14]:
    df[f"rainfall_roll_sum_{window}"] = df["rainfall_mm"].rolling(window).sum()

df["rainfall_intensity"] = df["rainfall_mm"] / (df["rainfall_mm"].rolling(7).mean() + 1e-6)
df["consecutive_rainy_days"] = (df["rainfall_mm"] > 1).astype(int).groupby(
    (df["rainfall_mm"] <= 1).cumsum()
).cumsum()

df["discharge_change_1d"] = df["river_discharge"].diff(1)
df["discharge_change_3d"] = df["river_discharge"].diff(3)
df["discharge_change_7d"] = df["river_discharge"].diff(7)
df["discharge_roll_mean_7"] = df["river_discharge"].rolling(7).mean()
df["discharge_roll_max_7"] = df["river_discharge"].rolling(7).max()
df["discharge_roll_max_14"] = df["river_discharge"].rolling(14).max()
df["rainfall_roll_sum_21"] = df["rainfall_mm"].rolling(21).sum()

df["month"] = df["date"].dt.month
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# ---------------------------------------------------------------------------
# Predict flood risk for the most recent complete row
# ---------------------------------------------------------------------------
df_ready = df.dropna(subset=[c for c in FEATURE_COLUMNS if c in df.columns]).reset_index(drop=True)

st.title("🌊 Flood Risk Forecasting — Indus Basin")
st.caption(f"Location: {location_name} | Forecast horizon: next {FORECAST_HORIZON_DAYS} days | Data auto-refreshes hourly")

if df_ready.empty:
    st.error("Not enough recent data to compute a prediction yet — try increasing 'Days of history' in the sidebar.")
else:
    latest_row = df_ready.iloc[[-1]]
    missing_cols = [c for c in FEATURE_COLUMNS if c not in latest_row.columns]
    if missing_cols:
        st.error(f"Feature mismatch — missing columns: {missing_cols}")
    else:
        X_latest = latest_row[FEATURE_COLUMNS]
        flood_prob = model.predict_proba(X_latest)[0, 1]
        risk_flag = flood_prob > THRESHOLD

        if flood_prob < THRESHOLD * 0.5:
            risk_label, color = "Low", "green"
        elif flood_prob < THRESHOLD:
            risk_label, color = "Medium", "orange"
        else:
            risk_label, color = "High", "red"

        col1, col2, col3 = st.columns(3)
        col1.metric("Flood Probability", f"{flood_prob:.1%}")
        col2.metric("Decision Threshold", f"{THRESHOLD:.2f}")
        col3.markdown(f"### Risk Level: :{color}[{risk_label}]")

        st.markdown("---")

# ---------------------------------------------------------------------------
# Live graphs
# ---------------------------------------------------------------------------
st.subheader("📈 Recent Rainfall")
fig1 = px.line(df, x="date", y="rainfall_mm", labels={"rainfall_mm": "Rainfall (mm)"})
st.plotly_chart(fig1, use_container_width=True)

st.subheader("🌊 Recent River Discharge")
fig2 = px.line(df, x="date", y="river_discharge", labels={"river_discharge": "Discharge (m³/s)"})
st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------------------
# 7-day probability trend (using the last available rows, not a true future forecast —
# river/weather forecasts beyond ~a week need a separate forecast API to extend further)
# ---------------------------------------------------------------------------
st.subheader(f"📅 Last {min(7, len(df_ready))} Days — Flood Probability Trend")
if not df_ready.empty:
    recent = df_ready.tail(7).copy()
    recent["flood_probability"] = model.predict_proba(recent[FEATURE_COLUMNS])[:, 1]
    fig3 = px.bar(recent, x="date", y="flood_probability", range_y=[0, 1])
    fig3.add_hline(y=THRESHOLD, line_dash="dash", line_color="red",
                    annotation_text="Decision threshold")
    st.plotly_chart(fig3, use_container_width=True)

# ---------------------------------------------------------------------------
# Model comparison (static — from your notebook's evaluation results)
# ---------------------------------------------------------------------------
st.subheader("📊 Model Comparison (from training/evaluation)")
comparison_df = pd.DataFrame([
    {"Model": "Random Forest", "Recall": 0.70, "Precision": 0.52, "F1": 0.60, "ROC-AUC": 0.80},
    {"Model": "LightGBM", "Recall": 0.69, "Precision": 0.48, "F1": 0.57, "ROC-AUC": 0.79},
    {"Model": "XGBoost (tuned, deployed)", "Recall": 1.00, "Precision": 0.37, "F1": 0.54, "ROC-AUC": 0.81},
])
st.dataframe(comparison_df, use_container_width=True)
st.caption(
    "XGBoost (tuned) is the deployed model — chosen for maximum recall (catches all "
    "known flood events in testing), prioritizing missed-flood risk over false alarms, "
    "which is standard practice for early-warning systems."
)

st.markdown("---")
st.caption(
    "Data sources: NASA POWER (rainfall, temperature, humidity, wind) and the Open-Meteo "
    "Flood API (river discharge, based on the GloFAS reanalysis). Both are free, public APIs."
)
