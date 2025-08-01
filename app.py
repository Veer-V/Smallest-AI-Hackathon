# air_quality_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
from datetime import datetime

# --- Constants ---
CO2_PM25_FACTOR = 0.035
CO2_CO_FACTOR = 10

def pm25_color_code(pm25):
    if pm25 <= 50: return "green"
    elif pm25 <= 100: return "yellow"
    elif pm25 <= 150: return "orange"
    elif pm25 <= 200: return "red"
    elif pm25 <= 300: return "purple"
    else: return "maroon"

def estimate_co2(pm25, co):
    return pm25 * CO2_PM25_FACTOR + co * CO2_CO_FACTOR

def enrich(df):
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df = df[(df['PM2.5'] > 0) & (df['PM2.5'] < 1000)]
    df['month'] = df['timestamp'].dt.month
    df['hour'] = df['timestamp'].dt.hour
    df['weekday'] = df['timestamp'].dt.weekday
    df['year'] = df['timestamp'].dt.year
    df['season'] = df['month'].apply(lambda m: 'Winter' if m in [12,1,2] else 'Summer' if m in [6,7,8] else 'Transition')
    df['is_weekend'] = df['weekday'] >= 5
    df['humidity_pressure_ratio'] = df['humidity'] / (df['pressure'] + 1)
    df['estimated_CO2'] = estimate_co2(df['PM2.5'], df['CO'])
    for col in ['country', 'state', 'city']:
        if col not in df.columns:
            df[col] = 'Unknown'
    return df

@st.cache_data
def load_and_prepare_data():
    df1 = pd.read_csv("air_quality_dataset.csv", parse_dates=['timestamp'], dayfirst=True)
    df2 = pd.read_csv("global_air_quality_dataset.csv", parse_dates=['timestamp'])
    df1 = enrich(df1)
    df2 = enrich(df2)
    combined = pd.concat([df1, df2], ignore_index=True)
    combined = pd.get_dummies(combined, columns=['season'], drop_first=False)
    for col in ['season_Summer', 'season_Winter']:
        if col not in combined.columns:
            combined[col] = 0
    return combined

@st.cache_resource
def train_stacked_model(df):
    features = ['temperature', 'humidity', 'pressure', 'NO2', 'O3', 'CO',
                'month', 'hour', 'weekday', 'is_weekend',
                'humidity_pressure_ratio', 'season_Summer', 'season_Winter']
    target = np.log1p(df['PM2.5'])
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    expanded = poly.fit_transform(df[features])
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(expanded)

    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, target, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=350, max_depth=14, random_state=42)
    xgb = XGBRegressor(n_estimators=350, learning_rate=0.02, max_depth=9,
                       subsample=0.85, colsample_bytree=0.85, random_state=42)
    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    stacked_input = np.column_stack((rf.predict(X_test), xgb.predict(X_test)))
    meta_model = Ridge(alpha=0.5)
    meta_model.fit(stacked_input, y_test)

    final_log_pred = meta_model.predict(stacked_input)
    final_pred = np.expm1(final_log_pred)
    actual = np.expm1(y_test)
    rmse = np.sqrt(mean_squared_error(actual, final_pred))
    r2 = r2_score(actual, final_pred)

    return rf, xgb, meta_model, scaler, poly, rmse, r2

def main():
    st.set_page_config(page_title="🌐 Enhanced Air Quality Dashboard", layout="wide")

    # Inject custom CSS for improved UI
    st.markdown(
        """
        <style>
        /* General body styles */
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f0f2f6;
            color: #333333;
        }
        /* Title style */
        .css-1v3fvcr h1 {
            color: #2c3e50;
            font-weight: 700;
            font-size: 3rem;
            margin-bottom: 0.5rem;
        }
        /* Sidebar header */
        .css-1d391kg h2 {
            color: #2980b9;
            font-weight: 600;
        }
        /* Sidebar selectbox */
        .css-1v3fvcr select {
            border-radius: 8px;
            border: 1px solid #2980b9;
            padding: 0.3rem;
            font-size: 1rem;
        }
        /* Metric styles */
        .stMetric {
            background-color: #3498db;
            color: white;
            border-radius: 10px;
            padding: 1rem;
            margin: 0.5rem 0;
            font-size: 1.5rem;
            font-weight: 600;
            text-align: center;
        }
        /* Button style */
        div.stButton > button {
            background-color: #2980b9;
            color: white;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-size: 1.1rem;
            font-weight: 600;
            border: none;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        div.stButton > button:hover {
            background-color: #1c5980;
        }
        /* Form input styles */
        input[type=number] {
            border-radius: 6px;
            border: 1px solid #2980b9;
            padding: 0.3rem;
            font-size: 1rem;
            width: 100%;
        }
        /* Plotly chart container */
        .js-plotly-plot {
            border-radius: 12px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            background-color: white;
            padding: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("🌍 🌍 Global Air Quality & CO₂ Intelligence Dashboard (XGBoost + RF)")
    st.markdown("Now with interaction features and smarter stacking using Ridge regression.")

    df = load_and_prepare_data()
    rf_model, xgb_model, meta_model, scaler, poly, rmse, r2 = train_stacked_model(df)

    # UI: Location filter
    st.sidebar.header("📍 Location Filter")
    country = st.sidebar.selectbox("Country", sorted(df['country'].unique()))
    state = st.sidebar.selectbox("State", sorted(df[df['country'] == country]['state'].unique()))
    city = st.sidebar.selectbox("City", sorted(df[(df['country'] == country) & (df['state'] == state)]['city'].unique()))
    filtered_df = df[(df['country'] == country) & (df['state'] == state) & (df['city'] == city)]

    st.markdown(f"### 📊 Air Quality in {city}, {state}, {country}")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.line(filtered_df, x='timestamp', y='PM2.5', title="PM2.5 Trend"), use_container_width=True)
    with col2:
        st.plotly_chart(px.line(filtered_df, x='timestamp', y='estimated_CO2', title="Estimated CO₂ Emission"), use_container_width=True)

    st.markdown("### 🌍 Country-wise Annual CO₂")
    summary_df = df.groupby(['country', 'year'])['estimated_CO2'].mean().reset_index()
    st.plotly_chart(px.bar(summary_df, x='year', y='estimated_CO2', color='country'), use_container_width=True)

    st.markdown("### 🧠 Model Performance")
    st.metric(label="Improved RMSE", value=f"{rmse:.2f}")
    st.metric(label="Improved R² Score", value=f"{r2:.4f}")

    st.markdown("### 🔍 Predict Custom PM2.5 & CO₂")
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            temp = st.number_input("Temperature (°C)", value=25.0)
            humidity = st.number_input("Humidity (%)", value=50.0)
            month = st.number_input("Month", 1, 12, value=datetime.now().month)
        with col2:
            pressure = st.number_input("Pressure (hPa)", value=1013.0)
            no2 = st.number_input("NO2 (ppb)", value=20.0)
            hour = st.number_input("Hour", 0, 23, value=datetime.now().hour)
        with col3:
            o3 = st.number_input("O3 (ppb)", value=20.0)
            co = st.number_input("CO (ppm)", value=1.0)
            weekday = datetime.now().weekday()
            season_summer = 1 if month in [6, 7, 8] else 0
            season_winter = 1 if month in [12, 1, 2] else 0
            is_weekend = 1 if weekday >= 5 else 0

        hpr = humidity / (pressure + 1)
        submitted = st.form_submit_button("Predict")

        if submitted:
            base_input = np.array([[temp, humidity, pressure, no2, o3, co, month, hour, weekday,
                                    is_weekend, hpr, season_summer, season_winter]])
            expanded_input = poly.transform(base_input)
            scaled_input = scaler.transform(expanded_input)
            rf_pred = rf_model.predict(scaled_input)[0]
            xgb_pred = xgb_model.predict(scaled_input)[0]
            stacked_input = np.column_stack(([rf_pred], [xgb_pred]))
            final_log_pred = meta_model.predict(stacked_input)[0]
            pm25_pred = np.expm1(final_log_pred)
            co2_pred = estimate_co2(pm25_pred, co)
            color = pm25_color_code(pm25_pred)

            st.markdown(f"**Predicted PM2.5:** <span style='color:{color}; font-weight:bold;'>{pm25_pred:.2f} µg/m³</span>", unsafe_allow_html=True)
            st.markdown(f"**Estimated CO₂ Emission:** {co2_pred:.2f} kg")

if __name__ == "__main__":
    main()
