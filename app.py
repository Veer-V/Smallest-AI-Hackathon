# 🌐 Global Air Quality Dashboard with XGBoost + Random Forest Ensemble

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
from datetime import datetime

# 🚀 Constants
CO2_PM25_FACTOR = 0.035
CO2_CO_FACTOR = 10

# 🎨 Color coding for PM2.5
def pm25_color_code(pm25):
    if pm25 <= 50: return "green"
    elif pm25 <= 100: return "yellow"
    elif pm25 <= 150: return "orange"
    elif pm25 <= 200: return "red"
    elif pm25 <= 300: return "purple"
    else: return "maroon"

# 💨 CO₂ Estimation
def estimate_co2(pm25, co):
    return pm25 * CO2_PM25_FACTOR + co * CO2_CO_FACTOR

@st.cache_data
def load_and_merge_data():
    df1 = pd.read_csv("air_quality_dataset.csv", parse_dates=['timestamp'], dayfirst=True)
    df2 = pd.read_csv("global_air_quality_dataset.csv", parse_dates=['timestamp'])

    # Clean missing data with updated syntax
    df1 = df1.ffill().bfill()
    df2 = df2.ffill().bfill()

    # Ensure required location columns exist
    for col in ['country', 'state', 'city']:
        if col not in df1.columns:
            df1[col] = 'Unknown'
        if col not in df2.columns:
            df2[col] = 'Unknown'

    # CO₂ Estimation
    df1['estimated_CO2'] = estimate_co2(df1['PM2.5'], df1['CO'])
    df2['estimated_CO2'] = estimate_co2(df2['PM2.5'], df2['CO'])

    # Add year/month
    for df in [df1, df2]:
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month

    # Match column order before merging
    df2 = df2[df1.columns]  # Align columns
    combined_df = pd.concat([df1, df2], ignore_index=True)

    return combined_df

@st.cache_resource
def train_ensemble_model(df):
    features = df[['temperature', 'humidity', 'pressure', 'NO2', 'O3', 'CO']]
    target = df['PM2.5']

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, target, test_size=0.2, random_state=42
    )

    rf = RandomForestRegressor(n_estimators=150, random_state=42)
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6,
                       subsample=0.8, colsample_bytree=0.8, random_state=42)

    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    rf_pred = rf.predict(X_test)
    xgb_pred = xgb.predict(X_test)
    ensemble_pred = (rf_pred + xgb_pred) / 2

    rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
    r2 = r2_score(y_test, ensemble_pred)

    return rf, xgb, scaler, rmse, r2

# 🧠 Streamlit App
def main():
    st.set_page_config(page_title="🌐 Air Quality Ensemble Dashboard", layout="wide")
    st.title("🌍 Global Air Quality & CO₂ Intelligence Dashboard")
    st.markdown("##### Ensemble Model: XGBoost + Random Forest for Better PM2.5 Prediction Accuracy")

    df = load_and_merge_data()
    rf_model, xgb_model, scaler, rmse, r2 = train_ensemble_model(df)

    st.sidebar.header("🌐 Select Location")
    country = st.sidebar.selectbox("Country", sorted(df['country'].dropna().unique()))
    state = st.sidebar.selectbox("State", sorted(df[df['country'] == country]['state'].dropna().unique()))
    city = st.sidebar.selectbox("City", sorted(df[(df['country'] == country) & (df['state'] == state)]['city'].dropna().unique()))
    filtered_df = df[(df['country'] == country) & (df['state'] == state) & (df['city'] == city)]

    st.markdown(f"### 📍 Air Quality in {city}, {state}, {country}")

    if filtered_df.empty:
        st.warning("⚠️ No data available for this selection.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.line(filtered_df, x='timestamp', y='PM2.5', title='PM2.5 Over Time', markers=True), use_container_width=True)
        with col2:
            st.plotly_chart(px.line(filtered_df, x='timestamp', y='estimated_CO2', title='Estimated CO₂ Over Time', markers=True), use_container_width=True)

        st.markdown("### 🌏 Country-wise Annual CO₂ Emissions")
        summary_df = df.groupby(['country', 'year'])['estimated_CO2'].mean().reset_index()
        st.plotly_chart(px.bar(summary_df, x='year', y='estimated_CO2', color='country', title='Average CO₂ by Country per Year'), use_container_width=True)

    st.markdown(f"### 🎯 Model Performance")
    st.metric(label="Ensemble RMSE", value=f"{rmse:.2f}")
    st.metric(label="Ensemble R² Score", value=f"{r2:.4f}")

    st.markdown("### 🧪 Predict PM2.5 and CO₂ for Custom Inputs")
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            temp = st.number_input("Temperature (°C)", value=25.0)
            humidity = st.number_input("Humidity (%)", value=50.0)
        with col2:
            pressure = st.number_input("Pressure (hPa)", value=1013.0)
            no2 = st.number_input("NO2 (ppb)", value=20.0)
        with col3:
            o3 = st.number_input("O3 (ppb)", value=20.0)
            co = st.number_input("CO (ppm)", value=1.0)

        submitted = st.form_submit_button("Predict")

        if submitted:
            input_data = np.array([[temp, humidity, pressure, no2, o3, co]])
            scaled_input = scaler.transform(input_data)
            rf_pred = rf_model.predict(scaled_input)[0]
            xgb_pred = xgb_model.predict(scaled_input)[0]
            pm25_pred = (rf_pred + xgb_pred) / 2
            co2_pred = estimate_co2(pm25_pred, co)
            color = pm25_color_code(pm25_pred)

            st.markdown(f"**Predicted PM2.5:** <span style='color:{color}; font-weight:bold;'>{pm25_pred:.2f} µg/m³</span>", unsafe_allow_html=True)
            st.markdown(f"**Estimated CO₂ Emission:** {co2_pred:.2f} kg")

if __name__ == "__main__":
    main()
