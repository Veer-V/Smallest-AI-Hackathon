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

@st.cache_data
def load_and_merge_data():
    df1 = pd.read_csv("air_quality_dataset.csv", parse_dates=['timestamp'], dayfirst=True)
    df2 = pd.read_csv("global_air_quality_dataset.csv", parse_dates=['timestamp'])

    df1 = df1.ffill().bfill()
    df2 = df2.ffill().bfill()

    for col in ['country', 'state', 'city']:
        if col not in df1.columns:
            df1[col] = 'Unknown'
        if col not in df2.columns:
            df2[col] = 'Unknown'

    df1['estimated_CO2'] = estimate_co2(df1['PM2.5'], df1['CO'])
    df2['estimated_CO2'] = estimate_co2(df2['PM2.5'], df2['CO'])

    for df in [df1, df2]:
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month

    df2 = df2[df1.columns]
    combined_df = pd.concat([df1, df2], ignore_index=True)
    return combined_df

@st.cache_resource
def train_ensemble_model(df):
    df = df.copy()
    df = df[df["PM2.5"] > 0]
    df['humidity_temp'] = df['humidity'] * df['temperature']
    df['NO2_O3_ratio'] = df['NO2'] / (df['O3'] + 1)

    features = df[['temperature', 'humidity', 'pressure', 'NO2', 'O3', 'CO', 'month', 'year', 'humidity_temp', 'NO2_O3_ratio']]
    target = np.log1p(df['PM2.5'])

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42)
    xgb = XGBRegressor(n_estimators=200, learning_rate=0.03, max_depth=6, subsample=0.9, colsample_bytree=0.9, random_state=42)

    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    rf_pred = rf.predict(X_test)
    xgb_pred = xgb.predict(X_test)
    ensemble_pred = 0.4 * rf_pred + 0.6 * xgb_pred

    rmse = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(ensemble_pred)))
    r2 = r2_score(np.expm1(y_test), np.expm1(ensemble_pred))

    return rf, xgb, scaler, features.columns.tolist(), rmse, r2

def main():
    st.set_page_config(page_title="🌐 Air Quality Ensemble Dashboard", layout="wide")
    st.title("🌍 Global Air Quality & CO₂ Intelligence Dashboard")
    st.markdown("##### Ensemble Model: XGBoost + Random Forest with Feature Engineering")

    df = load_and_merge_data()
    rf_model, xgb_model, scaler, feature_names, rmse, r2 = train_ensemble_model(df)

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

        st.markdown("### 🧭 Air Quality Pie Chart (CO₂ vs Oxygen Proxy)")
        pie_data = filtered_df[['estimated_CO2', 'O3']].mean()
        pie_df = pd.DataFrame({
            'Type': ['CO₂', 'Oxygen (via O₃)'],
            'Value': [pie_data['estimated_CO2'], pie_data['O3']]
        })
        fig = px.pie(pie_df, values='Value', names='Type',
                     title=f"Air Composition in {city}, {state}, {country}",
                     color_discrete_map={'CO₂': 'red', 'Oxygen (via O₃)': 'green'})
        st.plotly_chart(fig, use_container_width=True)

        def get_aqi_label(pm25):
            if pm25 <= 50: return "Good"
            elif pm25 <= 100: return "Moderate"
            elif pm25 <= 150: return "Unhealthy for Sensitive Groups"
            elif pm25 <= 200: return "Unhealthy"
            elif pm25 <= 300: return "Very Unhealthy"
            else: return "Hazardous"

        st.markdown("### 🧾 Air Quality Rating by Location")
        filtered_df['AQI Category'] = filtered_df['PM2.5'].apply(get_aqi_label)
        aqi_summary = filtered_df['AQI Category'].value_counts().reset_index()
        aqi_summary.columns = ['AQI Level', 'Count']
        fig_bar = px.bar(aqi_summary, x='AQI Level', y='Count', color='AQI Level',
                         color_discrete_map={
                             "Good": "green", "Moderate": "yellow", "Unhealthy for Sensitive Groups": "orange",
                             "Unhealthy": "red", "Very Unhealthy": "purple", "Hazardous": "maroon"
                         },
                         title=f"Air Quality Levels in {city}, {state}, {country}")
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown(f"### 🗺️ AQI Distribution per City in {country}")
        country_df = df[df['country'] == country].copy()
        country_df['AQI Category'] = country_df['PM2.5'].apply(get_aqi_label)
        city_aqi_counts = country_df.groupby(['city', 'AQI Category']).size().reset_index(name='Count')
        fig_country_aqi = px.bar(city_aqi_counts, x="city", y="Count", color="AQI Category",
                                 title=f"AQI Distribution per City in {country}", barmode='group',
                                 color_discrete_map={
                                     "Good": "green", "Moderate": "yellow", "Unhealthy for Sensitive Groups": "orange",
                                     "Unhealthy": "red", "Very Unhealthy": "purple", "Hazardous": "maroon"
                                 })
        st.plotly_chart(fig_country_aqi, use_container_width=True)

        # 🌐 Global Maps
        st.markdown("### 🌍 Global PM2.5 Air Quality Map")
        country_pm25 = df.groupby('country')['PM2.5'].mean().reset_index()
        fig_pm25_map = px.choropleth(country_pm25, locations='country', locationmode='country names',
                                     color='PM2.5', color_continuous_scale='YlOrRd',
                                     title='Average PM2.5 Levels by Country')
        st.plotly_chart(fig_pm25_map, use_container_width=True)

        st.markdown("### 🌍 Global Estimated CO₂ Emission Map")
        country_co2 = df.groupby('country')['estimated_CO2'].mean().reset_index()
        fig_co2_map = px.choropleth(country_co2, locations='country', locationmode='country names',
                                    color='estimated_CO2', color_continuous_scale='Plasma',
                                    title='Average Estimated CO₂ Emission by Country')
        st.plotly_chart(fig_co2_map, use_container_width=True)

    st.markdown("### 🎯 Model Performance")
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
            month = datetime.now().month
            year = datetime.now().year
            humidity_temp = humidity * temp
            no2_o3_ratio = no2 / (o3 + 1)

            input_array = pd.DataFrame([[temp, humidity, pressure, no2, o3, co, month, year, humidity_temp, no2_o3_ratio]],
                                       columns=feature_names)
            scaled_input = scaler.transform(input_array)
            rf_pred = rf_model.predict(scaled_input)[0]
            xgb_pred = xgb_model.predict(scaled_input)[0]
            pm25_pred = np.expm1(0.4 * rf_pred + 0.6 * xgb_pred)
            co2_pred = estimate_co2(pm25_pred, co)
            color = pm25_color_code(pm25_pred)

            st.markdown(f"**Predicted PM2.5:** <span style='color:{color}; font-weight:bold;'>{pm25_pred:.2f} µg/m³</span>", unsafe_allow_html=True)
            st.markdown(f"**Estimated CO₂ Emission:** {co2_pred:.2f} kg")

if __name__ == "__main__":
    main()
