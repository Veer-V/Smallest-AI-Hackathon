# Global Air Quality & CO₂ Intelligence Dashboard

## Project Overview
This project is a Streamlit-based interactive dashboard that provides insights into global air quality and CO₂ emissions. It leverages two comprehensive datasets to analyze air pollution metrics such as PM2.5, CO, NO2, and O3 across various countries, states, and cities. The dashboard also includes an ensemble machine learning model to predict PM2.5 levels based on environmental factors.

## Data Sources
- **air_quality_dataset.csv**: Local air quality measurements with timestamps and pollutant concentrations.
- **global_air_quality_dataset.csv**: Global air quality data with similar features.
  
Both datasets are merged and preprocessed to fill missing values and harmonize columns.

## Data Preprocessing
- Forward and backward filling of missing data.
- Addition of estimated CO₂ emissions calculated from PM2.5 and CO concentrations.
- Extraction of year and month from timestamps for temporal analysis.

## Machine Learning Model
- Ensemble model combining Random Forest and XGBoost regressors.
- Features include temperature, humidity, pressure, NO2, O3, CO, month, year, and engineered features such as humidity-temperature interaction and NO2/O3 ratio.
- Target variable is log-transformed PM2.5 concentration.
- Model performance metrics:
  - Root Mean Squared Error (RMSE)
  - R² Score

## Dashboard Features
- **Location Selection**: Sidebar dropdowns to select country, state, and city.
- **Time Series Visualizations**:
  - PM2.5 levels over time.
  - Estimated CO₂ emissions over time.
- **Country-wise Annual CO₂ Emissions**: Bar chart showing average CO₂ emissions by country per year.
- **Prediction Form**: Input environmental parameters to predict PM2.5 and estimated CO₂ emissions.
- **Performance Metrics**: Display of ensemble model RMSE and R² score.

## Technologies Used
- Python 3.x
- Streamlit for interactive web app
- Pandas and NumPy for data manipulation
- Scikit-learn and XGBoost for machine learning
- Plotly Express for interactive visualizations

## How to Use
1. Run the Streamlit app (`streamlit run app.py`).
2. Use the sidebar to select a location (country, state, city).
3. View time series charts and CO₂ emissions for the selected location.
4. Use the prediction form to input custom environmental data and get PM2.5 and CO₂ predictions.
5. Analyze model performance metrics displayed on the dashboard.

## Future Enhancements
- Incorporate additional pollutants and environmental factors.
- Add more granular geographic selections.
- Include real-time data updates.
- Enhance interactivity with additional charts such as pie charts for pollutant composition.

## Contact
For questions or contributions, please contact the project maintainer.

---
This documentation provides a comprehensive overview suitable for presentation slides summarizing the project.
