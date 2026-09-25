# HeatShield AI 🌡️🌳

## AI-Powered Urban Heat Monitoring & Cooling Planner

HeatShield AI is a prototype system that uses satellite data and machine learning to analyze urban heat patterns and explore possible cooling interventions.

The project focuses on **Kochi, Kerala**.

## 🎯 Problem

Urban areas can become significantly warmer because of:

- Dense buildings and concrete surfaces
- Reduced vegetation
- Heat-absorbing urban surfaces
- Limited natural cooling

HeatShield AI aims to help identify areas experiencing higher surface temperatures and understand their relationship with vegetation.

## 🛰️ Data

The prototype uses:

- **Landsat 8/9** satellite data
- **Land Surface Temperature (LST)**
- **NDVI (Normalized Difference Vegetation Index)**

Around **3,000 locations** were sampled from the study area.

## 🤖 Machine Learning

A **Random Forest Regression** model is used to analyze the relationship between vegetation and surface temperature.

### Current Model

- Input: NDVI
- Target: Surface Temperature
- Training samples: 2,998
- Model: Random Forest Regressor
- Trees: 100

### Results

- MAE: **~1.06°C**
- R²: **~0.64**

These results represent the current prototype model and should not be interpreted as a fully validated city-scale prediction system.

## 📊 Dashboard

The Streamlit dashboard provides:

- 🌡️ Surface temperature statistics
- 🗺️ Urban heat map
- 🌳 Vegetation analysis
- 🤖 Machine learning results
- ❄️ Cooling intervention simulator
- 💡 AI-based intervention suggestions

## 🔄 System Pipeline

```text
Satellite Data
      ↓
Temperature + NDVI
      ↓
Data Sampling
      ↓
Machine Learning
      ↓
Heat Analysis
      ↓
Cooling Planner
      ↓
Interactive Dashboard
