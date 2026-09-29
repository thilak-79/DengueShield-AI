# DengueShield AI

**Explainable Dengue Activity Intelligence & Decision-Support Platform for Sri Lanka**

DengueShield AI is a machine-learning research and decision-support prototype for forecasting short-term dengue activity across Sri Lanka's 25 districts.

The system combines historical dengue surveillance, meteorological observations, temporal feature engineering, machine-learning models, walk-forward validation, SHAP explainability, a FastAPI backend, and an interactive React dashboard.

> **Important:** DengueShield AI is a research prototype. It is not a medical diagnostic system and is not an official Ministry of Health or National Dengue Control Unit forecasting or alert platform.

---

## Research Question

**Can machine-learning models combining historical dengue surveillance and meteorological data forecast short-term dengue activity across Sri Lankan districts?**

---

## System Modes

The dashboard deliberately separates two operating modes.

### Historical Evaluation

Used for:

- Historical model evaluation
- 2020–2025 walk-forward validation
- Model comparison
- Historical district forecasts
- Historical SHAP explanations
- Statistical relative-activity analysis

Historical model development uses **WER-aligned dengue surveillance**.

### Experimental Current 2026

Used for:

- Current NDCU surveillance
- Recent weather observations
- Experimental 1-week Random Forest inference
- Current district activity categories
- Current SHAP explanations

Current outputs are explicitly labeled:

**EXPERIMENTAL CURRENT INFERENCE**

because the current NDCU surveillance source and the historical WER-aligned training source have different reporting semantics.

---

## Main Features

- Forecasting for all 25 Sri Lankan districts
- 1-week, 2-week, and 4-week research horizons
- Persistence baseline
- Random Forest
- XGBoost
- LightGBM
- Chronological train/validation/test design
- 2020–2025 multi-year walk-forward evaluation
- Leakage-safe feature engineering
- SHAP global explanations
- SHAP district-level explanations
- Statistical relative-activity categories
- Forecast trend detection
- Sri Lanka choropleth map
- Historical/current mode switching
- Experimental NDCU 2026 inference
- FastAPI REST API
- React/Vite dashboard
- Automated validation tests

---

# Architecture

```text
                    HISTORICAL RESEARCH PIPELINE

        WER-Aligned Dengue Surveillance
                       +
              Historical Weather
                       |
                       v
              Feature Engineering
                       |
                       v
       Persistence / RF / XGBoost / LightGBM
                       |
                       v
          Chronological Walk-Forward
                 Evaluation
                       |
                       v
                SHAP Analysis
                       |
                       v
                FastAPI API
                       |
                       v
              React Dashboard
```

Experimental current inference:

```
                 CURRENT 2026 PIPELINE

          NDCU Weekly Dengue Update
                       +
              Recent Weather Data
                       |
                       v
             Current Data Audit
                       |
                       v
          Current Feature Generation
                       |
                       v
        Locked Random Forest Configuration
             Full-History Refit
                       |
                       v
          Experimental 1-Week Forecast
                       |
                       v
        Relative Activity + SHAP
                       |
                       v
              FastAPI / React
```

---

# Data Sources

## Historical Dengue Surveillance

Historical dengue surveillance is based on Sri Lankan district-level data aligned with Weekly Epidemiological Reports (WER), obtained through DengueDataHub.

Coverage:

- 2006–2025
- 25 geographic districts
- Weekly dengue case observations

Primary processed file:

```
data/processed/dengue_weekly_2006_2025.csv
```

---

## Historical Weather

Historical meteorological data was retrieved through Open-Meteo.

Variables include:

- Mean temperature
- Minimum temperature
- Maximum temperature
- Relative humidity
- Rainfall
- Number of rain days

Weather coverage used by the ML pipeline:

**2015–2025**

Processed file:

```
data/processed/weather_weekly_2015_2025.csv
```

Combined dengue-weather file:

```
data/processed/dengue_weather_weekly_2015_2025.csv
```

---

## Current 2026 Surveillance

Experimental current inference uses:

**National Dengue Control Unit (NDCU) Weekly Dengue Update**

Current prototype status:

- Year: 2026
- Latest week: Week 37
- Surveillance window: 07–13 September 2026
- Districts: 25
- National reported cases: 1,156

Forecast target:

**14–20 September 2026**

---

# Feature Engineering

The historical ML feature dataset contains approximately:

- 14,225 district-week rows
- 46 total columns

The Random Forest receives:

- 35 raw model inputs
- 34 numeric features
- 1 categorical district feature

Important features include:

### Dengue Activity

- Current dengue cases
- Cases lagged 1–4 weeks
- Recent 2-week case average
- Recent 4-week case average

### Weather

- Current temperature
- Current minimum temperature
- Current maximum temperature
- Current humidity
- Current rainfall
- Current rain-day count
- Weather lags
- Rolling weather summaries

### Geography

- District identity

### Seasonality

Seasonality is calculated using the midpoint day-of-year of each surveillance interval.

```
midpoint = start_date + (end_date - start_date) / 2

week_sin = sin(2π × day_of_year / 365.25)
week_cos = cos(2π × day_of_year / 365.25)
```

The implementation keeps the names `week_sin` and `week_cos` for compatibility.

---

# Models Evaluated

The research compares:

- Persistence baseline
- Random Forest
- XGBoost
- LightGBM

Random train/test splitting is not used.

Instead, model evaluation uses chronological target-date-based splits and expanding-window walk-forward validation.

---

# Evaluation Metrics

Primary metric:

- **MAE — Mean Absolute Error**

Secondary metrics:

- RMSE — Root Mean Squared Error
- sMAPE — Symmetric Mean Absolute Percentage Error

Lower values are better.

---

# Walk-Forward Evaluation

The main multi-year evaluation covers:

**2020–2025**

## 1-Week Horizon

Random Forest:

- Mean MAE ≈ **10.85**
- Median MAE ≈ **9.31**
- Mean RMSE ≈ **22.61**
- Mean sMAPE ≈ **56.8%**

Persistence:

- Mean MAE ≈ **10.96**

Random Forest achieved lower MAE in:

- 2022
- 2024
- 2025

Persistence achieved lower MAE in:

- 2020
- 2021
- 2023

Therefore Random Forest beat persistence in:

**3 of 6 evaluation years**

The average Random Forest advantage at the 1-week horizon is small and year-dependent.

---

## Longer Horizons

At the 2-week and 4-week horizons, persistence achieved lower mean MAE than both Random Forest and LightGBM across the six walk-forward evaluation years.

This shows that the persistence baseline remains highly competitive, especially for longer forecasting horizons.

---

# 2025 Historical Evaluation

Selected 2025 results:

| Horizon | Model         | MAE   | RMSE  | sMAPE  |
| ------- | ------------- | ----- | ----- | ------ |
| 1 Week  | Random Forest | 8.58  | 15.43 | 38.90% |
| 2 Weeks | Random Forest | 9.96  | 17.72 | 42.38% |
| 4 Weeks | Random Forest | 13.08 | 24.70 | 47.69% |
| 4 Weeks | LightGBM      | 12.85 | 24.83 | 47.42% |

Single-year performance is not interpreted as proof that one model is universally superior.

---

# Explainability

DengueShield AI uses **SHAP — SHapley Additive exPlanations**.

The system provides:

- Global feature importance
- District-specific local explanations
- Positive and negative SHAP contributions
- Friendly feature names
- Historical SHAP figures
- Current operational SHAP summaries

Example top drivers often include:

- Current dengue cases
- Cases 1 week ago
- Recent 2-week case average
- Recent 4-week case average

> SHAP explains model behavior. It does not establish causality.

For example, a positive rainfall SHAP contribution does not prove that rainfall caused dengue activity.

---

# Relative Activity Categories

Forecast activity is interpreted relative to each district's own historical distribution.

| Category  | Historical Percentile Range |
| --------- | --------------------------- |
| LOW       | Below P50                   |
| ELEVATED  | P50 to below P75            |
| HIGH      | P75 to below P90            |
| VERY HIGH | P90 and above               |

These are **statistical categories**.

They are not official Ministry of Health outbreak or alert thresholds.

---

# Forecast Trend Categories

Operational trend labels are:

- **INCREASING** — forecast change ≥ +10%
- **DECREASING** — forecast change ≤ -10%
- **STABLE** — change between -10% and +10%
- **UNKNOWN** — percentage change cannot be meaningfully calculated

These labels describe model-estimated change.

---

# Experimental Current 2026 Inference

The current model uses:

- NDCU Week 37 surveillance
- Recent Open-Meteo weather observations
- Same historical 35-feature schema
- Previously selected Random Forest configuration
- Full-history refit
- No new 2026 hyperparameter tuning

Current national information:

```
Current reported cases:       1,156
Forecast sum:                 ≈ 1,030.13
Model-estimated difference:   ≈ -125.87
Approximate relative change:  ≈ -10.9%
```

Current activity distribution:

```
VERY HIGH     1
HIGH          5
ELEVATED     13
LOW           6
```

Current trend distribution:

```
INCREASING     7
STABLE         7
DECREASING    10
UNKNOWN        1
```

---

# Example: Colombo

Experimental Current 2026:

```
Current cases:          207
1-week RF forecast:     ≈ 176.7
Persistence forecast:   207
Forecast change:        ≈ -14.6%
Trend:                  DECREASING
Relative activity:      ELEVATED
```

Historical thresholds:

```
P50 = 169
P75 = 292
P90 = 440
```

Top SHAP contributions include approximately:

```
Current dengue cases        +77.8
Cases 1 week ago            +26.8
Recent 2-week case average  +21.1
```

---

# Example: Kandy

Experimental Current 2026:

```
Current cases:          174
1-week RF forecast:     ≈ 137.3
Persistence forecast:   174
Forecast change:        ≈ -21.1%
Trend:                  DECREASING
Relative activity:      HIGH
```

Historical thresholds:

```
P50 = 62
P75 = 107
P90 ≈ 220
```

Top SHAP contributions include approximately:

```
Current dengue cases        +69.5
Cases 1 week ago            +12.7
Recent 2-week case average   +9.5
```

---

# Source-Bridge Diagnostic

Historical model development and current inference use different surveillance pipelines.

A diagnostic comparison between overlapping WER-aligned and NDCU observations produced approximately:

```
Overlapping district-weeks: 925
MAE:                        10.699
Correlation:                0.955
```

This indicates strong overall agreement but does not establish complete equivalence.

Therefore current outputs remain explicitly experimental.

---

# Technology Stack

## Machine Learning

- Python 3.11
- pandas
- NumPy
- scikit-learn
- XGBoost
- LightGBM
- SHAP
- joblib

## Backend

- FastAPI
- Uvicorn

## Frontend

- React
- Vite
- Tailwind CSS
- Recharts
- Leaflet
- React Leaflet
- Lucide React

---

# Repository Structure

```
DengueShield-AI/
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│       └── operational/
│
├── docs/
│   ├── data_sources.md
│   ├── limitations.md
│   ├── methodology.md
│   ├── model_card.md
│   └── operational_inference.md
│
├── figures/
│
├── frontend/
│   └── src/
│
├── models/
│   ├── explainability/
│   └── operational/
│
├── src/
│   ├── api/
│   ├── features/
│   ├── inference/
│   └── models/
│
├── tests/
│   ├── test_api.py
│   ├── test_current_pipeline.py
│   └── test_features.py
│
├── README.md
└── requirements.txt
```

---

# Installation

## 1. Clone Repository

```
git clone https://github.com/thilak-79/DengueShield-AI.git
cd DengueShield-AI
```

---

## 2. Create Python Environment

Windows PowerShell:

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## 3. Install Frontend Dependencies

```
cd frontend
npm install
cd ..
```

---

# Running the Backend

From the project root:

```
.\.venv\Scripts\Activate.ps1
uvicorn src.api.main:app --reload
```

Backend:

```
http://127.0.0.1:8000
```

Swagger API documentation:

```
http://127.0.0.1:8000/docs
```

---

# Running the Frontend

Open another PowerShell terminal:

```
cd frontend
npm run dev
```

Frontend normally runs at:

```
http://localhost:5173
```

---

# Main API Endpoints

## Historical Evaluation

```
GET /api/health
GET /api/districts
GET /api/activity/latest
GET /api/activity/summary
GET /api/district/{district}
GET /api/explanation/global
GET /api/explanation/district/{district}
GET /api/model/performance
GET /api/model/walk-forward
GET /api/model/walk-forward/details
GET /api/model/info
```

## Experimental Current 2026

```
GET /api/current/status
GET /api/current/activity
GET /api/current/summary
GET /api/current/districts
GET /api/current/district/{district}
GET /api/current/explanation/global
GET /api/current/explanation/district/{district}
```

---

# Automated Tests

Run the complete test suite:

```
pytest -v
```

Reported test result (from project notes):

```
53 passed
0 failed
```

Tests validate:

- Historical API endpoints
- Current API endpoints
- Historical/current separation
- 25-district completeness
- Current national total
- Missing forecasts
- Non-negative forecasts
- Activity distributions
- Trend distributions
- Weather completeness
- 35-feature model schema
- No 2026 hyperparameter retuning
- Historical/current seasonality consistency
- Rolling-feature definitions
- Full SHAP additivity
- Top-10 local explanation structure
- Published forecast post-processing

---

# Frontend Production Build

```
cd frontend
npm run build
```

---

# Documentation

Detailed project documentation is available in:

```
docs/methodology.md
docs/data_sources.md
docs/model_card.md
docs/limitations.md
docs/operational_inference.md
```

---

# Important Limitations

The main limitations include:

- Historical WER and current NDCU surveillance have different reporting semantics
- Random Forest does not consistently beat persistence
- Longer forecast horizons remain difficult
- Current inference is experimental
- Weather uses representative district locations
- Surveillance data can contain delays or revisions
- SHAP explanations are not causal
- Statistical activity categories are not official alerts
- The system does not make individual-level predictions

See:

```
docs/limitations.md
```

for the detailed discussion.

---

# Scientific Interpretation

The main finding is not that machine learning always beats traditional baselines.

Instead:

> Random Forest showed a small average improvement over persistence at the 1-week horizon, but the advantage varied substantially across years. Persistence remained competitive and achieved lower mean MAE at the 2-week and 4-week horizons.

The system therefore emphasizes transparent comparison, uncertainty, source differences, and explainability rather than presenting the ML model as universally superior.

---

# Disclaimer

DengueShield AI is a research and experimental decision-support prototype.

It is:

- Not a medical diagnostic system
- Not an official public-health alert system
- Not a replacement for epidemiologists
- Not a replacement for Ministry of Health surveillance
- Not intended for autonomous public-health decisions

Official public-health decisions should rely on validated surveillance systems, field evidence, epidemiological expertise, and relevant health authorities.

---

# Project Status

```
Historical data pipeline       COMPLETE
Weather pipeline               COMPLETE
Feature engineering            COMPLETE
Model comparison               COMPLETE
Walk-forward validation        COMPLETE
SHAP explainability            COMPLETE
Experimental current pipeline  COMPLETE
FastAPI backend                COMPLETE
React dashboard                COMPLETE
Automated tests                53 / 53 PASSED
Documentation                  COMPLETE
```
