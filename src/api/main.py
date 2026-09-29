"""
DengueShield AI
FastAPI Backend

Provides dashboard-ready endpoints for:

- Latest district activity intelligence
- District-level forecast details
- SHAP explanations
- Model performance
- Walk-forward validation
- Activity summaries
- Generated visualization files

Important
---------
Current API data represents the prototype / historical evaluation
pipeline.

The statistical activity levels are NOT official public-health
alert thresholds.

SHAP values explain model behavior and do NOT establish causality.
"""

from pathlib import Path
import json
import math
import sys
from typing import Any

import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


# ============================================================
# UTF-8
# ============================================================

try:
    sys.stdout.reconfigure(
        encoding="utf-8",
        errors="replace",
    )
except Exception:
    pass


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


RISK_FILE = (
    PROJECT_ROOT
    /
    "data/processed/latest_district_risk.csv"
)

RISK_SUMMARY_FILE = (
    PROJECT_ROOT
    /
    "data/processed/latest_district_risk_summary.csv"
)

RISK_METADATA_FILE = (
    PROJECT_ROOT
    /
    "data/processed/latest_district_risk_metadata.csv"
)

EXPLANATIONS_FILE = (
    PROJECT_ROOT
    /
    "models/explainability/"
    "district_latest_explanations.csv"
)

GLOBAL_SHAP_FILE = (
    PROJECT_ROOT
    /
    "models/explainability/"
    "shap_global_grouped_importance.csv"
)

MODEL_COMPARISON_FILE = (
    PROJECT_ROOT
    /
    "models/comparison/"
    "final_model_comparison.csv"
)

DISTRICT_CONSISTENCY_FILE = (
    PROJECT_ROOT
    /
    "models/comparison/"
    "district_consistency_summary.csv"
)

WALK_FORWARD_SUMMARY_FILE = (
    PROJECT_ROOT
    /
    "models/walk_forward/"
    "walk_forward_summary.csv"
)

WALK_FORWARD_METRICS_FILE = (
    PROJECT_ROOT
    /
    "models/walk_forward/"
    "walk_forward_metrics.csv"
)

DASHBOARD_FIGURES_DIR = (
    PROJECT_ROOT
    /
    "figures/dashboard"
)

EXPLAINABILITY_FIGURES_DIR = (
    PROJECT_ROOT
    /
    "figures/explainability"
)


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="DengueShield AI API",
    description=(
        "Explainable dengue activity forecasting "
        "and decision-support prototype for "
        "Sri Lankan districts."
    ),
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],

    allow_credentials=True,

    allow_methods=[
        "*"
    ],

    allow_headers=[
        "*"
    ],
)


# ============================================================
# STATIC FIGURES
# ============================================================

if DASHBOARD_FIGURES_DIR.exists():

    app.mount(
        "/figures/dashboard",
        StaticFiles(
            directory=str(
                DASHBOARD_FIGURES_DIR
            )
        ),
        name="dashboard-figures",
    )


if EXPLAINABILITY_FIGURES_DIR.exists():

    app.mount(
        "/figures/explainability",
        StaticFiles(
            directory=str(
                EXPLAINABILITY_FIGURES_DIR
            )
        ),
        name="explainability-figures",
    )


# ============================================================
# HELPERS
# ============================================================

def require_file(
    path: Path,
):
    """
    Raise a useful error if a required
    generated file does not exist.
    """

    if not path.exists():

        raise HTTPException(
            status_code=503,
            detail=(
                "Required generated data "
                f"is missing: {path.name}"
            ),
        )


def load_csv(
    path: Path,
):
    require_file(
        path
    )

    try:

        return pd.read_csv(
            path
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Could not load "
                f"{path.name}: {exc}"
            ),
        )


def dataframe_records(
    df: pd.DataFrame,
):
    """
    Convert DataFrame to JSON-safe records.

    pandas to_json converts NaN values
    safely to null.
    """

    return json.loads(
        df.to_json(
            orient="records",
            date_format="iso",
        )
    )


def clean_value(
    value: Any,
):
    """
    Convert NumPy/pandas values into
    JSON-safe Python values.
    """

    if value is None:

        return None

    if isinstance(
        value,
        pd.Timestamp,
    ):

        return value.isoformat()

    try:

        if pd.isna(
            value
        ):

            return None

    except Exception:

        pass

    if hasattr(
        value,
        "item",
    ):

        try:

            return value.item()

        except Exception:

            pass

    if isinstance(
        value,
        float,
    ):

        if not math.isfinite(
            value
        ):

            return None

    return value


def row_to_dict(
    row,
):
    return {
        key: clean_value(
            value
        )
        for key, value
        in row.items()
    }


def find_district(
    df: pd.DataFrame,
    district: str,
):
    """
    Case-insensitive district lookup.
    """

    matches = (
        df[
            df[
                "district"
            ]
            .astype(str)
            .str.lower()
            ==
            district.lower()
        ]
    )

    if matches.empty:

        raise HTTPException(
            status_code=404,
            detail=(
                f"District not found: "
                f"{district}"
            ),
        )

    return matches.iloc[
        0
    ]


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "project":
        "DengueShield AI",

        "status":
        "running",

        "api_version":
        "1.0.0",

        "documentation":
        "/docs",

        "important_note":
        (
            "Relative activity categories "
            "are statistical categories and "
            "are not official public-health "
            "alert thresholds."
        ),
    }


# ============================================================
# HEALTH
# ============================================================

@app.get(
    "/api/health"
)
def health():

    files = {

        "risk_data":
        RISK_FILE,

        "risk_summary":
        RISK_SUMMARY_FILE,

        "explanations":
        EXPLANATIONS_FILE,

        "model_comparison":
        MODEL_COMPARISON_FILE,

        "walk_forward":
        WALK_FORWARD_SUMMARY_FILE,
    }

    file_status = {
        name: path.exists()
        for name, path
        in files.items()
    }

    all_ready = all(
        file_status.values()
    )

    return {
        "status":
        (
            "healthy"
            if all_ready
            else
            "degraded"
        ),

        "files":
        file_status,

        "project_root":
        str(
            PROJECT_ROOT
        ),
    }


# ============================================================
# DISTRICT LIST
# ============================================================

@app.get(
    "/api/districts"
)
def get_districts():

    df = load_csv(
        RISK_FILE
    )

    districts = sorted(
        df[
            "district"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    return {
        "count":
        len(
            districts
        ),

        "districts":
        districts,
    }


# ============================================================
# ALL LATEST DISTRICT ACTIVITY
# ============================================================

@app.get(
    "/api/activity/latest"
)
def latest_activity():

    df = load_csv(
        RISK_FILE
    )

    sort_order = {
        "VERY HIGH": 0,
        "HIGH": 1,
        "ELEVATED": 2,
        "LOW": 3,
    }

    if (
        "activity_level"
        in
        df.columns
    ):

        df[
            "_activity_order"
        ] = (
            df[
                "activity_level"
            ]
            .map(
                sort_order
            )
            .fillna(
                99
            )
        )

        df = (
            df.sort_values(
                [
                    "_activity_order",
                    "forecast_cases_1w",
                ],
                ascending=[
                    True,
                    False,
                ],
            )
            .drop(
                columns=[
                    "_activity_order"
                ]
            )
        )

    return {
        "count":
        len(
            df
        ),

        "forecast_horizon_weeks":
        1,

        "model":
        "random_forest",

        "data":
        dataframe_records(
            df
        ),

        "disclaimer":
        (
            "Activity categories are based "
            "on district-specific historical "
            "percentiles and are not official "
            "public-health alert thresholds."
        ),
    }


# ============================================================
# ACTIVITY SUMMARY
# ============================================================

@app.get(
    "/api/activity/summary"
)
def activity_summary():

    df = load_csv(
        RISK_SUMMARY_FILE
    )

    return {
        "data":
        dataframe_records(
            df
        )
    }


# ============================================================
# ONE DISTRICT
# ============================================================

@app.get(
    "/api/district/{district}"
)
def district_detail(
    district: str,
):

    risk_df = load_csv(
        RISK_FILE
    )

    risk_row = find_district(
        risk_df,
        district,
    )

    explanations_df = load_csv(
        EXPLANATIONS_FILE
    )

    explanation_matches = (
        explanations_df[
            explanations_df[
                "district"
            ]
            .astype(str)
            .str.lower()
            ==
            district.lower()
        ]
        .copy()
    )

    explanation_matches = (
        explanation_matches.sort_values(
            "contribution_rank"
        )
    )

    top_drivers = []

    for _, row in (
        explanation_matches
        .head(
            10
        )
        .iterrows()
    ):

        top_drivers.append(
            {
                "rank":
                clean_value(
                    row[
                        "contribution_rank"
                    ]
                ),

                "feature":
                clean_value(
                    row[
                        "feature"
                    ]
                ),

                "feature_value":
                clean_value(
                    row[
                        "feature_value"
                    ]
                ),

                "shap_value":
                clean_value(
                    row[
                        "shap_value"
                    ]
                ),

                "direction":
                clean_value(
                    row[
                        "direction"
                    ]
                ),
            }
        )

    return {
        "district":
        row_to_dict(
            risk_row
        ),

        "explanation":
        top_drivers,

        "interpretation_note":
        (
            "SHAP explains why the model "
            "produced its prediction. "
            "It does not establish causal "
            "relationships."
        ),
    }


# ============================================================
# SHAP EXPLANATION ONLY
# ============================================================

@app.get(
    "/api/explanation/{district}"
)
def district_explanation(
    district: str,
):

    df = load_csv(
        EXPLANATIONS_FILE
    )

    matches = (
        df[
            df[
                "district"
            ]
            .astype(str)
            .str.lower()
            ==
            district.lower()
        ]
        .copy()
    )

    if matches.empty:

        raise HTTPException(
            status_code=404,
            detail=(
                f"No explanation found "
                f"for district: {district}"
            ),
        )

    matches = (
        matches.sort_values(
            "contribution_rank"
        )
    )

    return {
        "district":
        matches[
            "district"
        ]
        .iloc[
            0
        ],

        "drivers":
        dataframe_records(
            matches
        ),

        "note":
        (
            "Positive SHAP values push the "
            "forecast upward relative to the "
            "model baseline; negative values "
            "push it downward."
        ),

        "causality":
        False,
    }


# ============================================================
# GLOBAL SHAP IMPORTANCE
# ============================================================

@app.get(
    "/api/explanation/global"
)
def global_explanation():

    df = load_csv(
        GLOBAL_SHAP_FILE
    )

    return {
        "feature_count":
        len(
            df
        ),

        "features":
        dataframe_records(
            df
        ),

        "note":
        (
            "mean_abs_shap represents average "
            "feature impact magnitude. "
            "This is model explainability, "
            "not causal inference."
        ),
    }


# ============================================================
# MODEL PERFORMANCE
# ============================================================

@app.get(
    "/api/model/performance"
)
def model_performance():

    comparison = load_csv(
        MODEL_COMPARISON_FILE
    )

    district_consistency = load_csv(
        DISTRICT_CONSISTENCY_FILE
    )

    return {
        "primary_metric":
        "MAE",

        "single_year_comparison":
        dataframe_records(
            comparison
        ),

        "district_consistency":
        dataframe_records(
            district_consistency
        ),

        "interpretation":
        (
            "Machine-learning improvements "
            "over persistence are modest and "
            "vary by horizon and district."
        ),
    }


# ============================================================
# WALK-FORWARD SUMMARY
# ============================================================

@app.get(
    "/api/model/walk-forward"
)
def walk_forward_summary():

    summary = load_csv(
        WALK_FORWARD_SUMMARY_FILE
    )

    return {
        "evaluation_years":
        [
            2020,
            2021,
            2022,
            2023,
            2024,
            2025,
        ],

        "summary":
        dataframe_records(
            summary
        ),

        "methodology":
        (
            "For each test year, model "
            "configuration was selected using "
            "the immediately preceding "
            "validation year and trained only "
            "with historical data."
        ),
    }


# ============================================================
# WALK-FORWARD DETAILS
# ============================================================

@app.get(
    "/api/model/walk-forward/details"
)
def walk_forward_details():

    df = load_csv(
        WALK_FORWARD_METRICS_FILE
    )

    return {
        "count":
        len(
            df
        ),

        "data":
        dataframe_records(
            df
        ),
    }


# ============================================================
# MODEL / DATA INFORMATION
# ============================================================

@app.get(
    "/api/model/info"
)
def model_info():

    metadata = []

    if RISK_METADATA_FILE.exists():

        metadata_df = pd.read_csv(
            RISK_METADATA_FILE
        )

        metadata = dataframe_records(
            metadata_df
        )

    return {
        "primary_ml_model":
        "Random Forest",

        "primary_forecast_horizon":
        "1 week",

        "features":
        (
            "Historical dengue activity, "
            "meteorological variables, "
            "district identity and seasonal "
            "features."
        ),

        "activity_category_method":
        (
            "District-specific historical "
            "percentiles."
        ),

        "official_alert_threshold":
        False,

        "individual_diagnosis":
        False,

        "prototype_status":
        (
            "Historical evaluation / "
            "decision-support prototype"
        ),

        "metadata":
        metadata,
    }