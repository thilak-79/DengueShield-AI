"""
DengueShield AI
District Risk / Activity Intelligence Builder

Purpose
-------
Convert the latest one-week Random Forest forecasts and SHAP
explanations into a dashboard-ready district-level dataset.

This script creates STATISTICAL RELATIVE ACTIVITY LEVELS.

These are NOT official Ministry of Health alert thresholds.

Activity categories
-------------------
LOW:
    forecast < district historical 50th percentile

ELEVATED:
    50th <= forecast < 75th percentile

HIGH:
    75th <= forecast < 90th percentile

VERY HIGH:
    forecast >= 90th percentile

Historical percentiles are calculated separately for each district
using only observations that occurred BEFORE the forecast origin date.

Inputs
------
data/processed/ml_features_2015_2025.csv

models/explainability/shap_metadata_2025.csv

models/explainability/
    district_latest_explanations.csv

Outputs
-------
data/processed/latest_district_risk.csv

data/processed/
    latest_district_risk_summary.csv

data/processed/
    latest_district_risk_metadata.csv
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


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
# PATHS
# ============================================================

FEATURE_FILE = Path(
    "data/processed/"
    "ml_features_2015_2025.csv"
)

SHAP_METADATA_FILE = Path(
    "models/explainability/"
    "shap_metadata_2025.csv"
)

DISTRICT_EXPLANATIONS_FILE = Path(
    "models/explainability/"
    "district_latest_explanations.csv"
)

OUTPUT_FILE = Path(
    "data/processed/"
    "latest_district_risk.csv"
)

SUMMARY_FILE = Path(
    "data/processed/"
    "latest_district_risk_summary.csv"
)

METADATA_FILE = Path(
    "data/processed/"
    "latest_district_risk_metadata.csv"
)


# ============================================================
# SETTINGS
# ============================================================

EXPECTED_DISTRICTS = 25

TOP_DRIVER_COUNT = 3


# ============================================================
# HELPERS
# ============================================================

def activity_level(
    forecast,
    p50,
    p75,
    p90,
):
    """
    Statistical activity category.

    Not an official health alert classification.
    """

    if forecast < p50:

        return "LOW"

    if forecast < p75:

        return "ELEVATED"

    if forecast < p90:

        return "HIGH"

    return "VERY HIGH"


def safe_percent_change(
    current,
    forecast,
):
    """
    Percentage forecast change relative
    to current cases.

    If current is zero, return NaN rather
    than dividing by zero.
    """

    if current == 0:

        return np.nan

    return (
        (
            forecast
            -
            current
        )
        /
        current
        *
        100.0
    )


def trend_label(
    percent_change,
):
    """
    Simple descriptive forecast direction.

    This is NOT an epidemiological alert.
    """

    if pd.isna(
        percent_change
    ):

        return "UNKNOWN"

    if percent_change >= 10:

        return "INCREASING"

    if percent_change <= -10:

        return "DECREASING"

    return "STABLE"


def friendly_feature_name(
    feature,
):
    """
    Convert technical feature names into
    simple dashboard labels.
    """

    mappings = {

        "cases_current":
        "Current dengue cases",

        "cases_lag_1":
        "Cases 1 week ago",

        "cases_lag_2":
        "Cases 2 weeks ago",

        "cases_lag_3":
        "Cases 3 weeks ago",

        "cases_lag_4":
        "Cases 4 weeks ago",

        "cases_rolling_2":
        "Recent 2-week case average",

        "cases_rolling_4":
        "Recent 4-week case average",

        "rainfall_current":
        "Current rainfall",

        "rainfall_lag_1":
        "Rainfall 1 week ago",

        "rainfall_lag_2":
        "Rainfall 2 weeks ago",

        "rainfall_lag_3":
        "Rainfall 3 weeks ago",

        "rainfall_lag_4":
        "Rainfall 4 weeks ago",

        "rainfall_rolling_4":
        "Recent 4-week rainfall",

        "humidity_current":
        "Current humidity",

        "humidity_lag_1":
        "Humidity 1 week ago",

        "humidity_lag_2":
        "Humidity 2 weeks ago",

        "humidity_lag_3":
        "Humidity 3 weeks ago",

        "humidity_lag_4":
        "Humidity 4 weeks ago",

        "humidity_rolling_4":
        "Recent 4-week humidity",

        "temperature_current":
        "Current temperature",

        "temperature_lag_1":
        "Temperature 1 week ago",

        "temperature_lag_2":
        "Temperature 2 weeks ago",

        "temperature_lag_3":
        "Temperature 3 weeks ago",

        "temperature_lag_4":
        "Temperature 4 weeks ago",

        "temperature_rolling_4":
        "Recent 4-week temperature",

        "week_sin":
        "Seasonal timing",

        "week_cos":
        "Seasonal timing",
    }

    if feature.startswith(
        "district="
    ):

        return (
            "District identity"
        )

    return mappings.get(
        feature,
        feature.replace(
            "_",
            " ",
        ).title(),
    )


# ============================================================
# LOAD
# ============================================================

def load_inputs():

    required_files = [
        FEATURE_FILE,
        SHAP_METADATA_FILE,
        DISTRICT_EXPLANATIONS_FILE,
    ]

    for file in required_files:

        if not file.exists():

            raise FileNotFoundError(
                f"Missing required file: "
                f"{file}"
            )

    features = pd.read_csv(
        FEATURE_FILE,
        parse_dates=[
            "forecast_origin_date",
            "target_date_1w",
        ],
    )

    shap_metadata = pd.read_csv(
        SHAP_METADATA_FILE,
        parse_dates=[
            "forecast_origin_date",
            "target_date",
        ],
    )

    explanations = pd.read_csv(
        DISTRICT_EXPLANATIONS_FILE,
        parse_dates=[
            "forecast_origin_date",
            "target_date",
        ],
    )

    print(
        f"[PASS] Feature rows: "
        f"{len(features):,}"
    )

    print(
        f"[PASS] SHAP metadata rows: "
        f"{len(shap_metadata):,}"
    )

    print(
        f"[PASS] Explanation rows: "
        f"{len(explanations):,}"
    )

    return (
        features,
        shap_metadata,
        explanations,
    )


# ============================================================
# LATEST FORECAST PER DISTRICT
# ============================================================

def get_latest_forecasts(
    shap_metadata,
):

    latest_indices = (
        shap_metadata.groupby(
            "district"
        )[
            "target_date"
        ]
        .idxmax()
    )

    latest = (
        shap_metadata.loc[
            latest_indices
        ]
        .copy()
        .sort_values(
            "district"
        )
        .reset_index(
            drop=True
        )
    )

    if (
        latest[
            "district"
        ]
        .nunique()
        !=
        EXPECTED_DISTRICTS
    ):

        raise AssertionError(
            "Expected "
            f"{EXPECTED_DISTRICTS} "
            "districts in latest forecasts."
        )

    if (
        latest[
            "district"
        ]
        .duplicated()
        .any()
    ):

        raise AssertionError(
            "Duplicate latest district "
            "forecasts detected."
        )

    print(
        f"[PASS] Latest forecasts: "
        f"{len(latest)} districts"
    )

    return latest


# ============================================================
# HISTORICAL PERCENTILES
# ============================================================

def calculate_historical_reference(
    features,
    latest_forecasts,
):
    """
    Calculate district-specific historical
    distributions using ONLY observations
    before each forecast origin date.
    """

    rows = []

    for _, latest in (
        latest_forecasts.iterrows()
    ):

        district = (
            latest[
                "district"
            ]
        )

        forecast_origin = (
            latest[
                "forecast_origin_date"
            ]
        )

        history = (
            features[
                (
                    features[
                        "district"
                    ]
                    ==
                    district
                )
                &
                (
                    features[
                        "forecast_origin_date"
                    ]
                    <
                    forecast_origin
                )
            ][
                "cases_current"
            ]
            .dropna()
            .astype(float)
        )

        if len(
            history
        ) < 50:

            raise AssertionError(
                f"Insufficient historical "
                f"reference data for "
                f"{district}: "
                f"{len(history)} rows"
            )

        rows.append(
            {
                "district":
                district,

                "historical_observations":
                len(
                    history
                ),

                "historical_mean":
                float(
                    history.mean()
                ),

                "historical_median":
                float(
                    history.median()
                ),

                "historical_p50":
                float(
                    history.quantile(
                        0.50
                    )
                ),

                "historical_p75":
                float(
                    history.quantile(
                        0.75
                    )
                ),

                "historical_p90":
                float(
                    history.quantile(
                        0.90
                    )
                ),

                "historical_max":
                float(
                    history.max()
                ),
            }
        )

    reference = pd.DataFrame(
        rows
    )

    print(
        "[PASS] Historical district "
        "percentiles calculated"
    )

    return reference


# ============================================================
# TOP SHAP DRIVERS
# ============================================================

def build_driver_table(
    explanations,
):
    """
    Convert top SHAP contributions into
    one row per district.
    """

    rows = []

    for district, group in (
        explanations.groupby(
            "district"
        )
    ):

        group = (
            group.sort_values(
                "contribution_rank"
            )
            .head(
                TOP_DRIVER_COUNT
            )
        )

        record = {
            "district":
            district,
        }

        for position, (
            _,
            row,
        ) in enumerate(
            group.iterrows(),
            start=1,
        ):

            record[
                f"top_driver_{position}"
            ] = (
                friendly_feature_name(
                    str(
                        row[
                            "feature"
                        ]
                    )
                )
            )

            record[
                f"top_driver_{position}_technical"
            ] = (
                row[
                    "feature"
                ]
            )

            record[
                f"top_driver_{position}_value"
            ] = (
                row[
                    "feature_value"
                ]
            )

            record[
                f"top_driver_{position}_shap"
            ] = (
                row[
                    "shap_value"
                ]
            )

            record[
                f"top_driver_{position}_direction"
            ] = (
                row[
                    "direction"
                ]
            )

        rows.append(
            record
        )

    result = pd.DataFrame(
        rows
    )

    if (
        result[
            "district"
        ]
        .nunique()
        !=
        EXPECTED_DISTRICTS
    ):

        raise AssertionError(
            "Driver table does not contain "
            "all districts."
        )

    print(
        "[PASS] Top SHAP drivers "
        "prepared"
    )

    return result


# ============================================================
# BUILD FINAL INTELLIGENCE TABLE
# ============================================================

def build_risk_table(
    latest,
    reference,
    drivers,
):

    result = (
        latest.merge(
            reference,
            on="district",
            how="left",
            validate="one_to_one",
        )
        .merge(
            drivers,
            on="district",
            how="left",
            validate="one_to_one",
        )
    )

    result = result.rename(
        columns={
            "prediction_random_forest":
            "forecast_cases_1w",

            "prediction_persistence":
            "persistence_forecast_1w",

            "actual_cases":
            "observed_target_cases",
        }
    )

    # cases_current used by persistence
    result[
        "current_cases"
    ] = (
        result[
            "persistence_forecast_1w"
        ]
    )

    result[
        "forecast_absolute_change"
    ] = (
        result[
            "forecast_cases_1w"
        ]
        -
        result[
            "current_cases"
        ]
    )

    result[
        "forecast_percent_change"
    ] = result.apply(
        lambda row:
        safe_percent_change(
            row[
                "current_cases"
            ],
            row[
                "forecast_cases_1w"
            ],
        ),
        axis=1,
    )

    result[
        "forecast_trend"
    ] = (
        result[
            "forecast_percent_change"
        ]
        .apply(
            trend_label
        )
    )

    result[
        "activity_level"
    ] = result.apply(
        lambda row:
        activity_level(
            row[
                "forecast_cases_1w"
            ],
            row[
                "historical_p50"
            ],
            row[
                "historical_p75"
            ],
            row[
                "historical_p90"
            ],
        ),
        axis=1,
    )

    # Difference from simple persistence benchmark.
    result[
        "model_minus_persistence"
    ] = (
        result[
            "forecast_cases_1w"
        ]
        -
        result[
            "persistence_forecast_1w"
        ]
    )

    result[
        "forecast_horizon_weeks"
    ] = 1

    result[
        "forecast_model"
    ] = "random_forest"

    result[
        "activity_definition"
    ] = (
        "District-specific historical "
        "percentile category; not an "
        "official epidemiological alert."
    )

    result[
        "model_interpretation_note"
    ] = (
        "SHAP values explain model "
        "predictions and do not establish "
        "causal relationships."
    )

    # --------------------------------------------------------
    # Arrange dashboard-friendly columns
    # --------------------------------------------------------

    columns = [

        "district",

        "forecast_origin_date",
        "target_date",

        "forecast_horizon_weeks",
        "forecast_model",

        "current_cases",
        "forecast_cases_1w",
        "persistence_forecast_1w",

        "forecast_absolute_change",
        "forecast_percent_change",
        "forecast_trend",

        "historical_observations",
        "historical_mean",
        "historical_median",
        "historical_p50",
        "historical_p75",
        "historical_p90",
        "historical_max",

        "activity_level",

        "model_minus_persistence",

        "top_driver_1",
        "top_driver_1_technical",
        "top_driver_1_value",
        "top_driver_1_shap",
        "top_driver_1_direction",

        "top_driver_2",
        "top_driver_2_technical",
        "top_driver_2_value",
        "top_driver_2_shap",
        "top_driver_2_direction",

        "top_driver_3",
        "top_driver_3_technical",
        "top_driver_3_value",
        "top_driver_3_shap",
        "top_driver_3_direction",

        "observed_target_cases",

        "base_value",

        "activity_definition",
        "model_interpretation_note",
    ]

    result = result[
        columns
    ]

    result = (
        result.sort_values(
            [
                "activity_level",
                "forecast_cases_1w",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    return result


# ============================================================
# SUMMARY TABLE
# ============================================================

def build_summary(
    risk_table,
):

    level_order = [
        "LOW",
        "ELEVATED",
        "HIGH",
        "VERY HIGH",
    ]

    summary = (
        risk_table.groupby(
            "activity_level"
        )
        .agg(
            districts=(
                "district",
                "count",
            ),

            mean_current_cases=(
                "current_cases",
                "mean",
            ),

            mean_forecast_cases=(
                "forecast_cases_1w",
                "mean",
            ),

            mean_forecast_change_percent=(
                "forecast_percent_change",
                "mean",
            ),
        )
        .reindex(
            level_order
        )
        .fillna(
            0
        )
        .reset_index()
    )

    return summary


# ============================================================
# METADATA
# ============================================================

def build_metadata(
    risk_table,
):

    return pd.DataFrame(
        [
            {
                "metric":
                "district_count",

                "value":
                risk_table[
                    "district"
                ]
                .nunique(),
            },

            {
                "metric":
                "forecast_horizon_weeks",

                "value":
                1,
            },

            {
                "metric":
                "model",

                "value":
                "random_forest",
            },

            {
                "metric":
                "activity_threshold_type",

                "value":
                (
                    "district-specific "
                    "historical percentiles"
                ),
            },

            {
                "metric":
                "activity_levels",

                "value":
                (
                    "LOW < P50; "
                    "ELEVATED P50-P75; "
                    "HIGH P75-P90; "
                    "VERY HIGH >= P90"
                ),
            },

            {
                "metric":
                "official_alert_thresholds",

                "value":
                "No",
            },

            {
                "metric":
                "shap_causality",

                "value":
                (
                    "No - SHAP explains "
                    "model behavior only"
                ),
            },
        ]
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_output(
    result,
):

    if len(
        result
    ) != EXPECTED_DISTRICTS:

        raise AssertionError(
            "Final table should contain "
            f"{EXPECTED_DISTRICTS} rows."
        )

    if (
        result[
            "district"
        ]
        .nunique()
        !=
        EXPECTED_DISTRICTS
    ):

        raise AssertionError(
            "Expected 25 unique districts."
        )

    if (
        result[
            "forecast_cases_1w"
        ]
        .isna()
        .any()
    ):

        raise AssertionError(
            "Missing forecast values."
        )

    if (
        result[
            "activity_level"
        ]
        .isna()
        .any()
    ):

        raise AssertionError(
            "Missing activity levels."
        )

    allowed = {
        "LOW",
        "ELEVATED",
        "HIGH",
        "VERY HIGH",
    }

    unexpected = (
        set(
            result[
                "activity_level"
            ]
            .unique()
        )
        -
        allowed
    )

    if unexpected:

        raise AssertionError(
            "Unexpected activity levels: "
            f"{unexpected}"
        )

    print(
        "[PASS] Final risk table "
        "validation passed"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "=========================================="
    )

    print(
        "DENGUESHIELD AI"
    )

    print(
        "RISK / ACTIVITY INTELLIGENCE"
    )

    print(
        "=========================================="
    )

    (
        features,
        shap_metadata,
        explanations,
    ) = load_inputs()

    latest = (
        get_latest_forecasts(
            shap_metadata
        )
    )

    reference = (
        calculate_historical_reference(
            features,
            latest,
        )
    )

    drivers = (
        build_driver_table(
            explanations
        )
    )

    risk_table = (
        build_risk_table(
            latest,
            reference,
            drivers,
        )
    )

    validate_output(
        risk_table
    )

    summary = (
        build_summary(
            risk_table
        )
    )

    metadata = (
        build_metadata(
            risk_table
        )
    )

    risk_table.to_csv(
        OUTPUT_FILE,
        index=False,
        date_format="%Y-%m-%d",
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    metadata.to_csv(
        METADATA_FILE,
        index=False,
    )

    print()
    print(
        "=========================================="
    )

    print(
        "LATEST DISTRICT ACTIVITY"
    )

    print(
        "=========================================="
    )

    display_columns = [

        "district",

        "current_cases",

        "forecast_cases_1w",

        "forecast_percent_change",

        "forecast_trend",

        "activity_level",

        "top_driver_1",

        "top_driver_2",

        "top_driver_3",
    ]

    print(
        risk_table[
            display_columns
        ]
        .sort_values(
            "forecast_cases_1w",
            ascending=False,
        )
        .to_string(
            index=False
        )
    )

    print()
    print(
        "=========================================="
    )

    print(
        "ACTIVITY LEVEL SUMMARY"
    )

    print(
        "=========================================="
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "[PASS] Saved:"
    )

    print(
        f"       {OUTPUT_FILE}"
    )

    print(
        f"       {SUMMARY_FILE}"
    )

    print(
        f"       {METADATA_FILE}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Activity levels are statistical "
        "historical percentile categories."
    )

    print(
        "They are NOT official public-health "
        "alert thresholds."
    )


if __name__ == "__main__":
    main()