"""
DengueShield AI
Experimental Current 2026 Inference

Generates:

- 1-week RF forecast
- persistence benchmark
- current SHAP explanations
- experimental historical-percentile activity categories

IMPORTANT
---------
Current surveillance source = NDCU
Historical training source = WER-aligned dataset

Therefore outputs are EXPERIMENTAL CURRENT INFERENCE,
not validated official dengue alerts.
"""

from pathlib import Path
from datetime import datetime, timezone
import json
import sys

import joblib
import numpy as np
import pandas as pd
import shap


try:
    sys.stdout.reconfigure(
        encoding="utf-8",
        errors="replace",
    )
except Exception:
    pass


MODEL_FILE = Path(
    "models/operational/"
    "random_forest_1w_full_history.joblib"
)

FEATURE_FILE = Path(
    "data/processed/operational/"
    "current_features_2026.csv"
)

REFERENCE_FEATURE_FILE = Path(
    "data/processed/"
    "ml_features_2015_2025.csv"
)

METADATA_FILE = Path(
    "data/processed/operational/"
    "current_data_metadata.json"
)

FORECAST_FILE = Path(
    "data/processed/operational/"
    "current_forecast_2026.csv"
)

ACTIVITY_FILE = Path(
    "data/processed/operational/"
    "current_activity_2026.csv"
)

EXPLANATION_FILE = Path(
    "models/explainability/"
    "current_district_explanations_2026.csv"
)

GLOBAL_SHAP_FILE = Path(
    "models/explainability/"
    "current_shap_global_2026.csv"
)


def clean_feature_name(
    name,
):

    if name.startswith(
        "numeric__"
    ):

        return name.replace(
            "numeric__",
            "",
            1,
        )

    if name.startswith(
        "district__district_"
    ):

        return (
            "district="
            +
            name.replace(
                "district__district_",
                "",
                1,
            )
        )

    return name


def friendly_name(
    feature,
):

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

        "week_sin":
        "Seasonal timing",

        "week_cos":
        "Seasonal timing",
    }

    if feature.startswith(
        "district="
    ):

        return "District identity"

    return mappings.get(
        feature,
        feature.replace(
            "_",
            " ",
        ).title(),
    )


def activity_level(
    forecast,
    p50,
    p75,
    p90,
):

    if forecast < p50:
        return "LOW"

    if forecast < p75:
        return "ELEVATED"

    if forecast < p90:
        return "HIGH"

    return "VERY HIGH"


def trend(
    percentage,
):

    if pd.isna(
        percentage
    ):
        return "UNKNOWN"

    if percentage >= 10:
        return "INCREASING"

    if percentage <= -10:
        return "DECREASING"

    return "STABLE"


def main():

    print()
    print(
        "=========================================="
    )
    print(
        "DENGUESHIELD AI"
    )
    print(
        "EXPERIMENTAL CURRENT INFERENCE"
    )
    print(
        "=========================================="
    )

    pipeline = joblib.load(
        MODEL_FILE
    )

    current = pd.read_csv(
        FEATURE_FILE
    )

    historical = pd.read_csv(
        REFERENCE_FEATURE_FILE
    )

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        metadata = json.load(
            file
        )

    model_features = list(
        pipeline.feature_names_in_
    )

    X = current[
        model_features
    ]

    prediction = (
        pipeline.predict(
            X
        )
    )

    prediction = np.clip(
        prediction,
        0,
        None,
    )

    persistence = (
        current[
            "cases_current"
        ]
        .to_numpy(
            dtype=float
        )
    )

    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    preprocessor = (
        pipeline.named_steps[
            "preprocessor"
        ]
    )

    model = (
        pipeline.named_steps[
            "model"
        ]
    )

    transformed = np.asarray(
        preprocessor.transform(
            X
        ),
        dtype=float,
    )

    transformed_names_raw = (
        preprocessor
        .get_feature_names_out()
    )

    transformed_names = [
        clean_feature_name(
            name
        )
        for name
        in transformed_names_raw
    ]

    model_prediction = (
        model.predict(
            transformed
        )
    )

    difference = float(
        np.max(
            np.abs(
                model_prediction
                -
                prediction
            )
        )
    )

    if difference > 1e-8:

        raise AssertionError(
            "Pipeline/model predictions differ."
        )

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = (
        explainer.shap_values(
            transformed
        )
    )

    if isinstance(
        shap_values,
        list,
    ):

        shap_values = (
            shap_values[
                0
            ]
        )

    shap_values = np.asarray(
        shap_values
    )

    if (
        shap_values.ndim
        ==
        3
        and
        shap_values.shape[
            -1
        ]
        ==
        1
    ):

        shap_values = np.squeeze(
            shap_values,
            axis=-1,
        )

    base_value = float(
        np.asarray(
            explainer.expected_value
        )
        .reshape(
            -1
        )[
            0
        ]
    )

    reconstructed = (
        base_value
        +
        shap_values.sum(
            axis=1
        )
    )

    additivity_error = float(
        np.max(
            np.abs(
                reconstructed
                -
                model_prediction
            )
        )
    )

    if additivity_error > 1e-4:

        raise AssertionError(
            "Current SHAP additivity failed."
        )

    print(
        f"[PASS] SHAP additivity error: "
        f"{additivity_error:.10f}"
    )

    # --------------------------------------------------------
    # HISTORICAL DISTRICT REFERENCE
    # --------------------------------------------------------

    reference = (
        historical.groupby(
            "district"
        )[
            "cases_current"
        ]
        .agg(
            historical_mean="mean",
            historical_median="median",
            historical_p50=lambda x:
            x.quantile(
                0.50
            ),
            historical_p75=lambda x:
            x.quantile(
                0.75
            ),
            historical_p90=lambda x:
            x.quantile(
                0.90
            ),
            historical_max="max",
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # EXPLANATIONS
    # --------------------------------------------------------

    explanation_rows = []

    for row_index in range(
        len(
            current
        )
    ):

        district = (
            current.iloc[
                row_index
            ][
                "district"
            ]
        )

        local = pd.DataFrame(
            {
                "feature":
                transformed_names,

                "feature_value":
                transformed[
                    row_index
                ],

                "shap_value":
                shap_values[
                    row_index
                ],
            }
        )

        local[
            "abs_shap_value"
        ] = (
            local[
                "shap_value"
            ]
            .abs()
        )

        local[
            "direction"
        ] = np.where(
            local[
                "shap_value"
            ]
            >
            0,
            "increases_prediction",
            np.where(
                local[
                    "shap_value"
                ]
                <
                0,
                "decreases_prediction",
                "neutral",
            ),
        )

        local = (
            local.sort_values(
                "abs_shap_value",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

        local[
            "contribution_rank"
        ] = np.arange(
            1,
            len(
                local
            )
            +
            1
        )

        local[
            "district"
        ] = district

        local[
            "feature_friendly"
        ] = (
            local[
                "feature"
            ]
            .apply(
                friendly_name
            )
        )

        local[
            "forecast_cases_1w"
        ] = (
            prediction[
                row_index
            ]
        )

        local[
            "base_value"
        ] = (
            base_value
        )

        local[
            "forecast_origin_date"
        ] = metadata[
            "latest_week_end"
        ]

        local[
            "target_start_date"
        ] = metadata[
            "forecast_target_start"
        ]

        explanation_rows.append(
            local.head(
                10
            )
        )

    explanations = pd.concat(
        explanation_rows,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # GLOBAL CURRENT SHAP
    # --------------------------------------------------------

    global_shap = pd.DataFrame(
        {
            "feature":
            transformed_names,

            "mean_abs_shap":
            np.mean(
                np.abs(
                    shap_values
                ),
                axis=0,
            ),

            "mean_signed_shap":
            np.mean(
                shap_values,
                axis=0,
            ),
        }
    ).sort_values(
        "mean_abs_shap",
        ascending=False,
    )

    # --------------------------------------------------------
    # BASIC FORECAST TABLE
    # --------------------------------------------------------

    forecast = current[
        [
            "district",
            "year",
            "week",
            "forecast_origin_date",
            "target_date_1w",
            "cases_current",
        ]
    ].copy()

    forecast = forecast.rename(
        columns={
            "cases_current":
            "current_cases"
        }
    )

    forecast[
        "forecast_cases_1w"
    ] = prediction

    forecast[
        "persistence_forecast_1w"
    ] = persistence

    forecast[
        "model_minus_persistence"
    ] = (
        forecast[
            "forecast_cases_1w"
        ]
        -
        forecast[
            "persistence_forecast_1w"
        ]
    )

    forecast[
        "forecast_absolute_change"
    ] = (
        forecast[
            "forecast_cases_1w"
        ]
        -
        forecast[
            "current_cases"
        ]
    )

    forecast[
        "forecast_percent_change"
    ] = np.where(
        forecast[
            "current_cases"
        ]
        !=
        0,
        (
            forecast[
                "forecast_absolute_change"
            ]
            /
            forecast[
                "current_cases"
            ]
            *
            100.0
        ),
        np.nan,
    )

    forecast[
        "forecast_trend"
    ] = (
        forecast[
            "forecast_percent_change"
        ]
        .apply(
            trend
        )
    )

    forecast[
        "forecast_model"
    ] = (
        "random_forest_full_history"
    )

    forecast[
        "operational_status"
    ] = (
        "EXPERIMENTAL_CURRENT_INFERENCE"
    )

    forecast[
        "case_source"
    ] = "NDCU"

    forecast[
        "training_source"
    ] = "WER"

    forecast[
        "source_semantics_warning"
    ] = True

    forecast[
        "data_freshness_date"
    ] = metadata[
        "data_freshness_date"
    ]

    forecast[
        "generated_at_utc"
    ] = datetime.now(
        timezone.utc
    ).isoformat()

    # --------------------------------------------------------
    # ACTIVITY INTELLIGENCE
    # --------------------------------------------------------

    activity = forecast.merge(
        reference,
        on="district",
        how="left",
        validate="one_to_one",
    )

    activity[
        "relative_activity"
    ] = activity.apply(
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

    activity[
        "activity_definition"
    ] = (
        "Experimental district-specific "
        "historical percentile category; "
        "not an official public-health alert."
    )

    # --------------------------------------------------------
    # TOP 3 SHAP DRIVERS
    # --------------------------------------------------------

    for rank in [
        1,
        2,
        3,
    ]:

        top = (
            explanations[
                explanations[
                    "contribution_rank"
                ]
                ==
                rank
            ][
                [
                    "district",
                    "feature",
                    "feature_friendly",
                    "feature_value",
                    "shap_value",
                    "direction",
                ]
            ]
            .rename(
                columns={
                    "feature":
                    f"top_driver_{rank}_technical",

                    "feature_friendly":
                    f"top_driver_{rank}",

                    "feature_value":
                    f"top_driver_{rank}_value",

                    "shap_value":
                    f"top_driver_{rank}_shap",

                    "direction":
                    f"top_driver_{rank}_direction",
                }
            )
        )

        activity = activity.merge(
            top,
            on="district",
            how="left",
            validate="one_to_one",
        )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if len(
        activity
    ) != 25:

        raise AssertionError(
            "Expected 25 current activity rows."
        )

    if (
        activity[
            "forecast_cases_1w"
        ]
        .isna()
        .any()
    ):

        raise AssertionError(
            "Missing current forecasts."
        )

    if (
        activity[
            "relative_activity"
        ]
        .isna()
        .any()
    ):

        raise AssertionError(
            "Missing relative activity."
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    FORECAST_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    EXPLANATION_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    forecast.to_csv(
        FORECAST_FILE,
        index=False,
    )

    activity.to_csv(
        ACTIVITY_FILE,
        index=False,
    )

    explanations.to_csv(
        EXPLANATION_FILE,
        index=False,
    )

    global_shap.to_csv(
        GLOBAL_SHAP_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print()
    print(
        "=========================================="
    )

    print(
        "CURRENT 2026 FORECAST"
    )

    print(
        "=========================================="
    )

    print(
        activity[
            [
                "district",
                "current_cases",
                "forecast_cases_1w",
                "forecast_percent_change",
                "forecast_trend",
                "relative_activity",
                "top_driver_1",
                "top_driver_2",
                "top_driver_3",
            ]
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
        "[PASS] Districts: 25"
    )

    print(
        f"[PASS] Forecast file: "
        f"{FORECAST_FILE}"
    )

    print(
        f"[PASS] Activity file: "
        f"{ACTIVITY_FILE}"
    )

    print(
        f"[PASS] Explanations: "
        f"{EXPLANATION_FILE}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "These are experimental current "
        "inferences using NDCU surveillance "
        "with a WER-trained model."
    )

    print(
        "They are not official Ministry of "
        "Health forecasts or alert levels."
    )


if __name__ == "__main__":
    main()