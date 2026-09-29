"""
DengueShield AI
SHAP Explainability for 1-Week Random Forest

Purpose
-------
Explain predictions from the saved 1-week Random Forest model.

The model was previously selected using temporal validation and then
refitted on train + validation data.

This script explains the 2025 one-week prediction rows.

Outputs
-------
models/explainability/
    shap_global_importance.csv
    shap_global_grouped_importance.csv
    shap_values_2025.csv
    shap_feature_values_2025.csv
    shap_metadata_2025.csv
    district_latest_explanations.csv
    district_explanations/
        <district>_latest.csv

figures/explainability/
    shap_global_bar.png
    shap_summary_beeswarm.png
    shap_colombo_latest_waterfall.png

Important
---------
SHAP values explain model predictions.

They do NOT establish causal relationships.
"""

from pathlib import Path
import re
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


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

MODEL_FILE = Path(
    "models/random_forest/"
    "random_forest_1w.joblib"
)

OUTPUT_DIR = Path(
    "models/explainability"
)

DISTRICT_OUTPUT_DIR = (
    OUTPUT_DIR
    /
    "district_explanations"
)

FIGURE_DIR = Path(
    "figures/explainability"
)


GLOBAL_IMPORTANCE_FILE = (
    OUTPUT_DIR
    /
    "shap_global_importance.csv"
)

GROUPED_IMPORTANCE_FILE = (
    OUTPUT_DIR
    /
    "shap_global_grouped_importance.csv"
)

SHAP_VALUES_FILE = (
    OUTPUT_DIR
    /
    "shap_values_2025.csv"
)

FEATURE_VALUES_FILE = (
    OUTPUT_DIR
    /
    "shap_feature_values_2025.csv"
)

METADATA_FILE = (
    OUTPUT_DIR
    /
    "shap_metadata_2025.csv"
)

DISTRICT_EXPLANATIONS_FILE = (
    OUTPUT_DIR
    /
    "district_latest_explanations.csv"
)


# ============================================================
# MODEL FEATURES
# Must match train_random_forest.py exactly.
# ============================================================

NUMERIC_FEATURES = [

    "cases_current",

    "temperature_current",
    "temperature_min_current",
    "temperature_max_current",
    "humidity_current",
    "rainfall_current",
    "rain_days_current",

    "cases_lag_1",
    "cases_lag_2",
    "cases_lag_3",
    "cases_lag_4",

    "cases_rolling_2",
    "cases_rolling_4",

    "rainfall_lag_1",
    "rainfall_lag_2",
    "rainfall_lag_3",
    "rainfall_lag_4",

    "rainfall_rolling_4",

    "humidity_lag_1",
    "humidity_lag_2",
    "humidity_lag_3",
    "humidity_lag_4",

    "humidity_rolling_4",

    "temperature_lag_1",
    "temperature_lag_2",
    "temperature_lag_3",
    "temperature_lag_4",

    "temperature_rolling_4",

    "rain_days_lag_1",
    "rain_days_lag_2",
    "rain_days_lag_3",
    "rain_days_lag_4",

    "week_sin",
    "week_cos",
]


CATEGORICAL_FEATURES = [
    "district",
]


MODEL_FEATURES = (
    NUMERIC_FEATURES
    +
    CATEGORICAL_FEATURES
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_feature_name(
    feature_name,
):
    """
    Convert pipeline feature names into
    easier human-readable names.

    Examples
    --------
    numeric__cases_current
        -> cases_current

    district__district_Colombo
        -> district=Colombo
    """

    if feature_name.startswith(
        "numeric__"
    ):

        return feature_name.replace(
            "numeric__",
            "",
            1,
        )

    if feature_name.startswith(
        "district__district_"
    ):

        district = feature_name.replace(
            "district__district_",
            "",
            1,
        )

        return (
            f"district={district}"
        )

    return feature_name


def feature_group(
    cleaned_name,
):
    """
    Group all one-hot district columns
    under a single 'district' feature group.
    """

    if cleaned_name.startswith(
        "district="
    ):

        return "district"

    return cleaned_name


def safe_filename(
    text,
):
    """
    Convert district name to safe filename.
    """

    result = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        str(text),
    )

    return result.strip(
        "_"
    )


def normalize_shap_values(
    shap_values,
):
    """
    Handle SHAP output differences between
    package versions.

    Expected final shape:
        n_samples x n_features
    """

    if isinstance(
        shap_values,
        list,
    ):

        if len(
            shap_values
        ) != 1:

            raise ValueError(
                "Unexpected multiple SHAP "
                "output arrays for regression."
            )

        shap_values = (
            shap_values[
                0
            ]
        )

    shap_values = np.asarray(
        shap_values
    )

    # Some SHAP versions may return:
    # samples x features x 1
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

    if shap_values.ndim != 2:

        raise ValueError(
            "Unexpected SHAP array shape: "
            f"{shap_values.shape}"
        )

    return shap_values


def normalize_expected_value(
    expected_value,
):
    """
    Convert TreeExplainer expected value
    into a scalar for regression.
    """

    values = np.asarray(
        expected_value
    ).reshape(
        -1
    )

    if len(
        values
    ) != 1:

        raise ValueError(
            "Expected one regression "
            "base value but received: "
            f"{values}"
        )

    return float(
        values[
            0
        ]
    )


# ============================================================
# LOAD INPUT DATA
# ============================================================

def load_data():

    if not FEATURE_FILE.exists():

        raise FileNotFoundError(
            f"Missing feature dataset: "
            f"{FEATURE_FILE}"
        )

    date_columns = [
        "start_date",
        "end_date",
        "forecast_origin_date",
        "target_date_1w",
    ]

    df = pd.read_csv(
        FEATURE_FILE,
        parse_dates=date_columns,
    )

    missing = (
        set(MODEL_FEATURES)
        -
        set(df.columns)
    )

    if missing:

        raise ValueError(
            f"Missing model features: "
            f"{sorted(missing)}"
        )

    # --------------------------------------------------------
    # Use the same one-week 2025 target-date definition
    # used in model evaluation.
    # --------------------------------------------------------

    test = (
        df[
            (
                df[
                    "target_cases_1w"
                ]
                .notna()
            )
            &
            (
                df[
                    "target_date_1w"
                ]
                .dt.year
                ==
                2025
            )
        ]
        .copy()
    )

    test = (
        test.sort_values(
            [
                "target_date_1w",
                "district",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if test.empty:

        raise AssertionError(
            "No 2025 one-week test rows."
        )

    print(
        f"[PASS] Loaded "
        f"{len(df):,} total rows"
    )

    print(
        f"[PASS] 2025 one-week rows: "
        f"{len(test):,}"
    )

    print(
        f"[PASS] Districts: "
        f"{test['district'].nunique()}"
    )

    return test


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            f"Missing RF model: "
            f"{MODEL_FILE}"
        )

    pipeline = joblib.load(
        MODEL_FILE
    )

    if not hasattr(
        pipeline,
        "named_steps",
    ):

        raise TypeError(
            "Saved object is not "
            "a sklearn Pipeline."
        )

    if (
        "preprocessor"
        not in
        pipeline.named_steps
    ):

        raise KeyError(
            "Pipeline has no "
            "'preprocessor' step."
        )

    if (
        "model"
        not in
        pipeline.named_steps
    ):

        raise KeyError(
            "Pipeline has no "
            "'model' step."
        )

    print(
        f"[PASS] Loaded model:"
    )

    print(
        f"       {MODEL_FILE}"
    )

    return pipeline


# ============================================================
# BUILD SHAP EXPLANATION
# ============================================================

def calculate_shap(
    pipeline,
    test,
):

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

    X_raw = (
        test[
            MODEL_FEATURES
        ]
    )

    # --------------------------------------------------------
    # Transform using the exact fitted preprocessor
    # stored inside the model pipeline.
    # --------------------------------------------------------

    X_transformed = (
        preprocessor.transform(
            X_raw
        )
    )

    X_transformed = np.asarray(
        X_transformed,
        dtype=float,
    )

    feature_names_raw = (
        preprocessor
        .get_feature_names_out()
    )

    feature_names = [
        clean_feature_name(
            name
        )
        for name
        in feature_names_raw
    ]

    if (
        X_transformed.shape[
            1
        ]
        !=
        len(
            feature_names
        )
    ):

        raise AssertionError(
            "Transformed feature count "
            "does not match feature names."
        )

    print(
        f"[PASS] Transformed features: "
        f"{X_transformed.shape[1]}"
    )

    # --------------------------------------------------------
    # Verify transformed-model predictions are exactly
    # consistent with predictions from the full pipeline.
    # --------------------------------------------------------

    pipeline_predictions = (
        pipeline.predict(
            X_raw
        )
    )

    model_predictions = (
        model.predict(
            X_transformed
        )
    )

    prediction_difference = (
        np.max(
            np.abs(
                pipeline_predictions
                -
                model_predictions
            )
        )
    )

    print(
        "[CHECK] Pipeline/model "
        "prediction difference: "
        f"{prediction_difference:.12f}"
    )

    if prediction_difference > 1e-8:

        raise AssertionError(
            "Pipeline and transformed model "
            "predictions do not match."
        )

    # --------------------------------------------------------
    # Tree SHAP
    # --------------------------------------------------------

    print(
        "[BUILD] Calculating SHAP values..."
    )

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = (
        explainer.shap_values(
            X_transformed
        )
    )

    shap_values = (
        normalize_shap_values(
            shap_values
        )
    )

    base_value = (
        normalize_expected_value(
            explainer.expected_value
        )
    )

    if (
        shap_values.shape
        !=
        X_transformed.shape
    ):

        raise AssertionError(
            "SHAP matrix shape "
            "does not match feature matrix."
        )

    # --------------------------------------------------------
    # Additivity check
    #
    # Prediction ≈ base + sum(SHAP)
    # --------------------------------------------------------

    reconstructed = (
        base_value
        +
        shap_values.sum(
            axis=1
        )
    )

    additivity_error = (
        np.max(
            np.abs(
                reconstructed
                -
                model_predictions
            )
        )
    )

    print(
        "[CHECK] Maximum SHAP "
        "additivity error: "
        f"{additivity_error:.10f}"
    )

    if additivity_error > 1e-4:

        raise AssertionError(
            "SHAP additivity validation failed."
        )

    print(
        f"[PASS] SHAP base value: "
        f"{base_value:.4f}"
    )

    return (
        X_transformed,
        shap_values,
        feature_names,
        pipeline_predictions,
        base_value,
    )


# ============================================================
# SAVE MATRICES
# ============================================================

def save_shap_tables(
    test,
    X_transformed,
    shap_values,
    feature_names,
    predictions,
    base_value,
):

    # --------------------------------------------------------
    # SHAP values
    # --------------------------------------------------------

    shap_df = pd.DataFrame(
        shap_values,
        columns=feature_names,
    )

    shap_df.insert(
        0,
        "row_id",
        np.arange(
            len(
                shap_df
            )
        ),
    )

    shap_df.to_csv(
        SHAP_VALUES_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Transformed feature values
    # --------------------------------------------------------

    feature_df = pd.DataFrame(
        X_transformed,
        columns=feature_names,
    )

    feature_df.insert(
        0,
        "row_id",
        np.arange(
            len(
                feature_df
            )
        ),
    )

    feature_df.to_csv(
        FEATURE_VALUES_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Metadata linking every SHAP row back
    # to the epidemiological observation.
    # --------------------------------------------------------

    metadata = pd.DataFrame(
        {
            "row_id":
            np.arange(
                len(
                    test
                )
            ),

            "district":
            test[
                "district"
            ].values,

            "forecast_origin_date":
            test[
                "forecast_origin_date"
            ].values,

            "target_date":
            test[
                "target_date_1w"
            ].values,

            "actual_cases":
            test[
                "target_cases_1w"
            ].values,

            "prediction_random_forest":
            predictions,

            "prediction_persistence":
            test[
                "cases_current"
            ].values,

            "base_value":
            base_value,
        }
    )

    metadata.to_csv(
        METADATA_FILE,
        index=False,
        date_format="%Y-%m-%d",
    )

    print(
        "[PASS] Saved SHAP matrices "
        "and metadata"
    )

    return (
        shap_df,
        feature_df,
        metadata,
    )


# ============================================================
# GLOBAL FEATURE IMPORTANCE
# ============================================================

def create_global_importance(
    shap_values,
    feature_names,
):

    mean_abs_shap = (
        np.mean(
            np.abs(
                shap_values
            ),
            axis=0,
        )
    )

    mean_signed_shap = (
        np.mean(
            shap_values,
            axis=0,
        )
    )

    importance = pd.DataFrame(
        {
            "feature":
            feature_names,

            "mean_abs_shap":
            mean_abs_shap,

            "mean_signed_shap":
            mean_signed_shap,
        }
    )

    importance[
        "feature_group"
    ] = (
        importance[
            "feature"
        ]
        .apply(
            feature_group
        )
    )

    importance = (
        importance.sort_values(
            "mean_abs_shap",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    importance[
        "rank"
    ] = (
        np.arange(
            1,
            len(
                importance
            )
            +
            1
        )
    )

    importance.to_csv(
        GLOBAL_IMPORTANCE_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Group one-hot district features together.
    # --------------------------------------------------------

    grouped = (
        importance.groupby(
            "feature_group",
            as_index=False,
        )
        .agg(
            mean_abs_shap=(
                "mean_abs_shap",
                "sum",
            ),

            mean_signed_shap=(
                "mean_signed_shap",
                "sum",
            ),
        )
    )

    grouped = (
        grouped.sort_values(
            "mean_abs_shap",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    grouped[
        "rank"
    ] = np.arange(
        1,
        len(
            grouped
        )
        +
        1
    )

    grouped.to_csv(
        GROUPED_IMPORTANCE_FILE,
        index=False,
    )

    print()
    print(
        "TOP GLOBAL SHAP FEATURES"
    )

    print(
        importance[
            [
                "rank",
                "feature",
                "mean_abs_shap",
            ]
        ]
        .head(
            15
        )
        .to_string(
            index=False
        )
    )

    print()
    print(
        "TOP GROUPED FEATURES"
    )

    print(
        grouped.head(
            15
        ).to_string(
            index=False
        )
    )

    return (
        importance,
        grouped,
    )


# ============================================================
# DISTRICT LATEST EXPLANATIONS
# ============================================================

def create_latest_district_explanations(
    test,
    X_transformed,
    shap_values,
    feature_names,
    predictions,
    base_value,
):

    all_rows = []

    # Find latest row separately for every district.
    latest_indices = (
        test.groupby(
            "district"
        )[
            "target_date_1w"
        ]
        .idxmax()
    )

    for row_index in latest_indices:

        source = test.loc[
            row_index
        ]

        district = source[
            "district"
        ]

        row_shap = (
            shap_values[
                row_index
            ]
        )

        row_features = (
            X_transformed[
                row_index
            ]
        )

        local = pd.DataFrame(
            {
                "feature":
                feature_names,

                "feature_value":
                row_features,

                "shap_value":
                row_shap,
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
            "forecast_origin_date"
        ] = source[
            "forecast_origin_date"
        ]

        local[
            "target_date"
        ] = source[
            "target_date_1w"
        ]

        local[
            "actual_cases"
        ] = source[
            "target_cases_1w"
        ]

        local[
            "prediction_random_forest"
        ] = predictions[
            row_index
        ]

        local[
            "prediction_persistence"
        ] = source[
            "cases_current"
        ]

        local[
            "base_value"
        ] = base_value

        # Keep the top ten contributions
        # for human-readable district files.
        top_local = (
            local.head(
                10
            )
            .copy()
        )

        columns = [
            "district",
            "forecast_origin_date",
            "target_date",
            "actual_cases",
            "prediction_random_forest",
            "prediction_persistence",
            "base_value",
            "contribution_rank",
            "feature",
            "feature_value",
            "shap_value",
            "abs_shap_value",
            "direction",
        ]

        top_local = (
            top_local[
                columns
            ]
        )

        filename = (
            DISTRICT_OUTPUT_DIR
            /
            (
                safe_filename(
                    district
                )
                +
                "_latest.csv"
            )
        )

        top_local.to_csv(
            filename,
            index=False,
            date_format="%Y-%m-%d",
        )

        all_rows.append(
            top_local
        )

    combined = pd.concat(
        all_rows,
        ignore_index=True,
    )

    combined.to_csv(
        DISTRICT_EXPLANATIONS_FILE,
        index=False,
        date_format="%Y-%m-%d",
    )

    print()
    print(
        "[PASS] Created latest "
        "explanations for "
        f"{test['district'].nunique()} "
        "districts"
    )

    return combined


# ============================================================
# PLOTS
# ============================================================

def create_plots(
    test,
    X_transformed,
    shap_values,
    feature_names,
    base_value,
):

    # --------------------------------------------------------
    # Global SHAP bar plot
    # --------------------------------------------------------

    plt.figure()

    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=feature_names,
        plot_type="bar",
        max_display=15,
        show=False,
    )

    plt.tight_layout()

    bar_file = (
        FIGURE_DIR
        /
        "shap_global_bar.png"
    )

    plt.savefig(
        bar_file,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # --------------------------------------------------------
    # Beeswarm summary
    # --------------------------------------------------------

    plt.figure()

    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=feature_names,
        max_display=15,
        show=False,
    )

    plt.tight_layout()

    beeswarm_file = (
        FIGURE_DIR
        /
        "shap_summary_beeswarm.png"
    )

    plt.savefig(
        beeswarm_file,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # --------------------------------------------------------
    # Latest Colombo explanation
    # --------------------------------------------------------

    colombo = (
        test[
            test[
                "district"
            ]
            ==
            "Colombo"
        ]
    )

    if not colombo.empty:

        row_index = (
            colombo[
                "target_date_1w"
            ]
            .idxmax()
        )

        explanation = shap.Explanation(
            values=(
                shap_values[
                    row_index
                ]
            ),

            base_values=(
                base_value
            ),

            data=(
                X_transformed[
                    row_index
                ]
            ),

            feature_names=(
                feature_names
            ),
        )

        plt.figure()

        shap.plots.waterfall(
            explanation,
            max_display=12,
            show=False,
        )

        waterfall_file = (
            FIGURE_DIR
            /
            "shap_colombo_latest_waterfall.png"
        )

        plt.savefig(
            waterfall_file,
            dpi=200,
            bbox_inches="tight",
        )

        plt.close()

        print(
            "[PASS] Created Colombo "
            "waterfall plot"
        )

    print(
        "[PASS] Created SHAP plots"
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
        "SHAP EXPLAINABILITY"
    )

    print(
        "=========================================="
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DISTRICT_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"[PASS] SHAP version: "
        f"{shap.__version__}"
    )

    test = load_data()

    pipeline = load_model()

    (
        X_transformed,
        shap_values,
        feature_names,
        predictions,
        base_value,
    ) = calculate_shap(
        pipeline,
        test,
    )

    save_shap_tables(
        test,
        X_transformed,
        shap_values,
        feature_names,
        predictions,
        base_value,
    )

    create_global_importance(
        shap_values,
        feature_names,
    )

    district_explanations = (
        create_latest_district_explanations(
            test,
            X_transformed,
            shap_values,
            feature_names,
            predictions,
            base_value,
        )
    )

    create_plots(
        test,
        X_transformed,
        shap_values,
        feature_names,
        base_value,
    )

    # --------------------------------------------------------
    # Show latest Colombo explanation
    # --------------------------------------------------------

    colombo = (
        district_explanations[
            district_explanations[
                "district"
            ]
            ==
            "Colombo"
        ]
    )

    if not colombo.empty:

        print()
        print(
            "=========================================="
        )

        print(
            "LATEST COLOMBO EXPLANATION"
        )

        print(
            "=========================================="
        )

        display_columns = [
            "contribution_rank",
            "feature",
            "feature_value",
            "shap_value",
            "direction",
        ]

        print(
            colombo[
                display_columns
            ]
            .to_string(
                index=False
            )
        )

    print()
    print(
        "=========================================="
    )

    print(
        "SHAP EXPLAINABILITY COMPLETE"
    )

    print(
        "=========================================="
    )

    print()
    print(
        "[PASS] Saved:"
    )

    print(
        f"       "
        f"{GLOBAL_IMPORTANCE_FILE}"
    )

    print(
        f"       "
        f"{GROUPED_IMPORTANCE_FILE}"
    )

    print(
        f"       "
        f"{SHAP_VALUES_FILE}"
    )

    print(
        f"       "
        f"{FEATURE_VALUES_FILE}"
    )

    print(
        f"       "
        f"{METADATA_FILE}"
    )

    print(
        f"       "
        f"{DISTRICT_EXPLANATIONS_FILE}"
    )

    print(
        f"       "
        f"{DISTRICT_OUTPUT_DIR}"
    )

    print(
        f"       "
        f"{FIGURE_DIR}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "SHAP explains model predictions."
    )

    print(
        "It does not establish causality."
    )


if __name__ == "__main__":
    main()