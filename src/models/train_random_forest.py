"""
DengueShield AI
Random Forest Dengue Forecasting

Forecast horizons:
    1 week
    2 weeks
    4 weeks

Leakage-safe split:
    Train targets      <= 2023
    Validation targets == 2024
    Test targets       == 2025

Workflow for each horizon:
    1. Fit candidate RF configurations on TRAIN only.
    2. Select best configuration using VALIDATION MAE.
    3. Never use TEST for hyperparameter selection.
    4. Refit best configuration on TRAIN + VALIDATION.
    5. Evaluate once on 2025 TEST.
    6. Compare test result against persistence baseline.

Categorical feature:
    district -> one-hot encoded

Metrics:
    MAE
    RMSE
    sMAPE
"""

from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# UTF-8 OUTPUT
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

INPUT_FILE = Path(
    "data/processed/"
    "ml_features_2015_2025.csv"
)

BASELINE_METRICS_FILE = Path(
    "models/baseline/"
    "baseline_metrics.csv"
)

OUTPUT_DIR = Path(
    "models/random_forest"
)

METRICS_FILE = (
    OUTPUT_DIR
    / "random_forest_metrics.csv"
)

PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "random_forest_predictions.csv"
)

SEARCH_FILE = (
    OUTPUT_DIR
    / "random_forest_validation_search.csv"
)

FEATURE_IMPORTANCE_FILE = (
    OUTPUT_DIR
    / "random_forest_feature_importance.csv"
)

COMPARISON_FILE = (
    OUTPUT_DIR
    / "random_forest_vs_baseline.csv"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

HORIZONS = [
    1,
    2,
    4,
]


# ============================================================
# FEATURES
# ============================================================

NUMERIC_FEATURES = [

    # --------------------------------------------
    # Current dengue state
    # --------------------------------------------
    "cases_current",

    # --------------------------------------------
    # Current weather
    # --------------------------------------------
    "temperature_current",
    "temperature_min_current",
    "temperature_max_current",
    "humidity_current",
    "rainfall_current",
    "rain_days_current",

    # --------------------------------------------
    # Dengue history
    # --------------------------------------------
    "cases_lag_1",
    "cases_lag_2",
    "cases_lag_3",
    "cases_lag_4",

    "cases_rolling_2",
    "cases_rolling_4",

    # --------------------------------------------
    # Rainfall history
    # --------------------------------------------
    "rainfall_lag_1",
    "rainfall_lag_2",
    "rainfall_lag_3",
    "rainfall_lag_4",

    "rainfall_rolling_4",

    # --------------------------------------------
    # Humidity history
    # --------------------------------------------
    "humidity_lag_1",
    "humidity_lag_2",
    "humidity_lag_3",
    "humidity_lag_4",

    "humidity_rolling_4",

    # --------------------------------------------
    # Temperature history
    # --------------------------------------------
    "temperature_lag_1",
    "temperature_lag_2",
    "temperature_lag_3",
    "temperature_lag_4",

    "temperature_rolling_4",

    # --------------------------------------------
    # Rain-day history
    # --------------------------------------------
    "rain_days_lag_1",
    "rain_days_lag_2",
    "rain_days_lag_3",
    "rain_days_lag_4",

    # --------------------------------------------
    # Seasonality
    # --------------------------------------------
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
# SMALL VALIDATION-ONLY PARAMETER SEARCH
# ============================================================

PARAMETER_CANDIDATES = [

    {
        "name": "rf_depth12",

        "n_estimators": 300,

        "max_depth": 12,

        "min_samples_split": 2,

        "min_samples_leaf": 2,

        "max_features": 0.7,
    },

    {
        "name": "rf_depth18",

        "n_estimators": 400,

        "max_depth": 18,

        "min_samples_split": 2,

        "min_samples_leaf": 2,

        "max_features": 0.8,
    },

    {
        "name": "rf_regularized",

        "n_estimators": 400,

        "max_depth": None,

        "min_samples_split": 4,

        "min_samples_leaf": 3,

        "max_features": 1.0,
    },
]


# ============================================================
# LOAD
# ============================================================

def load_data():
    """
    Load feature dataset.
    """

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Missing feature dataset: "
            f"{INPUT_FILE}"
        )

    date_columns = [
        "start_date",
        "end_date",
        "forecast_origin_date",
        "target_date_1w",
        "target_date_2w",
        "target_date_4w",
    ]

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=date_columns,
    )

    missing_features = (
        set(MODEL_FEATURES)
        -
        set(df.columns)
    )

    if missing_features:

        raise ValueError(
            "Missing model features: "
            f"{sorted(missing_features)}"
        )

    print(
        f"[PASS] Loaded "
        f"{len(df):,} rows"
    )

    print(
        f"[PASS] Districts: "
        f"{df['district'].nunique()}"
    )

    print(
        f"[PASS] Numeric features: "
        f"{len(NUMERIC_FEATURES)}"
    )

    return df


# ============================================================
# METRICS
# ============================================================

def smape(
    y_true,
    y_pred,
):
    """
    Symmetric Mean Absolute Percentage Error.

    Zero-zero pairs contribute zero.
    """

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    denominator = (
        np.abs(y_true)
        +
        np.abs(y_pred)
    )

    numerator = (
        200.0
        *
        np.abs(
            y_pred
            -
            y_true
        )
    )

    result = np.zeros_like(
        denominator,
        dtype=float,
    )

    np.divide(
        numerator,
        denominator,
        out=result,
        where=denominator != 0,
    )

    return float(
        np.mean(result)
    )


def calculate_metrics(
    y_true,
    y_pred,
):

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = (
        mean_squared_error(
            y_true,
            y_pred,
        )
        ** 0.5
    )

    smape_value = smape(
        y_true,
        y_pred,
    )

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "smape": float(smape_value),
    }


# ============================================================
# DATA SPLITS
# ============================================================

def prepare_horizon_data(
    df,
    horizon,
):
    """
    Split using TARGET DATE, not row year.

    This prevents late-December forecasts from
    crossing accidentally into another split.
    """

    target_column = (
        f"target_cases_{horizon}w"
    )

    target_date_column = (
        f"target_date_{horizon}w"
    )

    working = (
        df[
            df[
                target_column
            ]
            .notna()
        ]
        .copy()
    )

    target_year = (
        working[
            target_date_column
        ]
        .dt.year
    )

    train = (
        working[
            target_year <= 2023
        ]
        .copy()
    )

    validation = (
        working[
            target_year == 2024
        ]
        .copy()
    )

    test = (
        working[
            target_year == 2025
        ]
        .copy()
    )

    if train.empty:

        raise AssertionError(
            "Training set is empty."
        )

    if validation.empty:

        raise AssertionError(
            "Validation set is empty."
        )

    if test.empty:

        raise AssertionError(
            "Test set is empty."
        )

    print(
        f"Train:      "
        f"{len(train):,}"
    )

    print(
        f"Validation: "
        f"{len(validation):,}"
    )

    print(
        f"Test:       "
        f"{len(test):,}"
    )

    return (
        train,
        validation,
        test,
        target_column,
        target_date_column,
    )


# ============================================================
# PIPELINE
# ============================================================

def build_pipeline(
    parameters,
):
    """
    Create preprocessing + Random Forest pipeline.

    Numerical values pass through unchanged.
    District is one-hot encoded.
    """

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "district",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),

            (
                "numeric",
                "passthrough",
                NUMERIC_FEATURES,
            ),
        ],
        remainder="drop",
    )

    model = RandomForestRegressor(
        n_estimators=(
            parameters[
                "n_estimators"
            ]
        ),

        max_depth=(
            parameters[
                "max_depth"
            ]
        ),

        min_samples_split=(
            parameters[
                "min_samples_split"
            ]
        ),

        min_samples_leaf=(
            parameters[
                "min_samples_leaf"
            ]
        ),

        max_features=(
            parameters[
                "max_features"
            ]
        ),

        criterion=(
            "squared_error"
        ),

        random_state=(
            RANDOM_STATE
        ),

        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),

            (
                "model",
                model,
            ),
        ]
    )

    return pipeline


# ============================================================
# PARAMETER SEARCH
# ============================================================

def select_best_parameters(
    train,
    validation,
    target_column,
    horizon,
):
    """
    Hyperparameter selection uses 2024 validation only.

    2025 test data is never used here.
    """

    print()
    print(
        "VALIDATION MODEL SELECTION"
    )

    X_train = (
        train[
            MODEL_FEATURES
        ]
    )

    y_train = (
        train[
            target_column
        ]
    )

    X_validation = (
        validation[
            MODEL_FEATURES
        ]
    )

    y_validation = (
        validation[
            target_column
        ]
    )

    search_rows = []

    best_pipeline = None
    best_parameters = None
    best_metrics = None

    best_mae = float(
        "inf"
    )

    for config in (
        PARAMETER_CANDIDATES
    ):

        print()
        print(
            f"  Training: "
            f"{config['name']}"
        )

        pipeline = (
            build_pipeline(
                config
            )
        )

        pipeline.fit(
            X_train,
            y_train,
        )

        predictions = (
            pipeline.predict(
                X_validation
            )
        )

        predictions = np.clip(
            predictions,
            0,
            None,
        )

        metrics = (
            calculate_metrics(
                y_validation,
                predictions,
            )
        )

        print(
            f"    MAE:   "
            f"{metrics['mae']:.4f}"
        )

        print(
            f"    RMSE:  "
            f"{metrics['rmse']:.4f}"
        )

        print(
            f"    sMAPE: "
            f"{metrics['smape']:.2f}%"
        )

        search_rows.append(
            {
                "horizon_weeks":
                horizon,

                "configuration":
                config["name"],

                "n_estimators":
                config[
                    "n_estimators"
                ],

                "max_depth":
                config[
                    "max_depth"
                ],

                "min_samples_split":
                config[
                    "min_samples_split"
                ],

                "min_samples_leaf":
                config[
                    "min_samples_leaf"
                ],

                "max_features":
                config[
                    "max_features"
                ],

                "validation_mae":
                metrics["mae"],

                "validation_rmse":
                metrics["rmse"],

                "validation_smape":
                metrics["smape"],
            }
        )

        if (
            metrics["mae"]
            <
            best_mae
        ):

            best_mae = (
                metrics["mae"]
            )

            best_pipeline = (
                pipeline
            )

            best_parameters = (
                config.copy()
            )

            best_metrics = (
                metrics.copy()
            )

    print()
    print(
        "[SELECTED]"
    )

    print(
        f"  Configuration: "
        f"{best_parameters['name']}"
    )

    print(
        f"  Validation MAE: "
        f"{best_metrics['mae']:.4f}"
    )

    return (
        best_pipeline,
        best_parameters,
        best_metrics,
        search_rows,
    )


# ============================================================
# PREDICTION TABLE
# ============================================================

def make_prediction_table(
    source_df,
    target_column,
    target_date_column,
    predictions,
    horizon,
    split_name,
    training_stage,
):
    """
    Create reusable prediction output.
    """

    output = pd.DataFrame(
        {
            "district":
            source_df[
                "district"
            ].values,

            "forecast_origin_date":
            source_df[
                "forecast_origin_date"
            ].values,

            "target_date":
            source_df[
                target_date_column
            ].values,

            "horizon_weeks":
            horizon,

            "split":
            split_name,

            "training_stage":
            training_stage,

            "actual":
            source_df[
                target_column
            ].values,

            "prediction_random_forest":
            predictions,

            "prediction_persistence":
            source_df[
                "cases_current"
            ].values,
        }
    )

    return output


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def extract_feature_importance(
    pipeline,
    horizon,
):
    """
    Extract impurity-based Random Forest
    feature importance from final model.

    SHAP will be added later.
    """

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

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    importances = (
        model.feature_importances_
    )

    if (
        len(feature_names)
        !=
        len(importances)
    ):

        raise AssertionError(
            "Feature-name and "
            "importance lengths differ."
        )

    result = pd.DataFrame(
        {
            "horizon_weeks":
            horizon,

            "feature":
            feature_names,

            "importance":
            importances,
        }
    )

    result = (
        result.sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    return result


# ============================================================
# TRAIN ONE HORIZON
# ============================================================

def train_horizon(
    df,
    horizon,
):
    """
    Full leakage-safe workflow for one horizon.
    """

    print()
    print(
        "=" * 70
    )

    print(
        f"{horizon}-WEEK "
        f"RANDOM FOREST"
    )

    print(
        "=" * 70
    )

    (
        train,
        validation,
        test,
        target_column,
        target_date_column,
    ) = prepare_horizon_data(
        df,
        horizon,
    )

    # --------------------------------------------------------
    # SELECT USING VALIDATION
    # --------------------------------------------------------

    (
        validation_model,
        best_parameters,
        validation_metrics,
        search_rows,
    ) = select_best_parameters(
        train,
        validation,
        target_column,
        horizon,
    )

    # --------------------------------------------------------
    # Store unbiased validation predictions.
    #
    # This model saw TRAIN only.
    # --------------------------------------------------------

    validation_predictions = (
        validation_model.predict(
            validation[
                MODEL_FEATURES
            ]
        )
    )

    validation_predictions = (
        np.clip(
            validation_predictions,
            0,
            None,
        )
    )

    validation_prediction_table = (
        make_prediction_table(
            source_df=validation,
            target_column=target_column,
            target_date_column=(
                target_date_column
            ),
            predictions=(
                validation_predictions
            ),
            horizon=horizon,
            split_name="validation",
            training_stage=(
                "train_only"
            ),
        )
    )

    # --------------------------------------------------------
    # FINAL REFIT
    #
    # After hyperparameters are selected,
    # use all pre-test data:
    #
    #     train + validation
    #
    # The 2025 test set remains untouched.
    # --------------------------------------------------------

    train_validation = (
        pd.concat(
            [
                train,
                validation,
            ],
            ignore_index=True,
        )
    )

    print()
    print(
        "[REFIT] Best model on "
        "train + validation"
    )

    print(
        f"        Rows: "
        f"{len(train_validation):,}"
    )

    final_pipeline = (
        build_pipeline(
            best_parameters
        )
    )

    final_pipeline.fit(
        train_validation[
            MODEL_FEATURES
        ],
        train_validation[
            target_column
        ],
    )

    # --------------------------------------------------------
    # FINAL TEST
    # --------------------------------------------------------

    test_predictions = (
        final_pipeline.predict(
            test[
                MODEL_FEATURES
            ]
        )
    )

    test_predictions = (
        np.clip(
            test_predictions,
            0,
            None,
        )
    )

    test_metrics = (
        calculate_metrics(
            test[
                target_column
            ],
            test_predictions,
        )
    )

    print()
    print(
        "FINAL 2025 TEST"
    )

    print(
        f"  MAE:   "
        f"{test_metrics['mae']:.4f}"
    )

    print(
        f"  RMSE:  "
        f"{test_metrics['rmse']:.4f}"
    )

    print(
        f"  sMAPE: "
        f"{test_metrics['smape']:.2f}%"
    )

    # --------------------------------------------------------
    # Persistence on exactly same test rows
    # --------------------------------------------------------

    persistence_metrics = (
        calculate_metrics(
            test[
                target_column
            ],
            test[
                "cases_current"
            ],
        )
    )

    print()
    print(
        "PERSISTENCE BENCHMARK"
    )

    print(
        f"  MAE:   "
        f"{persistence_metrics['mae']:.4f}"
    )

    print(
        f"  RMSE:  "
        f"{persistence_metrics['rmse']:.4f}"
    )

    print(
        f"  sMAPE: "
        f"{persistence_metrics['smape']:.2f}%"
    )

    # --------------------------------------------------------
    # Percentage improvement
    # --------------------------------------------------------

    mae_improvement = (
        (
            persistence_metrics[
                "mae"
            ]
            -
            test_metrics[
                "mae"
            ]
        )
        /
        persistence_metrics[
            "mae"
        ]
        *
        100.0
    )

    rmse_improvement = (
        (
            persistence_metrics[
                "rmse"
            ]
            -
            test_metrics[
                "rmse"
            ]
        )
        /
        persistence_metrics[
            "rmse"
        ]
        *
        100.0
    )

    print()
    print(
        "IMPROVEMENT VS PERSISTENCE"
    )

    print(
        f"  MAE improvement: "
        f"{mae_improvement:.2f}%"
    )

    print(
        f"  RMSE improvement: "
        f"{rmse_improvement:.2f}%"
    )

    # --------------------------------------------------------
    # Prediction output
    # --------------------------------------------------------

    test_prediction_table = (
        make_prediction_table(
            source_df=test,
            target_column=target_column,
            target_date_column=(
                target_date_column
            ),
            predictions=test_predictions,
            horizon=horizon,
            split_name="test",
            training_stage=(
                "train_plus_validation"
            ),
        )
    )

    # --------------------------------------------------------
    # Metrics rows
    # --------------------------------------------------------

    metrics_rows = [

        {
            "horizon_weeks":
            horizon,

            "split":
            "validation",

            "model":
            "random_forest",

            "mae":
            validation_metrics[
                "mae"
            ],

            "rmse":
            validation_metrics[
                "rmse"
            ],

            "smape":
            validation_metrics[
                "smape"
            ],

            "selected_configuration":
            best_parameters[
                "name"
            ],
        },

        {
            "horizon_weeks":
            horizon,

            "split":
            "test",

            "model":
            "random_forest",

            "mae":
            test_metrics[
                "mae"
            ],

            "rmse":
            test_metrics[
                "rmse"
            ],

            "smape":
            test_metrics[
                "smape"
            ],

            "selected_configuration":
            best_parameters[
                "name"
            ],
        },
    ]

    # --------------------------------------------------------
    # Save final model locally
    # --------------------------------------------------------

    model_file = (
        OUTPUT_DIR
        /
        f"random_forest_{horizon}w.joblib"
    )

    joblib.dump(
        final_pipeline,
        model_file,
    )

    print()
    print(
        f"[PASS] Saved model:"
    )

    print(
        f"       {model_file}"
    )

    # --------------------------------------------------------
    # Save params as JSON
    # --------------------------------------------------------

    params_file = (
        OUTPUT_DIR
        /
        f"random_forest_{horizon}w_params.json"
    )

    with open(
        params_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            best_parameters,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    feature_importance = (
        extract_feature_importance(
            final_pipeline,
            horizon,
        )
    )

    predictions = pd.concat(
        [
            validation_prediction_table,
            test_prediction_table,
        ],
        ignore_index=True,
    )

    return {
        "metrics":
        metrics_rows,

        "search":
        search_rows,

        "predictions":
        predictions,

        "feature_importance":
        feature_importance,

        "persistence_metrics":
        persistence_metrics,

        "random_forest_test_metrics":
        test_metrics,

        "best_parameters":
        best_parameters,
    }


# ============================================================
# BASELINE COMPARISON
# ============================================================

def create_baseline_comparison(
    rf_metrics,
):
    """
    Compare Random Forest with saved persistence
    baseline metrics.

    If baseline CSV is unavailable, this step
    is skipped safely.
    """

    if not (
        BASELINE_METRICS_FILE.exists()
    ):

        print(
            "[WARNING] "
            "Baseline metrics file missing."
        )

        return pd.DataFrame()

    baseline = pd.read_csv(
        BASELINE_METRICS_FILE
    )

    persistence = (
        baseline[
            (
                baseline[
                    "split"
                ]
                ==
                "test"
            )
            &
            (
                baseline[
                    "model"
                ]
                ==
                "persistence"
            )
        ]
        .copy()
    )

    persistence = (
        persistence[
            [
                "horizon_weeks",
                "mae",
                "rmse",
                "smape",
            ]
        ]
        .rename(
            columns={
                "mae":
                "persistence_mae",

                "rmse":
                "persistence_rmse",

                "smape":
                "persistence_smape",
            }
        )
    )

    rf_test = (
        rf_metrics[
            rf_metrics[
                "split"
            ]
            ==
            "test"
        ][
            [
                "horizon_weeks",
                "mae",
                "rmse",
                "smape",
            ]
        ]
        .copy()
        .rename(
            columns={
                "mae":
                "random_forest_mae",

                "rmse":
                "random_forest_rmse",

                "smape":
                "random_forest_smape",
            }
        )
    )

    comparison = (
        persistence.merge(
            rf_test,
            on="horizon_weeks",
            how="inner",
            validate="one_to_one",
        )
    )

    comparison[
        "mae_improvement_percent"
    ] = (
        (
            comparison[
                "persistence_mae"
            ]
            -
            comparison[
                "random_forest_mae"
            ]
        )
        /
        comparison[
            "persistence_mae"
        ]
        *
        100.0
    )

    comparison[
        "rmse_improvement_percent"
    ] = (
        (
            comparison[
                "persistence_rmse"
            ]
            -
            comparison[
                "random_forest_rmse"
            ]
        )
        /
        comparison[
            "persistence_rmse"
        ]
        *
        100.0
    )

    comparison[
        "smape_improvement_percent"
    ] = (
        (
            comparison[
                "persistence_smape"
            ]
            -
            comparison[
                "random_forest_smape"
            ]
        )
        /
        comparison[
            "persistence_smape"
        ]
        *
        100.0
    )

    return comparison


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
        "RANDOM FOREST FORECASTING"
    )

    print(
        "=========================================="
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_data()

    all_metrics = []

    all_search_rows = []

    all_predictions = []

    all_importances = []

    for horizon in (
        HORIZONS
    ):

        result = (
            train_horizon(
                df,
                horizon,
            )
        )

        all_metrics.extend(
            result[
                "metrics"
            ]
        )

        all_search_rows.extend(
            result[
                "search"
            ]
        )

        all_predictions.append(
            result[
                "predictions"
            ]
        )

        all_importances.append(
            result[
                "feature_importance"
            ]
        )

    # ========================================================
    # SAVE OVERALL RESULTS
    # ========================================================

    metrics_df = pd.DataFrame(
        all_metrics
    )

    search_df = pd.DataFrame(
        all_search_rows
    )

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    feature_importance_df = (
        pd.concat(
            all_importances,
            ignore_index=True,
        )
    )

    metrics_df.to_csv(
        METRICS_FILE,
        index=False,
    )

    search_df.to_csv(
        SEARCH_FILE,
        index=False,
    )

    predictions_df.to_csv(
        PREDICTIONS_FILE,
        index=False,
        date_format="%Y-%m-%d",
    )

    feature_importance_df.to_csv(
        FEATURE_IMPORTANCE_FILE,
        index=False,
    )

    # ========================================================
    # COMPARE TO BASELINE
    # ========================================================

    comparison = (
        create_baseline_comparison(
            metrics_df
        )
    )

    if not comparison.empty:

        comparison.to_csv(
            COMPARISON_FILE,
            index=False,
        )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print(
        "=========================================="
    )

    print(
        "RANDOM FOREST METRICS"
    )

    print(
        "=========================================="
    )

    print(
        metrics_df.to_string(
            index=False
        )
    )

    if not comparison.empty:

        print()
        print(
            "=========================================="
        )

        print(
            "RANDOM FOREST VS PERSISTENCE"
        )

        print(
            "=========================================="
        )

        print(
            comparison.to_string(
                index=False
            )
        )

    print()
    print(
        "=========================================="
    )

    print(
        "TOP FEATURES"
    )

    print(
        "=========================================="
    )

    for horizon in HORIZONS:

        print()
        print(
            f"{horizon}-week:"
        )

        top = (
            feature_importance_df[
                feature_importance_df[
                    "horizon_weeks"
                ]
                ==
                horizon
            ]
            .head(10)
        )

        print(
            top[
                [
                    "feature",
                    "importance",
                ]
            ]
            .to_string(
                index=False
            )
        )

    print()
    print(
        "[PASS] Saved:"
    )

    print(
        f"       {METRICS_FILE}"
    )

    print(
        f"       {SEARCH_FILE}"
    )

    print(
        f"       {PREDICTIONS_FILE}"
    )

    print(
        f"       {FEATURE_IMPORTANCE_FILE}"
    )

    if not comparison.empty:

        print(
            f"       {COMPARISON_FILE}"
        )

    print()
    print(
        "=========================================="
    )

    print(
        "RANDOM FOREST STAGE COMPLETE"
    )

    print(
        "=========================================="
    )


if __name__ == "__main__":
    main()