"""
DengueShield AI
Leakage-Safe XGBoost Dengue Forecasting

Forecast horizons:
    1 week
    2 weeks
    4 weeks

Split:
    target date <= 2023 -> train
    target date == 2024 -> validation
    target date == 2025 -> final test

Important:
    Validation selects hyperparameters.
    2025 test data is never used for model selection.

Models are compared against:
    persistence baseline
    Random Forest

District:
    one-hot encoded
"""

from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


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

INPUT_FILE = Path(
    "data/processed/"
    "ml_features_2015_2025.csv"
)

BASELINE_FILE = Path(
    "models/baseline/"
    "baseline_metrics.csv"
)

RANDOM_FOREST_FILE = Path(
    "models/random_forest/"
    "random_forest_metrics.csv"
)

OUTPUT_DIR = Path(
    "models/xgboost"
)

METRICS_FILE = (
    OUTPUT_DIR
    / "xgboost_metrics.csv"
)

SEARCH_FILE = (
    OUTPUT_DIR
    / "xgboost_validation_search.csv"
)

PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "xgboost_predictions.csv"
)

IMPORTANCE_FILE = (
    OUTPUT_DIR
    / "xgboost_feature_importance.csv"
)

COMPARISON_FILE = (
    OUTPUT_DIR
    / "xgboost_model_comparison.csv"
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
# VALIDATION-ONLY PARAMETER CANDIDATES
# ============================================================

PARAMETER_CANDIDATES = [

    {
        "name":
        "xgb_conservative",

        "n_estimators":
        400,

        "learning_rate":
        0.03,

        "max_depth":
        4,

        "min_child_weight":
        5,

        "subsample":
        0.8,

        "colsample_bytree":
        0.8,

        "reg_alpha":
        0.0,

        "reg_lambda":
        2.0,
    },

    {
        "name":
        "xgb_balanced",

        "n_estimators":
        500,

        "learning_rate":
        0.03,

        "max_depth":
        6,

        "min_child_weight":
        5,

        "subsample":
        0.8,

        "colsample_bytree":
        0.8,

        "reg_alpha":
        0.1,

        "reg_lambda":
        3.0,
    },

    {
        "name":
        "xgb_regularized",

        "n_estimators":
        500,

        "learning_rate":
        0.025,

        "max_depth":
        5,

        "min_child_weight":
        10,

        "subsample":
        0.85,

        "colsample_bytree":
        0.75,

        "reg_alpha":
        0.5,

        "reg_lambda":
        5.0,
    },

]


# ============================================================
# LOAD
# ============================================================

def load_data():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Missing file: "
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

    missing = (
        set(MODEL_FEATURES)
        -
        set(df.columns)
    )

    if missing:

        raise ValueError(
            f"Missing features: "
            f"{sorted(missing)}"
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
        f"[PASS] XGBoost version: "
        f"{xgb.__version__}"
    )

    return df


# ============================================================
# METRICS
# ============================================================

def smape(
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

    return {

        "mae":
        float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        ),

        "rmse":
        float(
            mean_squared_error(
                y_true,
                y_pred,
            )
            ** 0.5
        ),

        "smape":
        smape(
            y_true,
            y_pred,
        ),
    }


# ============================================================
# SPLIT
# ============================================================

def prepare_horizon_data(
    df,
    horizon,
):

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

    if (
        train.empty
        or validation.empty
        or test.empty
    ):

        raise AssertionError(
            "A required split is empty."
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

    model = xgb.XGBRegressor(

        objective=(
            "reg:squarederror"
        ),

        tree_method="hist",

        n_estimators=(
            parameters[
                "n_estimators"
            ]
        ),

        learning_rate=(
            parameters[
                "learning_rate"
            ]
        ),

        max_depth=(
            parameters[
                "max_depth"
            ]
        ),

        min_child_weight=(
            parameters[
                "min_child_weight"
            ]
        ),

        subsample=(
            parameters[
                "subsample"
            ]
        ),

        colsample_bytree=(
            parameters[
                "colsample_bytree"
            ]
        ),

        reg_alpha=(
            parameters[
                "reg_alpha"
            ]
        ),

        reg_lambda=(
            parameters[
                "reg_lambda"
            ]
        ),

        random_state=(
            RANDOM_STATE
        ),

        n_jobs=-1,

        verbosity=0,
    )

    return Pipeline(
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


# ============================================================
# VALIDATION SEARCH
# ============================================================

def select_best_configuration(
    train,
    validation,
    target_column,
    horizon,
):

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

    best_pipeline = None

    best_parameters = None

    best_metrics = None

    best_mae = float(
        "inf"
    )

    search_rows = []

    print()
    print(
        "VALIDATION MODEL SELECTION"
    )

    for configuration in (
        PARAMETER_CANDIDATES
    ):

        print()
        print(
            f"  Training: "
            f"{configuration['name']}"
        )

        pipeline = (
            build_pipeline(
                configuration
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

        row = {
            "horizon_weeks":
            horizon,

            "configuration":
            configuration[
                "name"
            ],

            **{
                key: value
                for key, value
                in configuration.items()
                if key != "name"
            },

            "validation_mae":
            metrics[
                "mae"
            ],

            "validation_rmse":
            metrics[
                "rmse"
            ],

            "validation_smape":
            metrics[
                "smape"
            ],
        }

        search_rows.append(
            row
        )

        if (
            metrics[
                "mae"
            ]
            <
            best_mae
        ):

            best_mae = (
                metrics[
                    "mae"
                ]
            )

            best_pipeline = (
                pipeline
            )

            best_parameters = (
                configuration.copy()
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
# FEATURE IMPORTANCE
# ============================================================

def extract_feature_importance(
    pipeline,
    horizon,
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

    names = (
        preprocessor
        .get_feature_names_out()
    )

    importance = (
        model.feature_importances_
    )

    result = pd.DataFrame(
        {
            "horizon_weeks":
            horizon,

            "feature":
            names,

            "importance":
            importance,
        }
    )

    return (
        result
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# PREDICTION TABLE
# ============================================================

def make_prediction_table(
    data,
    predictions,
    horizon,
    split_name,
    target_column,
    target_date_column,
):

    return pd.DataFrame(
        {
            "district":
            data[
                "district"
            ].values,

            "forecast_origin_date":
            data[
                "forecast_origin_date"
            ].values,

            "target_date":
            data[
                target_date_column
            ].values,

            "horizon_weeks":
            horizon,

            "split":
            split_name,

            "actual":
            data[
                target_column
            ].values,

            "prediction_xgboost":
            predictions,

            "prediction_persistence":
            data[
                "cases_current"
            ].values,
        }
    )


# ============================================================
# ONE HORIZON
# ============================================================

def train_horizon(
    df,
    horizon,
):

    print()
    print(
        "=" * 70
    )

    print(
        f"{horizon}-WEEK XGBOOST"
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

    (
        validation_model,
        best_parameters,
        validation_metrics,
        search_rows,
    ) = select_best_configuration(
        train,
        validation,
        target_column,
        horizon,
    )

    # --------------------------------------------------------
    # VALIDATION PREDICTIONS
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

    validation_table = (
        make_prediction_table(
            validation,
            validation_predictions,
            horizon,
            "validation",
            target_column,
            target_date_column,
        )
    )

    # --------------------------------------------------------
    # REFIT TRAIN + VALIDATION
    # --------------------------------------------------------

    train_validation = pd.concat(
        [
            train,
            validation,
        ],
        ignore_index=True,
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
    # TEST
    # --------------------------------------------------------

    test_predictions = (
        final_pipeline.predict(
            test[
                MODEL_FEATURES
            ]
        )
    )

    test_predictions = np.clip(
        test_predictions,
        0,
        None,
    )

    test_metrics = (
        calculate_metrics(
            test[
                target_column
            ],
            test_predictions,
        )
    )

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

    print()
    print(
        f"MAE IMPROVEMENT "
        f"VS PERSISTENCE: "
        f"{mae_improvement:.2f}%"
    )

    # --------------------------------------------------------
    # SAVE FINAL MODEL
    # --------------------------------------------------------

    model_file = (
        OUTPUT_DIR
        /
        f"xgboost_{horizon}w.joblib"
    )

    joblib.dump(
        final_pipeline,
        model_file,
    )

    # --------------------------------------------------------
    # SAVE PARAMETERS
    # --------------------------------------------------------

    params_file = (
        OUTPUT_DIR
        /
        f"xgboost_{horizon}w_params.json"
    )

    with open(
        params_file,
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            best_parameters,
            handle,
            indent=4,
        )

    # --------------------------------------------------------
    # TEST TABLE
    # --------------------------------------------------------

    test_table = (
        make_prediction_table(
            test,
            test_predictions,
            horizon,
            "test",
            target_column,
            target_date_column,
        )
    )

    metrics_rows = [

        {
            "horizon_weeks":
            horizon,

            "split":
            "validation",

            "model":
            "xgboost",

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
            "xgboost",

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

    importance = (
        extract_feature_importance(
            final_pipeline,
            horizon,
        )
    )

    predictions = pd.concat(
        [
            validation_table,
            test_table,
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

        "importance":
        importance,
    }


# ============================================================
# MODEL COMPARISON
# ============================================================

def build_model_comparison(
    xgb_metrics,
):

    frames = []

    # --------------------------------------------------------
    # Persistence
    # --------------------------------------------------------

    if BASELINE_FILE.exists():

        baseline = pd.read_csv(
            BASELINE_FILE
        )

        baseline = baseline[
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
        ].copy()

        baseline[
            "model"
        ] = "persistence"

        frames.append(
            baseline[
                [
                    "horizon_weeks",
                    "model",
                    "mae",
                    "rmse",
                    "smape",
                ]
            ]
        )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    if RANDOM_FOREST_FILE.exists():

        rf = pd.read_csv(
            RANDOM_FOREST_FILE
        )

        rf = rf[
            rf[
                "split"
            ]
            ==
            "test"
        ].copy()

        frames.append(
            rf[
                [
                    "horizon_weeks",
                    "model",
                    "mae",
                    "rmse",
                    "smape",
                ]
            ]
        )

    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    xgb_test = xgb_metrics[
        xgb_metrics[
            "split"
        ]
        ==
        "test"
    ].copy()

    frames.append(
        xgb_test[
            [
                "horizon_weeks",
                "model",
                "mae",
                "rmse",
                "smape",
            ]
        ]
    )

    comparison = pd.concat(
        frames,
        ignore_index=True,
    )

    comparison = (
        comparison
        .sort_values(
            [
                "horizon_weeks",
                "mae",
            ]
        )
        .reset_index(
            drop=True
        )
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
        "XGBOOST FORECASTING"
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

    all_search = []

    all_predictions = []

    all_importance = []

    for horizon in HORIZONS:

        result = train_horizon(
            df,
            horizon,
        )

        all_metrics.extend(
            result[
                "metrics"
            ]
        )

        all_search.extend(
            result[
                "search"
            ]
        )

        all_predictions.append(
            result[
                "predictions"
            ]
        )

        all_importance.append(
            result[
                "importance"
            ]
        )

    metrics_df = pd.DataFrame(
        all_metrics
    )

    search_df = pd.DataFrame(
        all_search
    )

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    importance_df = pd.concat(
        all_importance,
        ignore_index=True,
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

    importance_df.to_csv(
        IMPORTANCE_FILE,
        index=False,
    )

    comparison = (
        build_model_comparison(
            metrics_df
        )
    )

    comparison.to_csv(
        COMPARISON_FILE,
        index=False,
    )

    print()
    print(
        "=========================================="
    )

    print(
        "XGBOOST RESULTS"
    )

    print(
        "=========================================="
    )

    print(
        metrics_df.to_string(
            index=False
        )
    )

    print()
    print(
        "=========================================="
    )

    print(
        "FINAL MODEL COMPARISON"
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
        "TOP XGBOOST FEATURES"
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
            importance_df[
                importance_df[
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
        f"       {IMPORTANCE_FILE}"
    )

    print(
        f"       {COMPARISON_FILE}"
    )

    print()
    print(
        "=========================================="
    )

    print(
        "XGBOOST STAGE COMPLETE"
    )

    print(
        "=========================================="
    )


if __name__ == "__main__":
    main()