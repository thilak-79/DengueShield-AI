"""
DengueShield AI
Leakage-Safe LightGBM Dengue Forecasting

Forecast horizons:
    1 week
    2 weeks
    4 weeks

Temporal split:
    target date <= 2023 -> train
    target date == 2024 -> validation
    target date == 2025 -> held-out evaluation

Model selection:
    Hyperparameters are selected using 2024 validation MAE.

After selection:
    Best configuration is refitted using train + validation.
    Performance is then reported for 2025.

Primary metric:
    MAE

Secondary metrics:
    RMSE
    sMAPE

District:
    one-hot encoded
"""

from pathlib import Path
import json
import sys

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

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

XGBOOST_FILE = Path(
    "models/xgboost/"
    "xgboost_metrics.csv"
)

OUTPUT_DIR = Path(
    "models/lightgbm"
)

METRICS_FILE = (
    OUTPUT_DIR
    / "lightgbm_metrics.csv"
)

SEARCH_FILE = (
    OUTPUT_DIR
    / "lightgbm_validation_search.csv"
)

PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "lightgbm_predictions.csv"
)

IMPORTANCE_FILE = (
    OUTPUT_DIR
    / "lightgbm_feature_importance.csv"
)

COMPARISON_FILE = (
    OUTPUT_DIR
    / "lightgbm_model_comparison.csv"
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
# SMALL VALIDATION-ONLY PARAMETER SEARCH
# ============================================================

PARAMETER_CANDIDATES = [

    {
        "name":
        "lgbm_conservative",

        "n_estimators":
        400,

        "learning_rate":
        0.03,

        "num_leaves":
        20,

        "max_depth":
        5,

        "min_child_samples":
        30,

        "subsample":
        0.8,

        "colsample_bytree":
        0.8,

        "reg_alpha":
        0.1,

        "reg_lambda":
        2.0,
    },

    {
        "name":
        "lgbm_balanced",

        "n_estimators":
        500,

        "learning_rate":
        0.025,

        "num_leaves":
        31,

        "max_depth":
        7,

        "min_child_samples":
        25,

        "subsample":
        0.8,

        "colsample_bytree":
        0.8,

        "reg_alpha":
        0.2,

        "reg_lambda":
        3.0,
    },

    {
        "name":
        "lgbm_regularized",

        "n_estimators":
        500,

        "learning_rate":
        0.025,

        "num_leaves":
        24,

        "max_depth":
        6,

        "min_child_samples":
        40,

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
# LOAD DATA
# ============================================================

def load_data():

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
            f"Missing features: "
            f"{sorted(missing_features)}"
        )

    if (
        df[
            MODEL_FEATURES
        ]
        .drop(
            columns=[
                "district"
            ]
        )
        .isna()
        .any()
        .any()
    ):

        raise AssertionError(
            "Missing numerical model features."
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

    print(
        f"[PASS] LightGBM version: "
        f"{lgb.__version__}"
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
# TEMPORAL SPLIT
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

    if (
        train.empty
        or validation.empty
        or test.empty
    ):

        raise AssertionError(
            "One temporal split is empty."
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
# BUILD PIPELINE
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

    model = lgb.LGBMRegressor(

        objective="regression",

        boosting_type="gbdt",

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

        num_leaves=(
            parameters[
                "num_leaves"
            ]
        ),

        max_depth=(
            parameters[
                "max_depth"
            ]
        ),

        min_child_samples=(
            parameters[
                "min_child_samples"
            ]
        ),

        subsample=(
            parameters[
                "subsample"
            ]
        ),

        # Required so subsample is actually used.
        subsample_freq=1,

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

        importance_type="gain",

        verbosity=-1,

        deterministic=True,

        force_col_wise=True,
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
# VALIDATION MODEL SELECTION
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

        search_rows.append(
            {
                "horizon_weeks":
                horizon,

                "configuration":
                configuration[
                    "name"
                ],

                "n_estimators":
                configuration[
                    "n_estimators"
                ],

                "learning_rate":
                configuration[
                    "learning_rate"
                ],

                "num_leaves":
                configuration[
                    "num_leaves"
                ],

                "max_depth":
                configuration[
                    "max_depth"
                ],

                "min_child_samples":
                configuration[
                    "min_child_samples"
                ],

                "subsample":
                configuration[
                    "subsample"
                ],

                "colsample_bytree":
                configuration[
                    "colsample_bytree"
                ],

                "reg_alpha":
                configuration[
                    "reg_alpha"
                ],

                "reg_lambda":
                configuration[
                    "reg_lambda"
                ],

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
# PREDICTION TABLE
# ============================================================

def make_prediction_table(
    source_df,
    predictions,
    horizon,
    split_name,
    target_column,
    target_date_column,
    training_stage,
):

    return pd.DataFrame(
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

            "prediction_lightgbm":
            predictions,

            "prediction_persistence":
            source_df[
                "cases_current"
            ].values,
        }
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
            "Feature names and "
            "importances do not match."
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

    print()
    print(
        "=" * 70
    )

    print(
        f"{horizon}-WEEK LIGHTGBM"
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
    # SELECT ON 2024
    # --------------------------------------------------------

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

    validation_predictions = np.clip(
        validation_predictions,
        0,
        None,
    )

    validation_table = (
        make_prediction_table(
            source_df=validation,
            predictions=(
                validation_predictions
            ),
            horizon=horizon,
            split_name="validation",
            target_column=(
                target_column
            ),
            target_date_column=(
                target_date_column
            ),
            training_stage="train_only",
        )
    )

    # --------------------------------------------------------
    # REFIT TRAIN + VALIDATION
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
    # 2025 EVALUATION
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
        "2025 HELD-OUT EVALUATION"
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
        "PERSISTENCE"
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

    improvement = (
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
        f"MAE improvement "
        f"vs persistence: "
        f"{improvement:.2f}%"
    )

    # --------------------------------------------------------
    # TEST PREDICTIONS
    # --------------------------------------------------------

    test_table = (
        make_prediction_table(
            source_df=test,
            predictions=(
                test_predictions
            ),
            horizon=horizon,
            split_name="test",
            target_column=(
                target_column
            ),
            target_date_column=(
                target_date_column
            ),
            training_stage=(
                "train_plus_validation"
            ),
        )
    )

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    model_file = (
        OUTPUT_DIR
        /
        f"lightgbm_{horizon}w.joblib"
    )

    joblib.dump(
        final_pipeline,
        model_file,
    )

    print()
    print(
        "[PASS] Saved model:"
    )

    print(
        f"       {model_file}"
    )

    # --------------------------------------------------------
    # SAVE PARAMETERS
    # --------------------------------------------------------

    params_file = (
        OUTPUT_DIR
        /
        f"lightgbm_{horizon}w_params.json"
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
    # METRIC ROWS
    # --------------------------------------------------------

    metrics_rows = [

        {
            "horizon_weeks":
            horizon,

            "split":
            "validation",

            "model":
            "lightgbm",

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
            "lightgbm",

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
# BUILD FOUR-MODEL COMPARISON
# ============================================================

def build_comparison(
    lightgbm_metrics,
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

    if XGBOOST_FILE.exists():

        xgb = pd.read_csv(
            XGBOOST_FILE
        )

        xgb = xgb[
            xgb[
                "split"
            ]
            ==
            "test"
        ].copy()

        frames.append(
            xgb[
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
    # LightGBM
    # --------------------------------------------------------

    lgb_test = (
        lightgbm_metrics[
            lightgbm_metrics[
                "split"
            ]
            ==
            "test"
        ]
        .copy()
    )

    frames.append(
        lgb_test[
            [
                "horizon_weeks",
                "model",
                "mae",
                "rmse",
                "smape",
            ]
        ]
    )

    result = pd.concat(
        frames,
        ignore_index=True,
    )

    result = (
        result.sort_values(
            [
                "horizon_weeks",
                "mae",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return result


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
        "LIGHTGBM FORECASTING"
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

    comparison = build_comparison(
        metrics_df
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
        "LIGHTGBM RESULTS"
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
        "FOUR-MODEL COMPARISON"
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
        "TOP LIGHTGBM FEATURES"
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
        "LIGHTGBM STAGE COMPLETE"
    )

    print(
        "=========================================="
    )


if __name__ == "__main__":
    main()