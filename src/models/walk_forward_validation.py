"""
DengueShield AI
Nested Walk-Forward / Expanding-Window Validation

Purpose
-------
Evaluate forecasting performance across multiple historical years
without using future years to select model settings.

For test year Y:

    Train:      target year <= Y - 2
    Validation: target year == Y - 1
    Test:       target year == Y

Example:

    Test 2023:
        Train      <= 2021
        Validation = 2022
        Test       = 2023

The validation year chooses model configuration.
The selected configuration is then refitted on:

    Train + Validation

before predicting the test year.

Models
------
1. Persistence baseline
2. Random Forest
3. LightGBM

Forecast horizons
-----------------
1 week
2 weeks
4 weeks

Primary metric
--------------
MAE

Secondary metrics
-----------------
RMSE
sMAPE
"""

from pathlib import Path
import gc
import sys

import lightgbm as lgb
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

OUTPUT_DIR = Path(
    "models/walk_forward"
)

METRICS_FILE = (
    OUTPUT_DIR
    / "walk_forward_metrics.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "walk_forward_summary.csv"
)

PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "walk_forward_predictions.csv"
)

SEARCH_FILE = (
    OUTPUT_DIR
    / "walk_forward_validation_search.csv"
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

TEST_YEARS = [
    2020,
    2021,
    2022,
    2023,
    2024,
    2025,
]


# ============================================================
# MODEL FEATURES
# ============================================================

NUMERIC_FEATURES = [

    # Current dengue state
    "cases_current",

    # Current weather
    "temperature_current",
    "temperature_min_current",
    "temperature_max_current",
    "humidity_current",
    "rainfall_current",
    "rain_days_current",

    # Dengue history
    "cases_lag_1",
    "cases_lag_2",
    "cases_lag_3",
    "cases_lag_4",

    "cases_rolling_2",
    "cases_rolling_4",

    # Rainfall history
    "rainfall_lag_1",
    "rainfall_lag_2",
    "rainfall_lag_3",
    "rainfall_lag_4",

    "rainfall_rolling_4",

    # Humidity history
    "humidity_lag_1",
    "humidity_lag_2",
    "humidity_lag_3",
    "humidity_lag_4",

    "humidity_rolling_4",

    # Temperature history
    "temperature_lag_1",
    "temperature_lag_2",
    "temperature_lag_3",
    "temperature_lag_4",

    "temperature_rolling_4",

    # Rain-day history
    "rain_days_lag_1",
    "rain_days_lag_2",
    "rain_days_lag_3",
    "rain_days_lag_4",

    # Seasonality
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
# RANDOM FOREST CANDIDATES
# ============================================================

RF_CANDIDATES = [

    {
        "name":
        "rf_depth12",

        "n_estimators":
        300,

        "max_depth":
        12,

        "min_samples_split":
        2,

        "min_samples_leaf":
        2,

        "max_features":
        0.7,
    },

    {
        "name":
        "rf_depth18",

        "n_estimators":
        400,

        "max_depth":
        18,

        "min_samples_split":
        2,

        "min_samples_leaf":
        2,

        "max_features":
        0.8,
    },

    {
        "name":
        "rf_regularized",

        "n_estimators":
        400,

        "max_depth":
        None,

        "min_samples_split":
        4,

        "min_samples_leaf":
        3,

        "max_features":
        1.0,
    },
]


# ============================================================
# LIGHTGBM CANDIDATES
# ============================================================

LGBM_CANDIDATES = [

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
            f"Missing input file: "
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
        f"[PASS] Years: "
        f"{df['year'].min()} "
        f"- "
        f"{df['year'].max()}"
    )

    print(
        f"[PASS] LightGBM: "
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
# PREPROCESSOR
# ============================================================

def build_preprocessor():

    return ColumnTransformer(
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


# ============================================================
# RANDOM FOREST PIPELINE
# ============================================================

def build_rf_pipeline(
    parameters,
):

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

        criterion="squared_error",

        random_state=(
            RANDOM_STATE
        ),

        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),

            (
                "model",
                model,
            ),
        ]
    )


# ============================================================
# LIGHTGBM PIPELINE
# ============================================================

def build_lgbm_pipeline(
    parameters,
):

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

        # Necessary to activate row subsampling.
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

        verbosity=-1,

        deterministic=True,

        force_col_wise=True,
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),

            (
                "model",
                model,
            ),
        ]
    )


# ============================================================
# TEMPORAL FOLD
# ============================================================

def create_fold(
    df,
    horizon,
    test_year,
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

    working[
        "target_year"
    ] = (
        working[
            target_date_column
        ]
        .dt.year
    )

    validation_year = (
        test_year
        -
        1
    )

    train_max_year = (
        test_year
        -
        2
    )

    train = (
        working[
            working[
                "target_year"
            ]
            <=
            train_max_year
        ]
        .copy()
    )

    validation = (
        working[
            working[
                "target_year"
            ]
            ==
            validation_year
        ]
        .copy()
    )

    test = (
        working[
            working[
                "target_year"
            ]
            ==
            test_year
        ]
        .copy()
    )

    if train.empty:

        raise AssertionError(
            f"Train empty for "
            f"{test_year}, "
            f"{horizon}w"
        )

    if validation.empty:

        raise AssertionError(
            f"Validation empty for "
            f"{test_year}, "
            f"{horizon}w"
        )

    if test.empty:

        raise AssertionError(
            f"Test empty for "
            f"{test_year}, "
            f"{horizon}w"
        )

    return (
        train,
        validation,
        test,
        target_column,
        target_date_column,
        validation_year,
        train_max_year,
    )


# ============================================================
# CONFIGURATION SEARCH
# ============================================================

def select_configuration(
    model_family,
    candidates,
    train,
    validation,
    target_column,
    horizon,
    test_year,
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

    best_config = None
    best_metrics = None
    best_mae = float(
        "inf"
    )

    search_rows = []

    for config in candidates:

        if (
            model_family
            ==
            "random_forest"
        ):

            pipeline = (
                build_rf_pipeline(
                    config
                )
            )

        elif (
            model_family
            ==
            "lightgbm"
        ):

            pipeline = (
                build_lgbm_pipeline(
                    config
                )
            )

        else:

            raise ValueError(
                f"Unknown model family: "
                f"{model_family}"
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

        search_rows.append(
            {
                "test_year":
                test_year,

                "horizon_weeks":
                horizon,

                "model":
                model_family,

                "configuration":
                config[
                    "name"
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

            best_config = (
                config.copy()
            )

            best_metrics = (
                metrics.copy()
            )

        del pipeline

        gc.collect()

    # Mark selected configuration
    for row in search_rows:

        row[
            "selected"
        ] = (
            row[
                "configuration"
            ]
            ==
            best_config[
                "name"
            ]
        )

    return (
        best_config,
        best_metrics,
        search_rows,
    )


# ============================================================
# FIT FINAL MODEL
# ============================================================

def fit_final_model(
    model_family,
    parameters,
    train_validation,
    test,
    target_column,
):

    if (
        model_family
        ==
        "random_forest"
    ):

        pipeline = (
            build_rf_pipeline(
                parameters
            )
        )

    elif (
        model_family
        ==
        "lightgbm"
    ):

        pipeline = (
            build_lgbm_pipeline(
                parameters
            )
        )

    else:

        raise ValueError(
            f"Unknown model family: "
            f"{model_family}"
        )

    pipeline.fit(
        train_validation[
            MODEL_FEATURES
        ],
        train_validation[
            target_column
        ],
    )

    predictions = (
        pipeline.predict(
            test[
                MODEL_FEATURES
            ]
        )
    )

    predictions = np.clip(
        predictions,
        0,
        None,
    )

    return (
        pipeline,
        predictions,
    )


# ============================================================
# ONE WALK-FORWARD FOLD
# ============================================================

def evaluate_fold(
    df,
    horizon,
    test_year,
):

    print()
    print(
        "=" * 75
    )

    print(
        f"HORIZON: {horizon} WEEK(S)"
        f" | TEST YEAR: {test_year}"
    )

    print(
        "=" * 75
    )

    (
        train,
        validation,
        test,
        target_column,
        target_date_column,
        validation_year,
        train_max_year,
    ) = create_fold(
        df,
        horizon,
        test_year,
    )

    print(
        f"Train target years: "
        f"<= {train_max_year}"
    )

    print(
        f"Validation year:     "
        f"{validation_year}"
    )

    print(
        f"Test year:           "
        f"{test_year}"
    )

    print(
        f"Train rows:          "
        f"{len(train):,}"
    )

    print(
        f"Validation rows:     "
        f"{len(validation):,}"
    )

    print(
        f"Test rows:           "
        f"{len(test):,}"
    )

    # ========================================================
    # PERSISTENCE
    # ========================================================

    persistence_predictions = (
        test[
            "cases_current"
        ]
        .to_numpy(
            dtype=float
        )
    )

    persistence_metrics = (
        calculate_metrics(
            test[
                target_column
            ],
            persistence_predictions,
        )
    )

    print()
    print(
        "Persistence:"
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

    # ========================================================
    # RANDOM FOREST MODEL SELECTION
    # ========================================================

    (
        rf_config,
        rf_validation_metrics,
        rf_search,
    ) = select_configuration(
        model_family="random_forest",
        candidates=RF_CANDIDATES,
        train=train,
        validation=validation,
        target_column=target_column,
        horizon=horizon,
        test_year=test_year,
    )

    print()
    print(
        "Random Forest selected:"
    )

    print(
        f"  {rf_config['name']}"
    )

    print(
        f"  Validation MAE: "
        f"{rf_validation_metrics['mae']:.4f}"
    )

    # ========================================================
    # LIGHTGBM MODEL SELECTION
    # ========================================================

    (
        lgbm_config,
        lgbm_validation_metrics,
        lgbm_search,
    ) = select_configuration(
        model_family="lightgbm",
        candidates=LGBM_CANDIDATES,
        train=train,
        validation=validation,
        target_column=target_column,
        horizon=horizon,
        test_year=test_year,
    )

    print()
    print(
        "LightGBM selected:"
    )

    print(
        f"  {lgbm_config['name']}"
    )

    print(
        f"  Validation MAE: "
        f"{lgbm_validation_metrics['mae']:.4f}"
    )

    # ========================================================
    # REFIT USING ALL PRE-TEST DATA
    # ========================================================

    train_validation = (
        pd.concat(
            [
                train,
                validation,
            ],
            ignore_index=True,
        )
    )

    # --------------------------------------------------------
    # RANDOM FOREST FINAL
    # --------------------------------------------------------

    (
        rf_model,
        rf_predictions,
    ) = fit_final_model(
        model_family="random_forest",
        parameters=rf_config,
        train_validation=(
            train_validation
        ),
        test=test,
        target_column=target_column,
    )

    rf_metrics = (
        calculate_metrics(
            test[
                target_column
            ],
            rf_predictions,
        )
    )

    # --------------------------------------------------------
    # LIGHTGBM FINAL
    # --------------------------------------------------------

    (
        lgbm_model,
        lgbm_predictions,
    ) = fit_final_model(
        model_family="lightgbm",
        parameters=lgbm_config,
        train_validation=(
            train_validation
        ),
        test=test,
        target_column=target_column,
    )

    lgbm_metrics = (
        calculate_metrics(
            test[
                target_column
            ],
            lgbm_predictions,
        )
    )

    print()
    print(
        "TEST RESULTS"
    )

    print(
        f"Random Forest:"
        f" MAE={rf_metrics['mae']:.4f}"
        f" RMSE={rf_metrics['rmse']:.4f}"
        f" sMAPE={rf_metrics['smape']:.2f}%"
    )

    print(
        f"LightGBM:    "
        f" MAE={lgbm_metrics['mae']:.4f}"
        f" RMSE={lgbm_metrics['rmse']:.4f}"
        f" sMAPE={lgbm_metrics['smape']:.2f}%"
    )

    # ========================================================
    # METRIC ROWS
    # ========================================================

    metric_rows = [

        {
            "test_year":
            test_year,

            "horizon_weeks":
            horizon,

            "model":
            "persistence",

            "mae":
            persistence_metrics[
                "mae"
            ],

            "rmse":
            persistence_metrics[
                "rmse"
            ],

            "smape":
            persistence_metrics[
                "smape"
            ],

            "selected_configuration":
            "none",

            "validation_mae":
            np.nan,

            "train_max_year":
            train_max_year,

            "validation_year":
            validation_year,

            "train_rows":
            len(train),

            "validation_rows":
            len(validation),

            "test_rows":
            len(test),
        },

        {
            "test_year":
            test_year,

            "horizon_weeks":
            horizon,

            "model":
            "random_forest",

            "mae":
            rf_metrics[
                "mae"
            ],

            "rmse":
            rf_metrics[
                "rmse"
            ],

            "smape":
            rf_metrics[
                "smape"
            ],

            "selected_configuration":
            rf_config[
                "name"
            ],

            "validation_mae":
            rf_validation_metrics[
                "mae"
            ],

            "train_max_year":
            train_max_year,

            "validation_year":
            validation_year,

            "train_rows":
            len(train),

            "validation_rows":
            len(validation),

            "test_rows":
            len(test),
        },

        {
            "test_year":
            test_year,

            "horizon_weeks":
            horizon,

            "model":
            "lightgbm",

            "mae":
            lgbm_metrics[
                "mae"
            ],

            "rmse":
            lgbm_metrics[
                "rmse"
            ],

            "smape":
            lgbm_metrics[
                "smape"
            ],

            "selected_configuration":
            lgbm_config[
                "name"
            ],

            "validation_mae":
            lgbm_validation_metrics[
                "mae"
            ],

            "train_max_year":
            train_max_year,

            "validation_year":
            validation_year,

            "train_rows":
            len(train),

            "validation_rows":
            len(validation),

            "test_rows":
            len(test),
        },
    ]

    # ========================================================
    # PREDICTION TABLE
    # ========================================================

    predictions = pd.DataFrame(
        {
            "test_year":
            test_year,

            "horizon_weeks":
            horizon,

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
                target_date_column
            ].values,

            "actual":
            test[
                target_column
            ].values,

            "prediction_persistence":
            persistence_predictions,

            "prediction_random_forest":
            rf_predictions,

            "prediction_lightgbm":
            lgbm_predictions,

            "rf_configuration":
            rf_config[
                "name"
            ],

            "lightgbm_configuration":
            lgbm_config[
                "name"
            ],
        }
    )

    search_rows = (
        rf_search
        +
        lgbm_search
    )

    # Clean memory before next fold
    del rf_model
    del lgbm_model

    gc.collect()

    return (
        metric_rows,
        predictions,
        search_rows,
    )


# ============================================================
# ADD COMPARISONS TO PERSISTENCE
# ============================================================

def add_baseline_comparison(
    metrics_df,
):

    persistence = (
        metrics_df[
            metrics_df[
                "model"
            ]
            ==
            "persistence"
        ][
            [
                "test_year",
                "horizon_weeks",
                "mae",
            ]
        ]
        .rename(
            columns={
                "mae":
                "persistence_mae"
            }
        )
    )

    result = (
        metrics_df.merge(
            persistence,
            on=[
                "test_year",
                "horizon_weeks",
            ],
            how="left",
            validate="many_to_one",
        )
    )

    result[
        "mae_improvement_vs_persistence"
    ] = (
        result[
            "persistence_mae"
        ]
        -
        result[
            "mae"
        ]
    )

    result[
        "mae_improvement_vs_persistence_percent"
    ] = np.where(
        result[
            "model"
        ]
        ==
        "persistence",
        0.0,
        (
            result[
                "mae_improvement_vs_persistence"
            ]
            /
            result[
                "persistence_mae"
            ]
            *
            100.0
        ),
    )

    result[
        "beats_persistence"
    ] = (
        result[
            "mae"
        ]
        <
        result[
            "persistence_mae"
        ]
    )

    result.loc[
        result[
            "model"
        ]
        ==
        "persistence",
        "beats_persistence",
    ] = False

    result[
        "fold_mae_rank"
    ] = (
        result.groupby(
            [
                "test_year",
                "horizon_weeks",
            ]
        )[
            "mae"
        ]
        .rank(
            method="min",
            ascending=True,
        )
        .astype(int)
    )

    return result


# ============================================================
# SUMMARY ACROSS YEARS
# ============================================================

def build_summary(
    metrics_df,
):

    summary = (
        metrics_df.groupby(
            [
                "horizon_weeks",
                "model",
            ]
        )
        .agg(
            years_evaluated=(
                "test_year",
                "nunique",
            ),

            mean_mae=(
                "mae",
                "mean",
            ),

            std_mae=(
                "mae",
                "std",
            ),

            median_mae=(
                "mae",
                "median",
            ),

            mean_rmse=(
                "rmse",
                "mean",
            ),

            mean_smape=(
                "smape",
                "mean",
            ),

            years_mae_better_than_persistence=(
                "beats_persistence",
                "sum",
            ),

            mean_mae_improvement_percent=(
                "mae_improvement_vs_persistence_percent",
                "mean",
            ),

            years_with_lowest_mae=(
                "fold_mae_rank",
                lambda values: int(
                    (
                        values
                        ==
                        1
                    )
                    .sum()
                ),
            ),
        )
        .reset_index()
    )

    summary[
        "years_mae_better_than_persistence"
    ] = (
        summary[
            "years_mae_better_than_persistence"
        ]
        .astype(int)
    )

    summary[
        "years_with_lowest_mae"
    ] = (
        summary[
            "years_with_lowest_mae"
        ]
        .astype(int)
    )

    summary = (
        summary.sort_values(
            [
                "horizon_weeks",
                "mean_mae",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return summary


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
        "NESTED WALK-FORWARD VALIDATION"
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

    all_predictions = []

    all_search_rows = []

    # ========================================================
    # WALK FORWARD
    # ========================================================

    for horizon in HORIZONS:

        for test_year in TEST_YEARS:

            (
                metric_rows,
                predictions,
                search_rows,
            ) = evaluate_fold(
                df,
                horizon,
                test_year,
            )

            all_metrics.extend(
                metric_rows
            )

            all_predictions.append(
                predictions
            )

            all_search_rows.extend(
                search_rows
            )

    # ========================================================
    # COMBINE
    # ========================================================

    metrics_df = pd.DataFrame(
        all_metrics
    )

    metrics_df = (
        add_baseline_comparison(
            metrics_df
        )
    )

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    search_df = pd.DataFrame(
        all_search_rows
    )

    summary_df = (
        build_summary(
            metrics_df
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    metrics_df.to_csv(
        METRICS_FILE,
        index=False,
    )

    predictions_df.to_csv(
        PREDICTIONS_FILE,
        index=False,
        date_format="%Y-%m-%d",
    )

    search_df.to_csv(
        SEARCH_FILE,
        index=False,
    )

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    # ========================================================
    # PRINT
    # ========================================================

    print()
    print(
        "=========================================="
    )

    print(
        "WALK-FORWARD SUMMARY"
    )

    print(
        "=========================================="
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print(
        "=========================================="
    )

    print(
        "YEAR-BY-YEAR MAE"
    )

    print(
        "=========================================="
    )

    yearly = (
        metrics_df.pivot_table(
            index=[
                "test_year",
                "horizon_weeks",
            ],
            columns="model",
            values="mae",
        )
        .reset_index()
    )

    print(
        yearly.to_string(
            index=False
        )
    )

    print()
    print(
        "=========================================="
    )

    print(
        "SELECTED CONFIGURATION COUNTS"
    )

    print(
        "=========================================="
    )

    selected = (
        search_df[
            search_df[
                "selected"
            ]
            ==
            True
        ]
        .groupby(
            [
                "horizon_weeks",
                "model",
                "configuration",
            ]
        )
        .size()
        .reset_index(
            name="times_selected"
        )
    )

    print(
        selected.to_string(
            index=False
        )
    )

    print()
    print(
        "[PASS] Saved:"
    )

    print(
        f"       "
        f"{METRICS_FILE}"
    )

    print(
        f"       "
        f"{SUMMARY_FILE}"
    )

    print(
        f"       "
        f"{PREDICTIONS_FILE}"
    )

    print(
        f"       "
        f"{SEARCH_FILE}"
    )

    print()
    print(
        "=========================================="
    )

    print(
        "WALK-FORWARD VALIDATION COMPLETE"
    )

    print(
        "=========================================="
    )


if __name__ == "__main__":
    main()