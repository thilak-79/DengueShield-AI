"""
DengueShield AI
Baseline Forecasting Models

Baselines
---------
1. Persistence:
       prediction = cases_current

2. Recent 4-week mean:
       prediction = cases_rolling_4

Forecast horizons:
       1 week
       2 weeks
       4 weeks

Evaluation split is based on TARGET DATE:

       Train      <= 2023
       Validation = 2024
       Test       = 2025

Metrics:
       MAE
       RMSE
       sMAPE
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


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

RESULTS_DIR = Path(
    "models/baseline"
)

METRICS_FILE = (
    RESULTS_DIR
    / "baseline_metrics.csv"
)

PREDICTIONS_FILE = (
    RESULTS_DIR
    / "baseline_predictions.csv"
)


# ============================================================
# FORECAST SETTINGS
# ============================================================

HORIZONS = [
    1,
    2,
    4,
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

    print(
        f"[PASS] Loaded "
        f"{len(df):,} feature rows"
    )

    print(
        f"[PASS] Districts: "
        f"{df['district'].nunique()}"
    )

    return df


# ============================================================
# SMAPE
# ============================================================

def smape(
    y_true,
    y_pred,
):
    """
    Symmetric Mean Absolute Percentage Error.

    Zero/zero pairs contribute 0 without
    producing divide-by-zero warnings.
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

    values = np.zeros_like(
        denominator,
        dtype=float,
    )

    np.divide(
        numerator,
        denominator,
        out=values,
        where=denominator != 0,
    )

    return float(
        np.mean(values)
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
):

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
        "mae": mae,
        "rmse": rmse,
        "smape": smape_value,
    }


# ============================================================
# TARGET-DATE SPLIT
# ============================================================

def assign_split(
    target_dates,
):
    """
    Assign split using the FUTURE target date.

    This prevents a forecast made in late 2023
    with a 2024 target from leaking into training.
    """

    year = (
        target_dates
        .dt.year
    )

    split = np.select(
        [
            year <= 2023,
            year == 2024,
            year == 2025,
        ],
        [
            "train",
            "validation",
            "test",
        ],
        default="unused",
    )

    return split


# ============================================================
# EVALUATE ONE HORIZON
# ============================================================

def evaluate_horizon(
    df,
    horizon,
):

    print()
    print(
        "=" * 60
    )

    print(
        f"{horizon}-WEEK FORECAST"
    )

    print(
        "=" * 60
    )

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
        "split"
    ] = assign_split(
        working[
            target_date_column
        ]
    )

    # --------------------------------------------------------
    # BASELINE PREDICTIONS
    # --------------------------------------------------------

    working[
        "prediction_persistence"
    ] = working[
        "cases_current"
    ]

    working[
        "prediction_rolling4"
    ] = working[
        "cases_rolling_4"
    ]

    # Dengue counts cannot be negative.
    working[
        "prediction_persistence"
    ] = (
        working[
            "prediction_persistence"
        ]
        .clip(
            lower=0
        )
    )

    working[
        "prediction_rolling4"
    ] = (
        working[
            "prediction_rolling4"
        ]
        .clip(
            lower=0
        )
    )

    # --------------------------------------------------------
    # PRINT SPLIT COUNTS
    # --------------------------------------------------------

    print(
        "\nSplit counts:"
    )

    print(
        working[
            "split"
        ]
        .value_counts()
        .to_string()
    )

    metrics_rows = []

    prediction_rows = []

    # --------------------------------------------------------
    # EVALUATE VALIDATION + TEST
    # --------------------------------------------------------

    for split_name in [
        "validation",
        "test",
    ]:

        split_df = (
            working[
                working[
                    "split"
                ]
                ==
                split_name
            ]
            .copy()
        )

        if split_df.empty:

            raise AssertionError(
                f"No rows for "
                f"{split_name}"
            )

        y_true = (
            split_df[
                target_column
            ]
            .astype(float)
        )

        print()
        print(
            f"{split_name.upper()}"
        )

        # --------------------------------------------
        # Persistence
        # --------------------------------------------

        persistence_metrics = (
            calculate_metrics(
                y_true,
                split_df[
                    "prediction_persistence"
                ],
            )
        )

        metrics_rows.append(
            {
                "horizon_weeks":
                horizon,

                "split":
                split_name,

                "model":
                "persistence",

                **persistence_metrics,
            }
        )

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

        # --------------------------------------------
        # Rolling mean
        # --------------------------------------------

        rolling_metrics = (
            calculate_metrics(
                y_true,
                split_df[
                    "prediction_rolling4"
                ],
            )
        )

        metrics_rows.append(
            {
                "horizon_weeks":
                horizon,

                "split":
                split_name,

                "model":
                "rolling_4week_mean",

                **rolling_metrics,
            }
        )

        print(
            "4-week mean:"
        )

        print(
            f"  MAE:   "
            f"{rolling_metrics['mae']:.4f}"
        )

        print(
            f"  RMSE:  "
            f"{rolling_metrics['rmse']:.4f}"
        )

        print(
            f"  sMAPE: "
            f"{rolling_metrics['smape']:.2f}%"
        )

        # --------------------------------------------
        # Store predictions
        # --------------------------------------------

        temp = pd.DataFrame(
            {
                "district":
                split_df[
                    "district"
                ],

                "forecast_origin_date":
                split_df[
                    "forecast_origin_date"
                ],

                "target_date":
                split_df[
                    target_date_column
                ],

                "horizon_weeks":
                horizon,

                "split":
                split_name,

                "actual":
                y_true,

                "prediction_persistence":
                split_df[
                    "prediction_persistence"
                ],

                "prediction_rolling4":
                split_df[
                    "prediction_rolling4"
                ],
            }
        )

        prediction_rows.append(
            temp
        )

    return (
        metrics_rows,
        prediction_rows,
    )


# ============================================================
# DISTRICT-LEVEL TEST METRICS
# ============================================================

def evaluate_test_by_district(
    predictions,
):
    """
    Additional district-level evaluation.

    Useful later to see whether a model works well
    nationally but poorly in certain districts.
    """

    print()
    print(
        "=" * 60
    )

    print(
        "TEST PERFORMANCE BY DISTRICT"
    )

    print(
        "=" * 60
    )

    test = (
        predictions[
            predictions[
                "split"
            ]
            ==
            "test"
        ]
        .copy()
    )

    rows = []

    for (
        horizon,
        district,
    ), group in test.groupby(
        [
            "horizon_weeks",
            "district",
        ]
    ):

        actual = (
            group[
                "actual"
            ]
        )

        persistence = (
            calculate_metrics(
                actual,
                group[
                    "prediction_persistence"
                ],
            )
        )

        rolling = (
            calculate_metrics(
                actual,
                group[
                    "prediction_rolling4"
                ],
            )
        )

        rows.append(
            {
                "horizon_weeks":
                horizon,

                "district":
                district,

                "persistence_mae":
                persistence[
                    "mae"
                ],

                "rolling4_mae":
                rolling[
                    "mae"
                ],
            }
        )

    district_df = (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "horizon_weeks",
                "district",
            ]
        )
    )

    output_file = (
        RESULTS_DIR
        /
        "baseline_metrics_by_district.csv"
    )

    district_df.to_csv(
        output_file,
        index=False,
    )

    print(
        f"[PASS] Saved "
        f"district metrics:"
    )

    print(
        f"       {output_file}"
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
        "BASELINE FORECASTING"
    )

    print(
        "=========================================="
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_data()

    all_metrics = []

    all_predictions = []

    for horizon in HORIZONS:

        (
            metrics_rows,
            prediction_rows,
        ) = evaluate_horizon(
            df,
            horizon,
        )

        all_metrics.extend(
            metrics_rows
        )

        all_predictions.extend(
            prediction_rows
        )

    metrics_df = (
        pd.DataFrame(
            all_metrics
        )
    )

    predictions_df = (
        pd.concat(
            all_predictions,
            ignore_index=True,
        )
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    metrics_df.to_csv(
        METRICS_FILE,
        index=False,
    )

    predictions_df.to_csv(
        PREDICTIONS_FILE,
        index=False,
        date_format="%Y-%m-%d",
    )

    print()
    print(
        "=========================================="
    )

    print(
        "OVERALL BASELINE RESULTS"
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
        f"[PASS] Metrics saved:"
    )

    print(
        f"       {METRICS_FILE}"
    )

    print(
        f"[PASS] Predictions saved:"
    )

    print(
        f"       {PREDICTIONS_FILE}"
    )

    evaluate_test_by_district(
        predictions_df
    )

    print()
    print(
        "=========================================="
    )

    print(
        "BASELINE STAGE COMPLETE"
    )

    print(
        "=========================================="
    )


if __name__ == "__main__":
    main()