"""
DengueShield AI
Final Model Comparison

Compares:
    Persistence
    Random Forest
    XGBoost
    LightGBM

Primary metric:
    MAE

Secondary:
    RMSE
    sMAPE

Also evaluates how many districts each ML model
beats persistence in.
"""

from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_DIR = Path(
    "models/comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


BASELINE_METRICS = Path(
    "models/baseline/"
    "baseline_metrics.csv"
)

RF_METRICS = Path(
    "models/random_forest/"
    "random_forest_metrics.csv"
)

XGB_METRICS = Path(
    "models/xgboost/"
    "xgboost_metrics.csv"
)

LGBM_METRICS = Path(
    "models/lightgbm/"
    "lightgbm_metrics.csv"
)


RF_PREDICTIONS = Path(
    "models/random_forest/"
    "random_forest_predictions.csv"
)

XGB_PREDICTIONS = Path(
    "models/xgboost/"
    "xgboost_predictions.csv"
)

LGBM_PREDICTIONS = Path(
    "models/lightgbm/"
    "lightgbm_predictions.csv"
)


FINAL_COMPARISON_FILE = (
    OUTPUT_DIR
    /
    "final_model_comparison.csv"
)

DISTRICT_SUMMARY_FILE = (
    OUTPUT_DIR
    /
    "district_consistency_summary.csv"
)

DISTRICT_DETAIL_FILE = (
    OUTPUT_DIR
    /
    "district_model_comparison.csv"
)


# ============================================================
# LOAD METRICS
# ============================================================

def load_test_metrics():

    frames = []

    # Persistence
    baseline = pd.read_csv(
        BASELINE_METRICS
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
        ][
            [
                "horizon_weeks",
                "model",
                "mae",
                "rmse",
                "smape",
            ]
        ]
        .copy()
    )

    frames.append(
        persistence
    )

    # Random Forest
    rf = pd.read_csv(
        RF_METRICS
    )

    rf = (
        rf[
            rf[
                "split"
            ]
            ==
            "test"
        ][
            [
                "horizon_weeks",
                "model",
                "mae",
                "rmse",
                "smape",
            ]
        ]
        .copy()
    )

    frames.append(
        rf
    )

    # XGBoost
    xgb = pd.read_csv(
        XGB_METRICS
    )

    xgb = (
        xgb[
            xgb[
                "split"
            ]
            ==
            "test"
        ][
            [
                "horizon_weeks",
                "model",
                "mae",
                "rmse",
                "smape",
            ]
        ]
        .copy()
    )

    frames.append(
        xgb
    )

    # LightGBM
    lgbm = pd.read_csv(
        LGBM_METRICS
    )

    lgbm = (
        lgbm[
            lgbm[
                "split"
            ]
            ==
            "test"
        ][
            [
                "horizon_weeks",
                "model",
                "mae",
                "rmse",
                "smape",
            ]
        ]
        .copy()
    )

    frames.append(
        lgbm
    )

    result = pd.concat(
        frames,
        ignore_index=True,
    )

    # --------------------------------------------
    # Add persistence MAE for comparison
    # --------------------------------------------

    persistence_lookup = (
        persistence.set_index(
            "horizon_weeks"
        )[
            "mae"
        ]
        .to_dict()
    )

    result[
        "persistence_mae"
    ] = (
        result[
            "horizon_weeks"
        ]
        .map(
            persistence_lookup
        )
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
            (
                result[
                    "persistence_mae"
                ]
                -
                result[
                    "mae"
                ]
            )
            /
            result[
                "persistence_mae"
            ]
            *
            100.0
        ),
    )

    # Descriptive ranking only.
    result[
        "mae_rank"
    ] = (
        result.groupby(
            "horizon_weeks"
        )[
            "mae"
        ]
        .rank(
            method="min",
            ascending=True,
        )
        .astype(int)
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
# DISTRICT EVALUATION
# ============================================================

def evaluate_prediction_file(
    path,
    model_name,
    prediction_column,
):

    df = pd.read_csv(
        path
    )

    rows = []

    for horizon in [
        1,
        2,
        4,
    ]:

        temp = (
            df[
                (
                    df[
                        "horizon_weeks"
                    ]
                    ==
                    horizon
                )
                &
                (
                    df[
                        "split"
                    ]
                    ==
                    "test"
                )
            ]
            .copy()
        )

        temp[
            "model_error"
        ] = (
            temp[
                "actual"
            ]
            -
            temp[
                prediction_column
            ]
        ).abs()

        temp[
            "persistence_error"
        ] = (
            temp[
                "actual"
            ]
            -
            temp[
                "prediction_persistence"
            ]
        ).abs()

        district = (
            temp.groupby(
                "district"
            )
            .agg(
                model_mae=(
                    "model_error",
                    "mean",
                ),

                persistence_mae=(
                    "persistence_error",
                    "mean",
                ),

                observations=(
                    "actual",
                    "size",
                ),
            )
            .reset_index()
        )

        district[
            "improvement"
        ] = (
            district[
                "persistence_mae"
            ]
            -
            district[
                "model_mae"
            ]
        )

        district[
            "model"
        ] = model_name

        district[
            "horizon_weeks"
        ] = horizon

        rows.append(
            district
        )

    return pd.concat(
        rows,
        ignore_index=True,
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
        "FINAL MODEL COMPARISON"
    )
    print(
        "=========================================="
    )

    # --------------------------------------------------------
    # OVERALL METRICS
    # --------------------------------------------------------

    comparison = (
        load_test_metrics()
    )

    comparison.to_csv(
        FINAL_COMPARISON_FILE,
        index=False,
    )

    print()
    print(
        "OVERALL HELD-OUT METRICS"
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # DISTRICT RESULTS
    # --------------------------------------------------------

    rf = evaluate_prediction_file(
        RF_PREDICTIONS,
        "random_forest",
        "prediction_random_forest",
    )

    xgb = evaluate_prediction_file(
        XGB_PREDICTIONS,
        "xgboost",
        "prediction_xgboost",
    )

    lgbm = evaluate_prediction_file(
        LGBM_PREDICTIONS,
        "lightgbm",
        "prediction_lightgbm",
    )

    district_detail = pd.concat(
        [
            rf,
            xgb,
            lgbm,
        ],
        ignore_index=True,
    )

    district_detail.to_csv(
        DISTRICT_DETAIL_FILE,
        index=False,
    )

    district_summary = (
        district_detail.groupby(
            [
                "model",
                "horizon_weeks",
            ]
        )
        .agg(
            districts_better=(
                "improvement",
                lambda x: int(
                    (x > 0).sum()
                ),
            ),

            districts_worse=(
                "improvement",
                lambda x: int(
                    (x < 0).sum()
                ),
            ),

            districts_equal=(
                "improvement",
                lambda x: int(
                    np.isclose(
                        x,
                        0.0,
                    ).sum()
                ),
            ),

            mean_district_mae_improvement=(
                "improvement",
                "mean",
            ),
        )
        .reset_index()
    )

    district_summary.to_csv(
        DISTRICT_SUMMARY_FILE,
        index=False,
    )

    print()
    print(
        "=========================================="
    )

    print(
        "DISTRICT CONSISTENCY"
    )

    print(
        "=========================================="
    )

    print(
        district_summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # LOWEST MAE FOR EACH HORIZON
    # --------------------------------------------------------

    print()
    print(
        "=========================================="
    )

    print(
        "LOWEST REPORTED MAE BY HORIZON"
    )

    print(
        "=========================================="
    )

    for horizon in [
        1,
        2,
        4,
    ]:

        subset = (
            comparison[
                comparison[
                    "horizon_weeks"
                ]
                ==
                horizon
            ]
            .sort_values(
                "mae"
            )
        )

        row = (
            subset.iloc[
                0
            ]
        )

        print(
            f"{horizon}-week: "
            f"{row['model']} "
            f"| MAE="
            f"{row['mae']:.4f}"
        )

    print()
    print(
        "[PASS] Saved:"
    )

    print(
        f"       "
        f"{FINAL_COMPARISON_FILE}"
    )

    print(
        f"       "
        f"{DISTRICT_DETAIL_FILE}"
    )

    print(
        f"       "
        f"{DISTRICT_SUMMARY_FILE}"
    )


if __name__ == "__main__":
    main()