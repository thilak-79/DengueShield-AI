from pathlib import Path

import pandas as pd


RF_FILE = Path(
    "models/random_forest/random_forest_predictions.csv"
)

XGB_FILE = Path(
    "models/xgboost/xgboost_predictions.csv"
)

OUTPUT_DIR = Path(
    "models/comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def evaluate_rf():

    df = pd.read_csv(
        RF_FILE
    )

    rows = []

    for horizon in [1, 2, 4]:

        x = df[
            (df["horizon_weeks"] == horizon)
            &
            (df["split"] == "test")
        ].copy()

        x["model_error"] = (
            x["actual"]
            -
            x["prediction_random_forest"]
        ).abs()

        x["persistence_error"] = (
            x["actual"]
            -
            x["prediction_persistence"]
        ).abs()

        result = (
            x.groupby(
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
            )
            .reset_index()
        )

        result[
            "improvement"
        ] = (
            result[
                "persistence_mae"
            ]
            -
            result[
                "model_mae"
            ]
        )

        result[
            "model"
        ] = "random_forest"

        result[
            "horizon_weeks"
        ] = horizon

        rows.append(
            result
        )

    return pd.concat(
        rows,
        ignore_index=True,
    )


def evaluate_xgb():

    df = pd.read_csv(
        XGB_FILE
    )

    rows = []

    for horizon in [1, 2, 4]:

        x = df[
            (df["horizon_weeks"] == horizon)
            &
            (df["split"] == "test")
        ].copy()

        x["model_error"] = (
            x["actual"]
            -
            x["prediction_xgboost"]
        ).abs()

        x["persistence_error"] = (
            x["actual"]
            -
            x["prediction_persistence"]
        ).abs()

        result = (
            x.groupby(
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
            )
            .reset_index()
        )

        result[
            "improvement"
        ] = (
            result[
                "persistence_mae"
            ]
            -
            result[
                "model_mae"
            ]
        )

        result[
            "model"
        ] = "xgboost"

        result[
            "horizon_weeks"
        ] = horizon

        rows.append(
            result
        )

    return pd.concat(
        rows,
        ignore_index=True,
    )


def main():

    rf = evaluate_rf()
    xgb = evaluate_xgb()

    result = pd.concat(
        [
            rf,
            xgb,
        ],
        ignore_index=True,
    )

    output_file = (
        OUTPUT_DIR
        /
        "district_level_test_metrics.csv"
    )

    result.to_csv(
        output_file,
        index=False,
    )

    print()
    print(
        "=========================================="
    )
    print(
        "DISTRICT-LEVEL TEST SUMMARY"
    )
    print(
        "=========================================="
    )

    summary = (
        result.groupby(
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
            mean_improvement=(
                "improvement",
                "mean",
            ),
        )
        .reset_index()
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        f"[PASS] Saved:"
    )
    print(
        f"       {output_file}"
    )


if __name__ == "__main__":
    main()