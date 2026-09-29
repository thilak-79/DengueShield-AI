"""
DengueShield AI
Operational 1-Week Random Forest Refit

Uses the exact previously selected Random Forest pipeline.
No new hyperparameter search is performed.

The pipeline is simply refitted on all available historical
rows with known 1-week targets.
"""

from pathlib import Path
from datetime import datetime, timezone
import json

import joblib
import pandas as pd

from sklearn.base import clone


REFERENCE_MODEL = Path(
    "models/random_forest/"
    "random_forest_1w.joblib"
)

FEATURE_FILE = Path(
    "data/processed/"
    "ml_features_2015_2025.csv"
)

OUTPUT_DIR = Path(
    "models/operational"
)

OUTPUT_MODEL = (
    OUTPUT_DIR
    /
    "random_forest_1w_full_history.joblib"
)

METADATA_FILE = (
    OUTPUT_DIR
    /
    "random_forest_1w_full_history_metadata.json"
)


def main():

    print()
    print(
        "=========================================="
    )
    print(
        "DENGUESHIELD AI"
    )
    print(
        "OPERATIONAL RF REFIT"
    )
    print(
        "=========================================="
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference_pipeline = (
        joblib.load(
            REFERENCE_MODEL
        )
    )

    if not hasattr(
        reference_pipeline,
        "feature_names_in_",
    ):
        raise AssertionError(
            "Saved pipeline does not expose "
            "feature_names_in_."
        )

    model_features = list(
        reference_pipeline
        .feature_names_in_
    )

    df = pd.read_csv(
        FEATURE_FILE,
        parse_dates=[
            "forecast_origin_date",
            "target_date_1w",
        ],
    )

    train = (
        df[
            df[
                "target_cases_1w"
            ]
            .notna()
        ]
        .copy()
    )

    missing = (
        set(
            model_features
        )
        -
        set(
            train.columns
        )
    )

    if missing:
        raise ValueError(
            f"Missing model features: "
            f"{sorted(missing)}"
        )

    operational_model = clone(
        reference_pipeline
    )

    operational_model.fit(
        train[
            model_features
        ],
        train[
            "target_cases_1w"
        ],
    )

    joblib.dump(
        operational_model,
        OUTPUT_MODEL,
    )

    metadata = {

        "model":
        "Random Forest",

        "forecast_horizon_weeks":
        1,

        "training_rows":
        len(
            train
        ),

        "training_feature_rows_start":
        str(
            train[
                "forecast_origin_date"
            ]
            .min()
            .date()
        ),

        "training_feature_rows_end":
        str(
            train[
                "forecast_origin_date"
            ]
            .max()
            .date()
        ),

        "training_target_start":
        str(
            train[
                "target_date_1w"
            ]
            .min()
            .date()
        ),

        "training_target_end":
        str(
            train[
                "target_date_1w"
            ]
            .max()
            .date()
        ),

        "feature_count":
        len(
            model_features
        ),

        "features":
        model_features,

        "template_model":
        str(
            REFERENCE_MODEL
        ),

        "hyperparameter_retuning":
        False,

        "generated_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),
    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    print(
        f"[PASS] Training rows: "
        f"{len(train):,}"
    )

    print(
        f"[PASS] Raw features: "
        f"{len(model_features)}"
    )

    print(
        "[PASS] No new model selection "
        "or hyperparameter tuning performed"
    )

    print(
        f"[PASS] Saved model: "
        f"{OUTPUT_MODEL}"
    )

    print(
        f"[PASS] Saved metadata: "
        f"{METADATA_FILE}"
    )


if __name__ == "__main__":
    main()