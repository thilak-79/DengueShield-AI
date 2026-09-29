"""
DengueShield AI
Feature Consistency Tests
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import shap

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


HISTORICAL_FEATURE_FILE = (
    PROJECT_ROOT
    /
    "data/processed/"
    "ml_features_2015_2025.csv"
)


CURRENT_FEATURE_FILE = (
    PROJECT_ROOT
    /
    "data/processed/operational/"
    "current_features_2026.csv"
)


CURRENT_METADATA_FILE = (
    PROJECT_ROOT
    /
    "data/processed/operational/"
    "current_data_metadata.json"
)


MODEL_METADATA_FILE = (
    PROJECT_ROOT
    /
    "models/operational/"
    "random_forest_1w_full_history_metadata.json"
)


CURRENT_SHAP_FILE = (
    PROJECT_ROOT
    /
    "models/explainability/"
    "current_district_explanations_2026.csv"
)

OPERATIONAL_MODEL_FILE = (
    PROJECT_ROOT
    /
    "models/operational/"
    "random_forest_1w_full_history.joblib"
)

CURRENT_FORECAST_FILE = (
    PROJECT_ROOT
    /
    "data/processed/operational/"
    "current_forecast_2026.csv"
)



# ============================================================
# HISTORICAL FEATURES
# ============================================================

def test_historical_feature_file_exists():

    assert (
        HISTORICAL_FEATURE_FILE.exists()
    )


def test_historical_seasonality_formula():

    df = pd.read_csv(
        HISTORICAL_FEATURE_FILE,
        parse_dates=[
            "start_date",
            "end_date",
        ],
    )

    midpoint = (
        df[
            "start_date"
        ]
        +
        (
            df[
                "end_date"
            ]
            -
            df[
                "start_date"
            ]
        )
        /
        2
    )

    day_of_year = (
        midpoint
        .dt
        .dayofyear
    )

    expected_sin = np.sin(
        2
        *
        np.pi
        *
        day_of_year
        /
        365.25
    )

    expected_cos = np.cos(
        2
        *
        np.pi
        *
        day_of_year
        /
        365.25
    )

    max_sin_error = np.max(
        np.abs(
            df[
                "week_sin"
            ]
            -
            expected_sin
        )
    )

    max_cos_error = np.max(
        np.abs(
            df[
                "week_cos"
            ]
            -
            expected_cos
        )
    )

    assert max_sin_error < 1e-10
    assert max_cos_error < 1e-10

def test_current_full_shap_additivity():
    """
    Recompute SHAP from the complete transformed model feature
    matrix and verify:

        expected value + all SHAP contributions
        == raw Random Forest prediction

    This is different from current_district_explanations_2026.csv,
    which intentionally contains only the top 10 contributions.
    """

    assert OPERATIONAL_MODEL_FILE.exists(), (
        f"Operational model missing: "
        f"{OPERATIONAL_MODEL_FILE}"
    )

    feature_df = pd.read_csv(
        CURRENT_FEATURE_FILE
    )

    with open(
        MODEL_METADATA_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        metadata = json.load(
            file
        )

    model_features = metadata[
        "features"
    ]

    pipeline = joblib.load(
        OPERATIONAL_MODEL_FILE
    )

    X = feature_df[
        model_features
    ]

    # Everything except final estimator.
    preprocessing_pipeline = (
        pipeline[:-1]
    )

    model = (
        pipeline.steps[-1][1]
    )

    transformed = (
        preprocessing_pipeline
        .transform(
            X
        )
    )

    raw_prediction = (
        model.predict(
            transformed
        )
    )

    pipeline_prediction = (
        pipeline.predict(
            X
        )
    )

    # Pipeline and final estimator should agree.
    assert np.allclose(
        raw_prediction,
        pipeline_prediction,
        atol=1e-8,
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
        shap_values.ndim == 3
        and
        shap_values.shape[-1] == 1
    ):
        shap_values = np.squeeze(
            shap_values,
            axis=-1,
        )

    base_value = float(
        np.asarray(
            explainer.expected_value
        )
        .reshape(-1)[0]
    )

    reconstructed = (
        base_value
        +
        shap_values.sum(
            axis=1
        )
    )

    errors = np.abs(
        reconstructed
        -
        raw_prediction
    )

    max_error = float(
        np.max(
            errors
        )
    )

    assert max_error < 1e-4, (
        "Full SHAP additivity failed. "
        f"Maximum error = "
        f"{max_error}"
    )

# ============================================================
# CURRENT MODEL FEATURES
# ============================================================

def test_current_features_have_25_rows():

    df = pd.read_csv(
        CURRENT_FEATURE_FILE
    )

    assert len(df) == 25

    assert (
        df[
            "district"
        ]
        .nunique()
        ==
        25
    )


def test_all_model_inputs_available():

    df = pd.read_csv(
        CURRENT_FEATURE_FILE
    )

    with open(
        MODEL_METADATA_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        metadata = json.load(
            file
        )

    features = metadata[
        "features"
    ]

    missing = [
        feature
        for feature in features
        if feature
        not in df.columns
    ]

    assert missing == []


def test_exact_raw_feature_count():

    with open(
        MODEL_METADATA_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        metadata = json.load(
            file
        )

    assert metadata[
        "feature_count"
    ] == 35

    assert len(
        metadata[
            "features"
        ]
    ) == 35


def test_current_model_inputs_have_no_missing_values():

    df = pd.read_csv(
        CURRENT_FEATURE_FILE
    )

    with open(
        MODEL_METADATA_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        metadata = json.load(
            file
        )

    features = metadata[
        "features"
    ]

    assert (
        df[
            features
        ]
        .isna()
        .sum()
        .sum()
        ==
        0
    )


# ============================================================
# CURRENT SEASONALITY
# ============================================================

def test_current_seasonality_matches_training_formula():

    df = pd.read_csv(
        CURRENT_FEATURE_FILE
    )

    with open(
        CURRENT_METADATA_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        metadata = json.load(
            file
        )

    start = pd.Timestamp(
        metadata[
            "latest_week_start"
        ]
    )

    end = pd.Timestamp(
        metadata[
            "latest_week_end"
        ]
    )

    midpoint = (
        start
        +
        (
            end
            -
            start
        )
        /
        2
    )

    day_of_year = (
        midpoint.dayofyear
    )

    expected_sin = np.sin(
        2
        *
        np.pi
        *
        day_of_year
        /
        365.25
    )

    expected_cos = np.cos(
        2
        *
        np.pi
        *
        day_of_year
        /
        365.25
    )

    assert (
        df[
            "week_sin"
        ]
        .nunique()
        ==
        1
    )

    assert (
        df[
            "week_cos"
        ]
        .nunique()
        ==
        1
    )

    actual_sin = float(
        df[
            "week_sin"
        ]
        .iloc[0]
    )

    actual_cos = float(
        df[
            "week_cos"
        ]
        .iloc[0]
    )

    assert abs(
        actual_sin
        -
        expected_sin
    ) < 1e-10

    assert abs(
        actual_cos
        -
        expected_cos
    ) < 1e-10


# ============================================================
# ROLLING FEATURE CONSISTENCY
# ============================================================

def test_cases_rolling_2_definition():

    df = pd.read_csv(
        CURRENT_FEATURE_FILE
    )

    expected = (
        df[
            "cases_lag_1"
        ]
        +
        df[
            "cases_lag_2"
        ]
    ) / 2.0

    assert np.allclose(
        df[
            "cases_rolling_2"
        ],
        expected,
        atol=1e-10,
    )


def test_cases_rolling_4_definition():

    df = pd.read_csv(
        CURRENT_FEATURE_FILE
    )

    expected = (
        df[
            [
                "cases_lag_1",
                "cases_lag_2",
                "cases_lag_3",
                "cases_lag_4",
            ]
        ]
        .mean(
            axis=1
        )
    )

    assert np.allclose(
        df[
            "cases_rolling_4"
        ],
        expected,
        atol=1e-10,
    )


def test_rainfall_rolling_4_definition():

    df = pd.read_csv(
        CURRENT_FEATURE_FILE
    )

    expected = (
        df[
            [
                "rainfall_lag_1",
                "rainfall_lag_2",
                "rainfall_lag_3",
                "rainfall_lag_4",
            ]
        ]
        .sum(
            axis=1
        )
    )

    assert np.allclose(
        df[
            "rainfall_rolling_4"
        ],
        expected,
        atol=1e-10,
    )


# ============================================================
# SHAP ADDITIVITY
# ============================================================

# ============================================================
# SAVED CURRENT SHAP EXPLANATIONS
# ============================================================

def test_current_shap_explanation_file_is_top10_per_district():
    """
    The saved district explanation artifact intentionally stores
    only the top 10 SHAP contributions for each district.

    It must NOT be used directly for an exact SHAP-additivity
    reconstruction because smaller omitted feature contributions
    are not present in this file.
    """

    df = pd.read_csv(
        CURRENT_SHAP_FILE
    )

    assert len(df) == 250

    assert (
        df[
            "district"
        ]
        .nunique()
        ==
        25
    )

    rows_per_district = (
        df.groupby(
            "district"
        )
        .size()
    )

    assert (
        rows_per_district
        ==
        10
    ).all()


def test_current_shap_top10_ranks_are_valid():
    """
    Confirm every district contains exactly contribution
    ranks 1 through 10 in the saved UI explanation artifact.
    """

    df = pd.read_csv(
        CURRENT_SHAP_FILE
    )

    assert (
        df[
            "shap_value"
        ]
        .isna()
        .sum()
        ==
        0
    )

    assert (
        df[
            "base_value"
        ]
        .isna()
        .sum()
        ==
        0
    )

    assert (
        df[
            "forecast_cases_1w"
        ]
        .isna()
        .sum()
        ==
        0
    )

    for district, group in df.groupby(
        "district"
    ):

        ranks = sorted(
            group[
                "contribution_rank"
            ]
            .astype(int)
            .tolist()
        )

        assert ranks == list(
            range(
                1,
                11,
            )
        ), (
            f"Invalid SHAP ranks "
            f"for {district}: {ranks}"
        )

def test_published_forecast_matches_model_postprocessing():
    """
    Published operational forecasts must equal the Random Forest
    prediction after the production non-negative clipping rule.
    """

    assert OPERATIONAL_MODEL_FILE.exists()

    feature_df = pd.read_csv(
        CURRENT_FEATURE_FILE
    )

    forecast_df = pd.read_csv(
        CURRENT_FORECAST_FILE
    )

    with open(
        MODEL_METADATA_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        metadata = json.load(
            file
        )

    model_features = metadata[
        "features"
    ]

    pipeline = joblib.load(
        OPERATIONAL_MODEL_FILE
    )

    raw_prediction = pipeline.predict(
        feature_df[
            model_features
        ]
    )

    expected_published = np.clip(
        raw_prediction,
        0,
        None,
    )

    expected_df = pd.DataFrame(
        {
            "district":
            feature_df[
                "district"
            ]
            .astype(str),

            "expected_forecast":
            expected_published,
        }
    )

    merged = forecast_df[
        [
            "district",
            "forecast_cases_1w",
        ]
    ].merge(
        expected_df,
        on="district",
        how="inner",
        validate="one_to_one",
    )

    assert len(
        merged
    ) == 25

    assert np.allclose(
        merged[
            "forecast_cases_1w"
        ],
        merged[
            "expected_forecast"
        ],
        atol=1e-8,
    )