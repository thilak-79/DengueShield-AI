"""
DengueShield AI
Experimental Current 2026 Pipeline Tests
"""

from pathlib import Path
import json

import pandas as pd


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


OPERATIONAL_DIR = (
    PROJECT_ROOT
    /
    "data/processed/operational"
)


ACTIVITY_FILE = (
    OPERATIONAL_DIR
    /
    "current_activity_2026.csv"
)


FEATURE_FILE = (
    OPERATIONAL_DIR
    /
    "current_features_2026.csv"
)


FORECAST_FILE = (
    OPERATIONAL_DIR
    /
    "current_forecast_2026.csv"
)


WEATHER_FILE = (
    OPERATIONAL_DIR
    /
    "current_weather_weekly.csv"
)


AUDIT_FILE = (
    OPERATIONAL_DIR
    /
    "current_data_audit.csv"
)


STATUS_FILE = (
    OPERATIONAL_DIR
    /
    "current_status_2026.json"
)


CURRENT_SHAP_FILE = (
    PROJECT_ROOT
    /
    "models/explainability/"
    "current_district_explanations_2026.csv"
)


MODEL_METADATA_FILE = (
    PROJECT_ROOT
    /
    "models/operational/"
    "random_forest_1w_full_history_metadata.json"
)


# ============================================================
# FILE EXISTENCE
# ============================================================

def test_required_operational_files_exist():

    files = [
        ACTIVITY_FILE,
        FEATURE_FILE,
        FORECAST_FILE,
        WEATHER_FILE,
        AUDIT_FILE,
        STATUS_FILE,
        CURRENT_SHAP_FILE,
        MODEL_METADATA_FILE,
    ]

    for path in files:

        assert (
            path.exists()
        ), f"Missing file: {path}"


# ============================================================
# CURRENT ACTIVITY
# ============================================================

def test_activity_has_25_districts():

    df = pd.read_csv(
        ACTIVITY_FILE
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


def test_activity_has_no_duplicate_districts():

    df = pd.read_csv(
        ACTIVITY_FILE
    )

    assert (
        df[
            "district"
        ]
        .duplicated()
        .sum()
        ==
        0
    )


def test_current_national_total_is_1156():

    df = pd.read_csv(
        ACTIVITY_FILE
    )

    assert (
        df[
            "current_cases"
        ]
        .sum()
        ==
        1156
    )


def test_current_forecasts_complete():

    df = pd.read_csv(
        ACTIVITY_FILE
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


def test_current_forecasts_nonnegative():

    df = pd.read_csv(
        ACTIVITY_FILE
    )

    assert (
        df[
            "forecast_cases_1w"
        ]
        >=
        0
    ).all()


def test_current_mode_label():

    df = pd.read_csv(
        ACTIVITY_FILE
    )

    assert (
        df[
            "operational_status"
        ]
        ==
        "EXPERIMENTAL_CURRENT_INFERENCE"
    ).all()


def test_source_semantics_warning_enabled():

    df = pd.read_csv(
        ACTIVITY_FILE
    )

    values = (
        df[
            "source_semantics_warning"
        ]
        .astype(str)
        .str.lower()
    )

    assert (
        values
        ==
        "true"
    ).all()


# ============================================================
# ACTIVITY COUNTS
# ============================================================

def test_activity_category_distribution():

    df = pd.read_csv(
        ACTIVITY_FILE
    )

    counts = (
        df[
            "relative_activity"
        ]
        .value_counts()
        .to_dict()
    )

    assert counts.get(
        "VERY HIGH",
        0,
    ) == 1

    assert counts.get(
        "HIGH",
        0,
    ) == 5

    assert counts.get(
        "ELEVATED",
        0,
    ) == 13

    assert counts.get(
        "LOW",
        0,
    ) == 6

    assert sum(
        counts.values()
    ) == 25


# ============================================================
# CURRENT WEATHER
# ============================================================

def test_current_weather_complete():

    df = pd.read_csv(
        WEATHER_FILE
    )

    assert len(df) == 125

    assert (
        df[
            "district"
        ]
        .nunique()
        ==
        25
    )


def test_current_weather_5_weeks():

    df = pd.read_csv(
        WEATHER_FILE
    )

    weeks = sorted(
        df[
            "week"
        ]
        .unique()
        .tolist()
    )

    assert weeks == [
        33,
        34,
        35,
        36,
        37,
    ]


def test_weather_has_25_districts_per_week():

    df = pd.read_csv(
        WEATHER_FILE
    )

    counts = (
        df.groupby(
            "week"
        )[
            "district"
        ]
        .nunique()
    )

    assert (
        counts
        ==
        25
    ).all()


# ============================================================
# MODEL METADATA
# ============================================================

def test_operational_model_has_35_raw_features():

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


def test_no_hyperparameter_retuning():

    with open(
        MODEL_METADATA_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        metadata = json.load(
            file
        )

    assert (
        metadata[
            "hyperparameter_retuning"
        ]
        is False
    )


# ============================================================
# CURRENT STATUS
# ============================================================

def test_current_status_metadata():

    with open(
        STATUS_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        status = json.load(
            file
        )

    assert status[
        "mode"
    ] == (
        "EXPERIMENTAL_CURRENT_INFERENCE"
    )

    assert status[
        "latest_year"
    ] == 2026

    assert status[
        "latest_week"
    ] == 37

    assert status[
        "surveillance_window_start"
    ] == "2026-09-07"

    assert status[
        "surveillance_window_end"
    ] == "2026-09-13"

    assert status[
        "forecast_target_start"
    ] == "2026-09-14"

    assert status[
        "forecast_target_end"
    ] == "2026-09-20"

    assert status[
        "district_count"
    ] == 25


def test_current_status_category_totals():

    with open(
        STATUS_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        status = json.load(
            file
        )

    assert sum(
        status[
            "activity_counts"
        ]
        .values()
    ) == 25

    assert sum(
        status[
            "trend_counts"
        ]
        .values()
    ) == 25


# ============================================================
# SPECIFIC DISTRICTS
# ============================================================

def test_colombo_current_values():

    df = pd.read_csv(
        ACTIVITY_FILE
    )

    row = df[
        df[
            "district"
        ]
        ==
        "Colombo"
    ].iloc[0]

    assert row[
        "current_cases"
    ] == 207

    assert abs(
        row[
            "forecast_cases_1w"
        ]
        -
        176.717084
    ) < 1e-3

    assert row[
        "relative_activity"
    ] == "ELEVATED"

    assert row[
        "forecast_trend"
    ] == "DECREASING"


def test_kandy_current_values():

    df = pd.read_csv(
        ACTIVITY_FILE
    )

    row = df[
        df[
            "district"
        ]
        ==
        "Kandy"
    ].iloc[0]

    assert row[
        "current_cases"
    ] == 174

    assert abs(
        row[
            "forecast_cases_1w"
        ]
        -
        137.282937
    ) < 1e-3

    assert row[
        "relative_activity"
    ] == "HIGH"

    assert row[
        "forecast_trend"
    ] == "DECREASING"