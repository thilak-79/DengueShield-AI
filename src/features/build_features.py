"""
DengueShield AI
Leakage-Safe Feature Engineering Pipeline

Input:
    data/processed/dengue_weather_weekly_2015_2025.csv

Output:
    data/processed/ml_features_2015_2025.csv

Forecast setup
--------------
At the end of surveillance period t, use only information that
would already be known at t to forecast:

    t + 1 week
    t + 2 weeks
    t + 4 weeks

Important
---------
This script uses ACTUAL start_date differences to validate
lags and targets.

Therefore:

    lag_1 must really be 7 days earlier
    lag_2 must really be 14 days earlier
    target_4w must really be 28 days later

This prevents irregular or missing WER weeks from silently
creating incorrect forecast horizons.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


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
    "dengue_weather_weekly_2015_2025.csv"
)

OUTPUT_FILE = Path(
    "data/processed/"
    "ml_features_2015_2025.csv"
)


# ============================================================
# SETTINGS
# ============================================================

LAG_WEEKS = [
    1,
    2,
    3,
    4,
]

FORECAST_HORIZONS = [
    1,
    2,
    4,
]

DAYS_PER_WEEK = 7


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """
    Load dengue + weather weekly dataset.
    """

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found: "
            f"{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=[
            "start_date",
            "end_date",
        ],
    )

    required_columns = {
        "year",
        "week",
        "start_date",
        "end_date",
        "district",
        "cases",
        "temperature_mean",
        "temperature_min",
        "temperature_max",
        "humidity_mean",
        "rainfall_sum",
        "rain_days",
    }

    missing = (
        required_columns
        -
        set(
            df.columns
        )
    )

    if missing:

        raise ValueError(
            f"Missing required columns: "
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
        f"[PASS] Years: "
        f"{df['year'].min()} "
        f"- "
        f"{df['year'].max()}"
    )

    return df


# ============================================================
# BASIC VALIDATION
# ============================================================

def validate_input(
    df,
):
    """
    Validate source dataset before feature engineering.
    """

    print()
    print(
        "=========================================="
    )

    print(
        "VALIDATING INPUT DATASET"
    )

    print(
        "=========================================="
    )

    # --------------------------------------------------------
    # Duplicate district surveillance periods
    # --------------------------------------------------------

    duplicates = (
        df.duplicated(
            subset=[
                "district",
                "start_date",
            ]
        )
    )

    if duplicates.any():

        raise AssertionError(
            f"Duplicate district/date "
            f"rows found: "
            f"{duplicates.sum()}"
        )

    print(
        "[PASS] No duplicate "
        "district/date rows"
    )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    required_numeric = [
        "cases",
        "temperature_mean",
        "temperature_min",
        "temperature_max",
        "humidity_mean",
        "rainfall_sum",
        "rain_days",
    ]

    missing_values = (
        df[
            required_numeric
        ]
        .isna()
        .sum()
        .sum()
    )

    if missing_values:

        raise AssertionError(
            f"Input contains "
            f"{missing_values} "
            f"missing required values."
        )

    print(
        "[PASS] No missing "
        "source values"
    )

    # --------------------------------------------------------
    # Basic weather sanity
    # --------------------------------------------------------

    if (
        (
            df["humidity_mean"]
            < 0
        )
        |
        (
            df["humidity_mean"]
            > 100
        )
    ).any():

        raise AssertionError(
            "Invalid humidity values."
        )

    if (
        df[
            "rainfall_sum"
        ]
        < 0
    ).any():

        raise AssertionError(
            "Negative rainfall values."
        )

    if (
        df[
            "temperature_min"
        ]
        >
        df[
            "temperature_max"
        ]
    ).any():

        raise AssertionError(
            "Invalid temperature range."
        )

    print(
        "[PASS] Weather values "
        "pass sanity checks"
    )


# ============================================================
# SORT CHRONOLOGICALLY
# ============================================================

def prepare_chronology(
    df,
):
    """
    Sort each district by actual surveillance start date.
    """

    df = (
        df.copy()
        .sort_values(
            [
                "district",
                "start_date",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # Explicit forecast origin:
    # information is assumed available after this date.
    df[
        "forecast_origin_date"
    ] = df[
        "end_date"
    ]

    return df


# ============================================================
# STRICT DATE-AWARE LAG
# ============================================================

def create_strict_lag(
    df,
    source_column,
    lag_weeks,
):
    """
    Create a lag only when the previous observation
    is exactly the requested number of weeks earlier.

    Example:
        lag_weeks = 2

    accepted only when:

        current start_date
        -
        historical start_date
        =
        14 days
    """

    group = df.groupby(
        "district",
        sort=False,
    )

    lagged_value = (
        group[
            source_column
        ]
        .shift(
            lag_weeks
        )
    )

    lagged_date = (
        group[
            "start_date"
        ]
        .shift(
            lag_weeks
        )
    )

    expected_days = (
        lag_weeks
        *
        DAYS_PER_WEEK
    )

    actual_days = (
        df[
            "start_date"
        ]
        -
        lagged_date
    ).dt.days

    valid = (
        actual_days
        ==
        expected_days
    )

    return (
        lagged_value.where(
            valid
        )
    )


# ============================================================
# STRICT FUTURE TARGET
# ============================================================

def create_strict_target(
    df,
    horizon_weeks,
):
    """
    Create future dengue target only if the future
    surveillance period is exactly the requested
    number of weeks later.

    Returns:
        target cases
        target start date
    """

    group = df.groupby(
        "district",
        sort=False,
    )

    future_cases = (
        group[
            "cases"
        ]
        .shift(
            -horizon_weeks
        )
    )

    future_start_date = (
        group[
            "start_date"
        ]
        .shift(
            -horizon_weeks
        )
    )

    expected_days = (
        horizon_weeks
        *
        DAYS_PER_WEEK
    )

    actual_days = (
        future_start_date
        -
        df[
            "start_date"
        ]
    ).dt.days

    valid = (
        actual_days
        ==
        expected_days
    )

    target = (
        future_cases.where(
            valid
        )
    )

    target_date = (
        future_start_date.where(
            valid
        )
    )

    return (
        target,
        target_date,
    )


# ============================================================
# CREATE LAG FEATURES
# ============================================================

def add_lag_features(
    df,
):
    """
    Add past dengue and weather information.
    """

    print()
    print(
        "[BUILD] Creating strict "
        "date-aware lag features..."
    )

    # --------------------------------------------------------
    # DENGUE CASE LAGS
    # --------------------------------------------------------

    for lag in LAG_WEEKS:

        df[
            f"cases_lag_{lag}"
        ] = create_strict_lag(
            df,
            "cases",
            lag,
        )

    # --------------------------------------------------------
    # RAINFALL LAGS
    # --------------------------------------------------------

    for lag in LAG_WEEKS:

        df[
            f"rainfall_lag_{lag}"
        ] = create_strict_lag(
            df,
            "rainfall_sum",
            lag,
        )

    # --------------------------------------------------------
    # HUMIDITY LAGS
    # --------------------------------------------------------

    for lag in LAG_WEEKS:

        df[
            f"humidity_lag_{lag}"
        ] = create_strict_lag(
            df,
            "humidity_mean",
            lag,
        )

    # --------------------------------------------------------
    # TEMPERATURE LAGS
    # --------------------------------------------------------

    for lag in LAG_WEEKS:

        df[
            f"temperature_lag_{lag}"
        ] = create_strict_lag(
            df,
            "temperature_mean",
            lag,
        )

    # --------------------------------------------------------
    # RAIN-DAY LAGS
    # --------------------------------------------------------

    for lag in LAG_WEEKS:

        df[
            f"rain_days_lag_{lag}"
        ] = create_strict_lag(
            df,
            "rain_days",
            lag,
        )

    return df


# ============================================================
# ROLLING FEATURES
# ============================================================

def add_rolling_features(
    df,
):
    """
    Rolling statistics use ONLY lagged/past values.

    This is intentionally stricter than:

        rolling().mean()

    on the current cases column, because we want the
    feature names to clearly represent historical data.
    """

    print(
        "[BUILD] Creating "
        "historical rolling features..."
    )

    # --------------------------------------------------------
    # CASES
    # --------------------------------------------------------

    df[
        "cases_rolling_2"
    ] = (
        df[
            [
                "cases_lag_1",
                "cases_lag_2",
            ]
        ]
        .mean(
            axis=1,
            skipna=False,
        )
    )

    df[
        "cases_rolling_4"
    ] = (
        df[
            [
                "cases_lag_1",
                "cases_lag_2",
                "cases_lag_3",
                "cases_lag_4",
            ]
        ]
        .mean(
            axis=1,
            skipna=False,
        )
    )

    # --------------------------------------------------------
    # RAINFALL
    # --------------------------------------------------------

    df[
        "rainfall_rolling_4"
    ] = (
        df[
            [
                "rainfall_lag_1",
                "rainfall_lag_2",
                "rainfall_lag_3",
                "rainfall_lag_4",
            ]
        ]
        .sum(
            axis=1,
            min_count=4,
        )
    )

    # --------------------------------------------------------
    # HUMIDITY
    # --------------------------------------------------------

    df[
        "humidity_rolling_4"
    ] = (
        df[
            [
                "humidity_lag_1",
                "humidity_lag_2",
                "humidity_lag_3",
                "humidity_lag_4",
            ]
        ]
        .mean(
            axis=1,
            skipna=False,
        )
    )

    # --------------------------------------------------------
    # TEMPERATURE
    # --------------------------------------------------------

    df[
        "temperature_rolling_4"
    ] = (
        df[
            [
                "temperature_lag_1",
                "temperature_lag_2",
                "temperature_lag_3",
                "temperature_lag_4",
            ]
        ]
        .mean(
            axis=1,
            skipna=False,
        )
    )

    return df


# ============================================================
# SEASONAL FEATURES
# ============================================================

def add_seasonality(
    df,
):
    """
    Create smooth annual cyclic seasonality.

    We use actual surveillance dates instead of blindly
    assuming the WER week number equals ISO week number.

    The feature names week_sin/week_cos are retained for
    readability in the ML dataset.
    """

    print(
        "[BUILD] Creating "
        "seasonal features..."
    )

    # Middle of surveillance interval.
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
        midpoint.dt.dayofyear
    )

    df[
        "week_sin"
    ] = np.sin(
        2
        *
        np.pi
        *
        day_of_year
        /
        365.25
    )

    df[
        "week_cos"
    ] = np.cos(
        2
        *
        np.pi
        *
        day_of_year
        /
        365.25
    )

    return df


# ============================================================
# FORECAST TARGETS
# ============================================================

def add_targets(
    df,
):
    """
    Add future dengue case targets.

    target_cases_1w -> exactly 7 days later
    target_cases_2w -> exactly 14 days later
    target_cases_4w -> exactly 28 days later
    """

    print(
        "[BUILD] Creating strict "
        "future forecast targets..."
    )

    for horizon in (
        FORECAST_HORIZONS
    ):

        (
            target,
            target_date,
        ) = create_strict_target(
            df,
            horizon,
        )

        df[
            f"target_cases_{horizon}w"
        ] = target

        df[
            f"target_date_{horizon}w"
        ] = target_date

    return df


# ============================================================
# CURRENT-WEEK FEATURES
# ============================================================

def rename_current_features(
    df,
):
    """
    Make it explicit that these measurements refer
    to the completed forecast-origin week.

    cases_current is valid because we forecast future
    dengue activity after observing the current
    completed surveillance period.
    """

    df = df.rename(
        columns={
            "cases":
            "cases_current",

            "rainfall_sum":
            "rainfall_current",

            "humidity_mean":
            "humidity_current",

            "temperature_mean":
            "temperature_current",

            "temperature_min":
            "temperature_min_current",

            "temperature_max":
            "temperature_max_current",

            "rain_days":
            "rain_days_current",
        }
    )

    return df


# ============================================================
# REMOVE ROWS WITHOUT COMPLETE HISTORY
# ============================================================

def remove_incomplete_feature_rows(
    df,
):
    """
    For the first ML version we require a complete
    four-week historical context.

    This naturally removes:
        - first four observations of each district
        - periods immediately after historical gaps

    Future TARGET values are NOT required here because
    the final weeks of 2025 naturally have no future
    labels yet.
    """

    required_features = [

        # Current state
        "cases_current",
        "temperature_current",
        "temperature_min_current",
        "temperature_max_current",
        "humidity_current",
        "rainfall_current",
        "rain_days_current",

        # Cases history
        "cases_lag_1",
        "cases_lag_2",
        "cases_lag_3",
        "cases_lag_4",

        # Rain history
        "rainfall_lag_1",
        "rainfall_lag_2",
        "rainfall_lag_3",
        "rainfall_lag_4",

        # Humidity history
        "humidity_lag_1",
        "humidity_lag_2",
        "humidity_lag_3",
        "humidity_lag_4",

        # Temperature history
        "temperature_lag_1",
        "temperature_lag_2",
        "temperature_lag_3",
        "temperature_lag_4",

        # Rain-day history
        "rain_days_lag_1",
        "rain_days_lag_2",
        "rain_days_lag_3",
        "rain_days_lag_4",

        # Rolling
        "cases_rolling_2",
        "cases_rolling_4",
        "rainfall_rolling_4",
        "humidity_rolling_4",
        "temperature_rolling_4",

        # Season
        "week_sin",
        "week_cos",
    ]

    before = len(
        df
    )

    cleaned = (
        df.dropna(
            subset=required_features
        )
        .copy()
    )

    removed = (
        before
        -
        len(
            cleaned
        )
    )

    print()
    print(
        f"[INFO] Rows before "
        f"history filtering: "
        f"{before:,}"
    )

    print(
        f"[INFO] Rows removed because "
        f"complete 4-week history was "
        f"not available: "
        f"{removed:,}"
    )

    print(
        f"[PASS] Feature-complete rows: "
        f"{len(cleaned):,}"
    )

    return cleaned


# ============================================================
# OUTPUT COLUMN ORDER
# ============================================================

def arrange_columns(
    df,
):
    """
    Keep a clear, human-readable column order.
    """

    columns = [

        # Identification
        "year",
        "week",
        "start_date",
        "end_date",
        "forecast_origin_date",
        "district",

        # Current state
        "cases_current",

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

        # Rain days
        "rain_days_lag_1",
        "rain_days_lag_2",
        "rain_days_lag_3",
        "rain_days_lag_4",

        # Seasonality
        "week_sin",
        "week_cos",

        # Targets
        "target_cases_1w",
        "target_date_1w",

        "target_cases_2w",
        "target_date_2w",

        "target_cases_4w",
        "target_date_4w",
    ]

    return (
        df[
            columns
        ]
        .copy()
    )


# ============================================================
# LEAKAGE VALIDATION
# ============================================================

def validate_feature_dates(
    df,
):
    """
    Explicitly prove forecast targets occur after
    forecast origin.

    This is a key leakage-safety check.
    """

    print()
    print(
        "=========================================="
    )

    print(
        "LEAKAGE / DATE VALIDATION"
    )

    print(
        "=========================================="
    )

    for horizon in (
        FORECAST_HORIZONS
    ):

        target_column = (
            f"target_cases_{horizon}w"
        )

        date_column = (
            f"target_date_{horizon}w"
        )

        valid = (
            df[
                target_column
            ]
            .notna()
        )

        subset = (
            df[
                valid
            ]
        )

        if subset.empty:

            raise AssertionError(
                f"No valid "
                f"{horizon}-week targets."
            )

        actual_difference = (
            subset[
                date_column
            ]
            -
            subset[
                "start_date"
            ]
        ).dt.days

        expected_difference = (
            horizon
            *
            DAYS_PER_WEEK
        )

        if not (
            actual_difference
            ==
            expected_difference
        ).all():

            raise AssertionError(
                f"Invalid "
                f"{horizon}-week "
                f"target dates detected."
            )

        # The future period must begin after
        # forecast information becomes available.
        if not (
            subset[
                date_column
            ]
            >
            subset[
                "forecast_origin_date"
            ]
        ).all():

            raise AssertionError(
                f"Potential leakage detected "
                f"for {horizon}-week target."
            )

        print(
            f"[PASS] "
            f"{horizon}-week target uses "
            f"exactly "
            f"{expected_difference} "
            f"calendar days"
        )


# ============================================================
# FEATURE VALIDATION
# ============================================================

def validate_output(
    df,
):
    """
    Final structural checks.
    """

    print()
    print(
        "=========================================="
    )

    print(
        "VALIDATING FEATURE DATASET"
    )

    print(
        "=========================================="
    )

    duplicates = (
        df.duplicated(
            subset=[
                "district",
                "start_date",
            ]
        )
    )

    if duplicates.any():

        raise AssertionError(
            "Duplicate feature rows."
        )

    print(
        "[PASS] No duplicate "
        "feature rows"
    )

    # --------------------------------------------------------
    # Target summary
    # --------------------------------------------------------

    for horizon in (
        FORECAST_HORIZONS
    ):

        column = (
            f"target_cases_{horizon}w"
        )

        count = (
            df[
                column
            ]
            .notna()
            .sum()
        )

        print(
            f"[INFO] Valid "
            f"{horizon}-week targets: "
            f"{count:,}"
        )

    # --------------------------------------------------------
    # District coverage
    # --------------------------------------------------------

    districts = (
        df[
            "district"
        ]
        .nunique()
    )

    if districts != 25:

        raise AssertionError(
            f"Expected 25 districts, "
            f"found {districts}"
        )

    print(
        "[PASS] 25 districts present"
    )

    # --------------------------------------------------------
    # Year coverage
    # --------------------------------------------------------

    print(
        f"[PASS] Years: "
        f"{df['year'].min()} "
        f"- "
        f"{df['year'].max()}"
    )


# ============================================================
# MODEL SPLIT INFORMATION
# ============================================================

def print_model_split_guidance(
    df,
):
    """
    IMPORTANT:
    Splits must later be based on TARGET DATE,
    not merely forecast-origin year.

    Otherwise a December training row could have
    a January validation target.
    """

    print()
    print(
        "=========================================="
    )

    print(
        "FUTURE MODEL SPLIT PLAN"
    )

    print(
        "=========================================="
    )

    for horizon in (
        FORECAST_HORIZONS
    ):

        target = (
            f"target_cases_{horizon}w"
        )

        target_date = (
            f"target_date_{horizon}w"
        )

        temp = (
            df[
                df[
                    target
                ]
                .notna()
            ]
            .copy()
        )

        target_year = (
            temp[
                target_date
            ]
            .dt.year
        )

        train_count = (
            target_year
            <= 2023
        ).sum()

        validation_count = (
            target_year
            == 2024
        ).sum()

        test_count = (
            target_year
            == 2025
        ).sum()

        print()
        print(
            f"{horizon}-week forecast:"
        )

        print(
            f"  Train target dates "
            f"<= 2023: "
            f"{train_count:,}"
        )

        print(
            f"  Validation target "
            f"dates in 2024: "
            f"{validation_count:,}"
        )

        print(
            f"  Final test target "
            f"dates in 2025: "
            f"{test_count:,}"
        )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    df,
):
    """
    Dataset summary.
    """

    print()
    print(
        "=========================================="
    )

    print(
        "ML FEATURE DATASET SUMMARY"
    )

    print(
        "=========================================="
    )

    summary = (
        df.groupby(
            "year"
        )
        .agg(
            weeks=(
                "week",
                "nunique",
            ),
            districts=(
                "district",
                "nunique",
            ),
            rows=(
                "district",
                "size",
            ),
        )
    )

    print(
        summary.to_string()
    )

    print()
    print(
        f"Total feature rows: "
        f"{len(df):,}"
    )

    print(
        f"Total columns: "
        f"{len(df.columns)}"
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
        "LEAKAGE-SAFE FEATURE ENGINEERING"
    )

    print(
        "=========================================="
    )

    # --------------------------------------------------------
    # Load + validate
    # --------------------------------------------------------

    df = load_data()

    validate_input(
        df
    )

    # --------------------------------------------------------
    # Chronology
    # --------------------------------------------------------

    df = prepare_chronology(
        df
    )

    # --------------------------------------------------------
    # Historical features
    # --------------------------------------------------------

    df = add_lag_features(
        df
    )

    df = add_rolling_features(
        df
    )

    # --------------------------------------------------------
    # Seasonality
    # --------------------------------------------------------

    df = add_seasonality(
        df
    )

    # --------------------------------------------------------
    # Future targets
    # --------------------------------------------------------

    df = add_targets(
        df
    )

    # --------------------------------------------------------
    # Rename current features
    # --------------------------------------------------------

    df = rename_current_features(
        df
    )

    # --------------------------------------------------------
    # Require complete historical context
    # --------------------------------------------------------

    df = remove_incomplete_feature_rows(
        df
    )

    # --------------------------------------------------------
    # Arrange
    # --------------------------------------------------------

    df = arrange_columns(
        df
    )

    # --------------------------------------------------------
    # Final validations
    # --------------------------------------------------------

    validate_feature_dates(
        df
    )

    validate_output(
        df
    )

    print_model_split_guidance(
        df
    )

    print_summary(
        df
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        date_format="%Y-%m-%d",
    )

    print()
    print(
        f"[PASS] Saved:"
    )

    print(
        f"       {OUTPUT_FILE}"
    )

    print()
    print(
        "=========================================="
    )

    print(
        "FEATURE ENGINEERING COMPLETE"
    )

    print(
        "=========================================="
    )


if __name__ == "__main__":
    main()