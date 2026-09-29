"""
DengueShield AI
Weekly Weather Feature Builder

Purpose
-------
Aggregate daily ERA5 weather into the exact date windows used
by the Sri Lankan Weekly Epidemiological Report dengue data.

Important:
We DO NOT assume that "week 1" means ISO Week 1.

Instead we use:

    start_date
    end_date

from each dengue surveillance observation.

Input:
    data/processed/dengue_weekly_2006_2025.csv

    data/raw/weather/daily/*.csv

Output:
    data/processed/weather_weekly_2015_2025.csv

    data/processed/
    dengue_weather_weekly_2015_2025.csv

Weather features:
    temperature_mean
    temperature_min
    temperature_max
    humidity_mean
    rainfall_sum
    rain_days

Definition:
    rain_days = number of days with precipitation >= 1 mm
"""

from pathlib import Path
import sys

import pandas as pd


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

DENGUE_FILE = Path(
    "data/processed/"
    "dengue_weekly_2006_2025.csv"
)

WEATHER_DIR = Path(
    "data/raw/weather/daily"
)

WEEKLY_WEATHER_FILE = Path(
    "data/processed/"
    "weather_weekly_2015_2025.csv"
)

COMBINED_FILE = Path(
    "data/processed/"
    "dengue_weather_weekly_2015_2025.csv"
)


# ============================================================
# MODEL PERIOD
# ============================================================

START_YEAR = 2015
END_YEAR = 2025


# A wet/rain day definition used here:
# daily precipitation >= 1 mm.
RAIN_DAY_THRESHOLD_MM = 1.0


# ============================================================
# LOAD DENGUE
# ============================================================

def load_dengue():

    if not DENGUE_FILE.exists():

        raise FileNotFoundError(
            f"Missing dengue dataset: "
            f"{DENGUE_FILE}"
        )

    df = pd.read_csv(
        DENGUE_FILE,
        parse_dates=[
            "start_date",
            "end_date",
        ],
    )

    df = (
        df[
            (
                df["year"]
                >= START_YEAR
            )
            &
            (
                df["year"]
                <= END_YEAR
            )
        ]
        .copy()
    )

    df = (
        df.sort_values(
            [
                "year",
                "week",
                "district",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    print(
        f"[PASS] Dengue rows "
        f"{START_YEAR}-{END_YEAR}: "
        f"{len(df):,}"
    )

    return df


# ============================================================
# LOAD DAILY WEATHER
# ============================================================

def load_weather():

    files = sorted(
        WEATHER_DIR.glob(
            "*.csv"
        )
    )

    if len(files) != 25:

        raise RuntimeError(
            f"Expected 25 weather files, "
            f"found {len(files)}."
        )

    frames = []

    for file in files:

        df = pd.read_csv(
            file,
            parse_dates=[
                "date"
            ],
        )

        frames.append(
            df
        )

    weather = pd.concat(
        frames,
        ignore_index=True,
    )

    weather = (
        weather.sort_values(
            [
                "district",
                "date",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    duplicates = (
        weather.duplicated(
            subset=[
                "district",
                "date",
            ]
        )
    )

    if duplicates.any():

        raise AssertionError(
            "Duplicate district-date "
            "weather rows detected."
        )

    required_columns = [
        "temperature_mean",
        "temperature_min",
        "temperature_max",
        "humidity_mean",
        "precipitation_sum",
    ]

    if (
        weather[
            required_columns
        ]
        .isna()
        .any()
        .any()
    ):

        raise AssertionError(
            "Missing weather values detected."
        )

    print(
        f"[PASS] Daily weather rows: "
        f"{len(weather):,}"
    )

    return weather


# ============================================================
# BUILD WEEKLY FEATURES
# ============================================================

def build_weekly_weather(
    dengue,
    weather,
):

    weather_by_district = {}

    for (
        district,
        district_weather,
    ) in weather.groupby(
        "district"
    ):

        temp = (
            district_weather
            .sort_values(
                "date"
            )
            .set_index(
                "date"
            )
        )

        weather_by_district[
            district
        ] = temp

    output_rows = []

    errors = []

    print(
        "\n[BUILD] Aggregating exact "
        "surveillance windows..."
    )

    for index, row in (
        dengue.iterrows()
    ):

        district = (
            row[
                "district"
            ]
        )

        start_date = (
            row[
                "start_date"
            ]
        )

        end_date = (
            row[
                "end_date"
            ]
        )

        if district not in (
            weather_by_district
        ):

            errors.append(
                (
                    row["year"],
                    row["week"],
                    district,
                    "district missing",
                )
            )

            continue

        district_weather = (
            weather_by_district[
                district
            ]
        )

        period = (
            district_weather
            .loc[
                start_date:end_date
            ]
            .copy()
        )

        expected_days = (
            end_date
            -
            start_date
        ).days + 1

        actual_days = len(
            period
        )

        if (
            actual_days
            != expected_days
        ):

            errors.append(
                (
                    row["year"],
                    row["week"],
                    district,
                    (
                        f"expected "
                        f"{expected_days} days, "
                        f"found "
                        f"{actual_days}"
                    ),
                )
            )

            continue

        output_rows.append(
            {
                "year": int(
                    row["year"]
                ),

                "week": int(
                    row["week"]
                ),

                "start_date": (
                    start_date
                ),

                "end_date": (
                    end_date
                ),

                "district": (
                    district
                ),

                "weather_days": (
                    actual_days
                ),

                "temperature_mean": (
                    period[
                        "temperature_mean"
                    ].mean()
                ),

                "temperature_min": (
                    period[
                        "temperature_min"
                    ].min()
                ),

                "temperature_max": (
                    period[
                        "temperature_max"
                    ].max()
                ),

                "humidity_mean": (
                    period[
                        "humidity_mean"
                    ].mean()
                ),

                "rainfall_sum": (
                    period[
                        "precipitation_sum"
                    ].sum()
                ),

                "rain_days": int(
                    (
                        period[
                            "precipitation_sum"
                        ]
                        >=
                        RAIN_DAY_THRESHOLD_MM
                    ).sum()
                ),
            }
        )

        if (
            (index + 1)
            % 2000
            == 0
        ):

            print(
                f"          Processed "
                f"{index + 1:,} rows"
            )

    if errors:

        print(
            "\n[ERROR] Weather-window "
            "problems detected:"
        )

        for error in (
            errors[:30]
        ):

            print(
                error
            )

        raise AssertionError(
            f"{len(errors)} surveillance "
            f"windows could not be "
            f"matched to weather."
        )

    weekly = pd.DataFrame(
        output_rows
    )

    return weekly


# ============================================================
# VALIDATION
# ============================================================

def validate_weekly(
    dengue,
    weekly,
):

    print()
    print(
        "=========================================="
    )

    print(
        "VALIDATING WEEKLY WEATHER"
    )

    print(
        "=========================================="
    )

    print(
        f"Dengue rows: "
        f"{len(dengue):,}"
    )

    print(
        f"Weather rows: "
        f"{len(weekly):,}"
    )

    if (
        len(dengue)
        != len(weekly)
    ):

        raise AssertionError(
            "Weekly weather row count "
            "does not match dengue data."
        )

    duplicates = (
        weekly.duplicated(
            subset=[
                "year",
                "week",
                "district",
            ]
        )
    )

    if duplicates.any():

        raise AssertionError(
            "Duplicate weekly weather "
            "rows detected."
        )

    weather_columns = [
        "temperature_mean",
        "temperature_min",
        "temperature_max",
        "humidity_mean",
        "rainfall_sum",
        "rain_days",
    ]

    missing = (
        weekly[
            weather_columns
        ]
        .isna()
        .sum()
    )

    if (
        missing.sum()
        > 0
    ):

        print(
            missing.to_string()
        )

        raise AssertionError(
            "Missing weekly weather "
            "features detected."
        )

    print(
        "[PASS] Row count matches "
        "dengue dataset"
    )

    print(
        "[PASS] No duplicate "
        "district-week rows"
    )

    print(
        "[PASS] No missing "
        "weather features"
    )


# ============================================================
# MERGE DENGUE + WEATHER
# ============================================================

def build_combined(
    dengue,
    weekly,
):

    keys = [
        "year",
        "week",
        "start_date",
        "end_date",
        "district",
    ]

    combined = (
        dengue.merge(
            weekly,
            on=keys,
            how="left",
            validate="one_to_one",
        )
    )

    feature_columns = [
        "temperature_mean",
        "temperature_min",
        "temperature_max",
        "humidity_mean",
        "rainfall_sum",
        "rain_days",
    ]

    if (
        combined[
            feature_columns
        ]
        .isna()
        .any()
        .any()
    ):

        raise AssertionError(
            "Merge created missing "
            "weather features."
        )

    return combined


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    combined,
):

    print()
    print(
        "=========================================="
    )

    print(
        "DENGUE + WEATHER DATASET SUMMARY"
    )

    print(
        "=========================================="
    )

    summary = (
        combined.groupby(
            "year"
        )
        .agg(
            weeks=(
                "week",
                "nunique",
            ),
            rows=(
                "district",
                "size",
            ),
            mean_cases=(
                "cases",
                "mean",
            ),
            mean_rainfall=(
                "rainfall_sum",
                "mean",
            ),
            mean_temperature=(
                "temperature_mean",
                "mean",
            ),
            mean_humidity=(
                "humidity_mean",
                "mean",
            ),
        )
    )

    print(
        summary.to_string()
    )

    print()
    print(
        f"Total rows: "
        f"{len(combined):,}"
    )

    print(
        f"Districts: "
        f"{combined['district'].nunique()}"
    )

    print(
        f"Years: "
        f"{combined['year'].min()} "
        f"- "
        f"{combined['year'].max()}"
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
        "BUILD WEEKLY WEATHER FEATURES"
    )

    print(
        "=========================================="
    )

    dengue = (
        load_dengue()
    )

    weather = (
        load_weather()
    )

    weekly = (
        build_weekly_weather(
            dengue,
            weather,
        )
    )

    validate_weekly(
        dengue,
        weekly,
    )

    WEEKLY_WEATHER_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    weekly.to_csv(
        WEEKLY_WEATHER_FILE,
        index=False,
        date_format="%Y-%m-%d",
    )

    print(
        f"\n[PASS] Saved:"
    )

    print(
        f"       "
        f"{WEEKLY_WEATHER_FILE}"
    )

    combined = (
        build_combined(
            dengue,
            weekly,
        )
    )

    combined.to_csv(
        COMBINED_FILE,
        index=False,
        date_format="%Y-%m-%d",
    )

    print(
        f"\n[PASS] Saved:"
    )

    print(
        f"       "
        f"{COMBINED_FILE}"
    )

    print_summary(
        combined
    )

    print()
    print(
        "=========================================="
    )

    print(
        "WEEKLY WEATHER BUILD COMPLETE"
    )

    print(
        "=========================================="
    )


if __name__ == "__main__":
    main()