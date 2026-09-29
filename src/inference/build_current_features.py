"""
DengueShield AI
Build Current 2026 Operational Features

Builds one feature row per district using:

- NDCU current / lagged cases
- ERA5 weather aggregated to exact NDCU weekly windows
- same historical lag / rolling feature definitions
- existing historical seasonality encoding

Output:
25 rows, one per Sri Lankan district.
"""

from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd


try:
    sys.stdout.reconfigure(
        encoding="utf-8",
        errors="replace",
    )
except Exception:
    pass


CASE_FILE = Path(
    "data/processed/"
    "dengue_weekly_historical.csv"
)

WEATHER_FILE = Path(
    "data/processed/operational/"
    "current_weather_weekly.csv"
)

REFERENCE_FEATURE_FILE = Path(
    "data/processed/"
    "ml_features_2015_2025.csv"
)

METADATA_FILE = Path(
    "data/processed/operational/"
    "current_data_metadata.json"
)

MODEL_FILE = Path(
    "models/operational/"
    "random_forest_1w_full_history.joblib"
)

OUTPUT_FILE = Path(
    "data/processed/operational/"
    "current_features_2026.csv"
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
        "BUILD CURRENT FEATURES"
    )
    print(
        "=========================================="
    )

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        metadata = json.load(
            file
        )

    year = int(
        metadata[
            "latest_year"
        ]
    )

    current_week = int(
        metadata[
            "latest_week"
        ]
    )

    required_weeks = [
        current_week,
        current_week - 1,
        current_week - 2,
        current_week - 3,
        current_week - 4,
    ]

    cases = pd.read_csv(
        CASE_FILE
    )

    weather = pd.read_csv(
        WEATHER_FILE,
        parse_dates=[
            "start_date",
            "end_date",
        ],
    )

    reference = pd.read_csv(
        REFERENCE_FEATURE_FILE
    )

    pipeline = joblib.load(
        MODEL_FILE
    )

    model_features = list(
        pipeline.feature_names_in_
    )

    # --------------------------------------------------------
    # SEASONAL VALUES
    #
    # EXACT reproduction of the historical feature pipeline:
    #
    # midpoint =
    #     start_date
    #     +
    #     (end_date - start_date) / 2
    #
    # seasonality uses midpoint day-of-year.
    # --------------------------------------------------------

    current_week_start = pd.Timestamp(
        metadata[
            "latest_week_start"
        ]
    )

    current_week_end = pd.Timestamp(
        metadata[
            "latest_week_end"
        ]
    )

    midpoint_date = (
        current_week_start
        +
        (
            current_week_end
            -
            current_week_start
        )
        /
        2
    )

    day_of_year = int(
        midpoint_date.dayofyear
    )

    week_sin = float(
        np.sin(
            2
            *
            np.pi
            *
            day_of_year
            /
            365.25
        )
    )

    week_cos = float(
        np.cos(
            2
            *
            np.pi
            *
            day_of_year
            /
            365.25
        )
    )

    print(
        f"[PASS] Reporting window: "
        f"{current_week_start.date()} "
        f"to "
        f"{current_week_end.date()}"
    )

    print(
        f"[PASS] Reporting midpoint: "
        f"{midpoint_date}"
    )

    print(
        f"[PASS] Seasonal day-of-year: "
        f"{day_of_year}"
    )

    print(
        f"[PASS] week_sin: "
        f"{week_sin:.10f}"
    )

    print(
        f"[PASS] week_cos: "
        f"{week_cos:.10f}"
    )

    # --------------------------------------------------------
    # CURRENT CASE MATRIX
    # --------------------------------------------------------

    current_cases = (
        cases[
            (
                cases[
                    "year"
                ]
                ==
                year
            )
            &
            (
                cases[
                    "week"
                ]
                .isin(
                    required_weeks
                )
            )
        ]
        .copy()
    )

    case_pivot = (
        current_cases.pivot(
            index="district",
            columns="week",
            values="cases",
        )
    )

    if len(
        case_pivot
    ) != 25:

        raise AssertionError(
            "Case history does not contain "
            "25 districts."
        )

    for week in required_weeks:

        if week not in case_pivot.columns:

            raise AssertionError(
                f"Missing case week {week}."
            )

    # --------------------------------------------------------
    # WEATHER MATRIX
    # --------------------------------------------------------

    weather = (
        weather[
            weather[
                "week"
            ]
            .isin(
                required_weeks
            )
        ]
        .copy()
    )

    if (
        weather.groupby(
            "week"
        )[
            "district"
        ]
        .nunique()
        .ne(
            25
        )
        .any()
    ):

        raise AssertionError(
            "Current weather does not have "
            "25 districts for every week."
        )

    weather_lookup = {
        (
            str(
                row[
                    "district"
                ]
            ),
            int(
                row[
                    "week"
                ]
            ),
        ):
        row
        for _,
        row
        in weather.iterrows()
    }

    rows = []

    for district in sorted(
        case_pivot.index
    ):

        def case(
            lag,
        ):
            return float(
                case_pivot.loc[
                    district,
                    current_week - lag,
                ]
            )

        def weather_value(
            lag,
            column,
        ):
            key = (
                district,
                current_week - lag,
            )

            if key not in weather_lookup:
                raise KeyError(
                    f"Missing weather "
                    f"{key}"
                )

            return float(
                weather_lookup[
                    key
                ][
                    column
                ]
            )

        cases_lags = [
            case(
                lag
            )
            for lag in range(
                1,
                5,
            )
        ]

        rainfall_lags = [
            weather_value(
                lag,
                "rainfall_sum",
            )
            for lag in range(
                1,
                5,
            )
        ]

        humidity_lags = [
            weather_value(
                lag,
                "humidity_mean",
            )
            for lag in range(
                1,
                5,
            )
        ]

        temperature_lags = [
            weather_value(
                lag,
                "temperature_mean",
            )
            for lag in range(
                1,
                5,
            )
        ]

        current_weather = (
            weather_lookup[
                (
                    district,
                    current_week,
                )
            ]
        )

        row = {

            "year":
            year,

            "week":
            current_week,

            "district":
            district,

            "start_date":
            current_weather[
                "start_date"
            ],

            "end_date":
            current_weather[
                "end_date"
            ],

            "forecast_origin_date":
            metadata[
                "latest_week_end"
            ],

            "target_date_1w":
            metadata[
                "forecast_target_start"
            ],

            # Current cases
            "cases_current":
            case(
                0
            ),

            # Current weather
            "temperature_current":
            float(
                current_weather[
                    "temperature_mean"
                ]
            ),

            "temperature_min_current":
            float(
                current_weather[
                    "temperature_min"
                ]
            ),

            "temperature_max_current":
            float(
                current_weather[
                    "temperature_max"
                ]
            ),

            "humidity_current":
            float(
                current_weather[
                    "humidity_mean"
                ]
            ),

            "rainfall_current":
            float(
                current_weather[
                    "rainfall_sum"
                ]
            ),

            "rain_days_current":
            float(
                current_weather[
                    "rain_days"
                ]
            ),

            # Cases lags
            "cases_lag_1":
            cases_lags[
                0
            ],

            "cases_lag_2":
            cases_lags[
                1
            ],

            "cases_lag_3":
            cases_lags[
                2
            ],

            "cases_lag_4":
            cases_lags[
                3
            ],

            "cases_rolling_2":
            float(
                np.mean(
                    cases_lags[
                        :2
                    ]
                )
            ),

            "cases_rolling_4":
            float(
                np.mean(
                    cases_lags
                )
            ),

            # Rainfall
            "rainfall_lag_1":
            rainfall_lags[
                0
            ],

            "rainfall_lag_2":
            rainfall_lags[
                1
            ],

            "rainfall_lag_3":
            rainfall_lags[
                2
            ],

            "rainfall_lag_4":
            rainfall_lags[
                3
            ],

            # 4-week accumulated rainfall
            "rainfall_rolling_4":
            float(
                np.sum(
                    rainfall_lags
                )
            ),

            # Humidity
            "humidity_lag_1":
            humidity_lags[
                0
            ],

            "humidity_lag_2":
            humidity_lags[
                1
            ],

            "humidity_lag_3":
            humidity_lags[
                2
            ],

            "humidity_lag_4":
            humidity_lags[
                3
            ],

            "humidity_rolling_4":
            float(
                np.mean(
                    humidity_lags
                )
            ),

            # Temperature
            "temperature_lag_1":
            temperature_lags[
                0
            ],

            "temperature_lag_2":
            temperature_lags[
                1
            ],

            "temperature_lag_3":
            temperature_lags[
                2
            ],

            "temperature_lag_4":
            temperature_lags[
                3
            ],

            "temperature_rolling_4":
            float(
                np.mean(
                    temperature_lags
                )
            ),

            # Rain days
            "rain_days_lag_1":
            weather_value(
                1,
                "rain_days",
            ),

            "rain_days_lag_2":
            weather_value(
                2,
                "rain_days",
            ),

            "rain_days_lag_3":
            weather_value(
                3,
                "rain_days",
            ),

            "rain_days_lag_4":
            weather_value(
                4,
                "rain_days",
            ),

            # Exact historical seasonality encoding
            "week_sin":
            week_sin,

            "week_cos":
            week_cos,

            "case_source":
            "NDCU",

            "weather_source":
            "Open-Meteo ERA5",

            "source_semantics_warning":
            True,
        }

        rows.append(
            row
        )

    features = pd.DataFrame(
        rows
    )

    missing_model_features = (
        set(
            model_features
        )
        -
        set(
            features.columns
        )
    )

    if missing_model_features:

        raise AssertionError(
            f"Missing operational model "
            f"features: "
            f"{sorted(missing_model_features)}"
        )

    if (
        features[
            model_features
        ]
        .isna()
        .any()
        .any()
    ):

        missing_table = (
            features[
                model_features
            ]
            .isna()
            .sum()
        )

        raise AssertionError(
            "Missing feature values:\n"
            f"{missing_table[missing_table > 0]}"
        )

    # Verify model preprocessor accepts the data.
    pipeline.named_steps[
        "preprocessor"
    ].transform(
        features[
            model_features
        ]
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    features.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"[PASS] Rows: "
        f"{len(features)}"
    )

    print(
        f"[PASS] Districts: "
        f"{features['district'].nunique()}"
    )

    print(
        f"[PASS] Model features: "
        f"{len(model_features)}"
    )

    print(
        "[PASS] No missing model inputs"
    )

    print(
        "[PASS] Saved:"
    )

    print(
        f"       {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()