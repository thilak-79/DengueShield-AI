"""
DengueShield AI
Current Weather Downloader

Downloads ERA5 weather through Open-Meteo for the latest
five NDCU weekly windows required by the 1-week RF model.
"""

from pathlib import Path
import json
import time
import sys

import pandas as pd
import requests


try:
    sys.stdout.reconfigure(
        encoding="utf-8",
        errors="replace",
    )
except Exception:
    pass


COORDINATE_FILE = Path(
    "data/raw/weather/district_coordinates.csv"
)

CALENDAR_FILE = Path(
    "data/processed/operational/"
    "current_week_calendar.csv"
)

METADATA_FILE = Path(
    "data/processed/operational/"
    "current_data_metadata.json"
)

DAILY_FILE = Path(
    "data/interim/current_weather_daily.csv"
)

WEEKLY_FILE = Path(
    "data/processed/operational/"
    "current_weather_weekly.csv"
)

API_URL = (
    "https://archive-api.open-meteo.com/"
    "v1/archive"
)


def detect_column(
    columns,
    candidates,
):

    lookup = {
        str(
            col
        ).lower():
        col
        for col
        in columns
    }

    for candidate in candidates:

        if candidate.lower() in lookup:

            return lookup[
                candidate.lower()
            ]

    raise ValueError(
        f"Could not find any of "
        f"{candidates} in columns "
        f"{list(columns)}"
    )


def download_district(
    session,
    district,
    latitude,
    longitude,
    start_date,
    end_date,
):

    params = {

        "latitude":
        float(
            latitude
        ),

        "longitude":
        float(
            longitude
        ),

        "start_date":
        str(
            start_date
        ),

        "end_date":
        str(
            end_date
        ),

        "daily":
        (
            "temperature_2m_mean,"
            "temperature_2m_min,"
            "temperature_2m_max,"
            "precipitation_sum"
        ),

        "hourly":
        "relative_humidity_2m",

        "timezone":
        "Asia/Colombo",

        "models":
        "era5",
    }

    last_error = None

    for attempt in range(
        1,
        4,
    ):

        try:

            response = session.get(
                API_URL,
                params=params,
                timeout=60,
            )

            response.raise_for_status()

            data = response.json()

            daily = pd.DataFrame(
                {
                    "date":
                    pd.to_datetime(
                        data[
                            "daily"
                        ][
                            "time"
                        ]
                    ),

                    "temperature_mean":
                    data[
                        "daily"
                    ][
                        "temperature_2m_mean"
                    ],

                    "temperature_min":
                    data[
                        "daily"
                    ][
                        "temperature_2m_min"
                    ],

                    "temperature_max":
                    data[
                        "daily"
                    ][
                        "temperature_2m_max"
                    ],

                    "rainfall":
                    data[
                        "daily"
                    ][
                        "precipitation_sum"
                    ],
                }
            )

            hourly = pd.DataFrame(
                {
                    "time":
                    pd.to_datetime(
                        data[
                            "hourly"
                        ][
                            "time"
                        ]
                    ),

                    "humidity":
                    data[
                        "hourly"
                    ][
                        "relative_humidity_2m"
                    ],
                }
            )

            hourly[
                "date"
            ] = (
                hourly[
                    "time"
                ]
                .dt.normalize()
            )

            humidity_daily = (
                hourly.groupby(
                    "date",
                    as_index=False,
                )[
                    "humidity"
                ]
                .mean()
                .rename(
                    columns={
                        "humidity":
                        "humidity_mean"
                    }
                )
            )

            daily = daily.merge(
                humidity_daily,
                on="date",
                how="left",
                validate="one_to_one",
            )

            daily[
                "district"
            ] = district

            return daily

        except Exception as exc:

            last_error = exc

            print(
                f"[WARN] {district} "
                f"attempt {attempt}: {exc}"
            )

            time.sleep(
                2 * attempt
            )

    raise RuntimeError(
        f"Weather download failed for "
        f"{district}: {last_error}"
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
        "CURRENT WEATHER DOWNLOAD"
    )
    print(
        "=========================================="
    )

    if not COORDINATE_FILE.exists():
        raise FileNotFoundError(
            COORDINATE_FILE
        )

    if not CALENDAR_FILE.exists():
        raise FileNotFoundError(
            CALENDAR_FILE
        )

    coordinates = pd.read_csv(
        COORDINATE_FILE
    )

    calendar = pd.read_csv(
        CALENDAR_FILE,
        parse_dates=[
            "start_date",
            "end_date",
        ],
    )

    district_col = detect_column(
        coordinates.columns,
        [
            "district",
            "name",
            "District",
        ],
    )

    latitude_col = detect_column(
        coordinates.columns,
        [
            "latitude",
            "lat",
        ],
    )

    longitude_col = detect_column(
        coordinates.columns,
        [
            "longitude",
            "lon",
            "lng",
        ],
    )

    if (
        coordinates[
            district_col
        ]
        .nunique()
        !=
        25
    ):
        raise AssertionError(
            "Expected 25 district coordinates."
        )

    start_date = (
        calendar[
            "start_date"
        ]
        .min()
        .date()
    )

    end_date = (
        calendar[
            "end_date"
        ]
        .max()
        .date()
    )

    print(
        f"[INFO] Downloading "
        f"{start_date} to {end_date}"
    )

    print(
        f"[INFO] Districts: "
        f"{coordinates[district_col].nunique()}"
    )

    session = requests.Session()

    daily_frames = []

    for index, row in (
        coordinates
        .sort_values(
            district_col
        )
        .iterrows()
    ):

        district = str(
            row[
                district_col
            ]
        ).strip()

        print(
            f"[DOWNLOAD] {district}"
        )

        frame = download_district(
            session=session,
            district=district,
            latitude=row[
                latitude_col
            ],
            longitude=row[
                longitude_col
            ],
            start_date=start_date,
            end_date=end_date,
        )

        daily_frames.append(
            frame
        )

        time.sleep(
            0.25
        )

    daily = pd.concat(
        daily_frames,
        ignore_index=True,
    )

    numeric_columns = [
        "temperature_mean",
        "temperature_min",
        "temperature_max",
        "rainfall",
        "humidity_mean",
    ]

    for column in numeric_columns:

        daily[
            column
        ] = pd.to_numeric(
            daily[
                column
            ],
            errors="coerce",
        )

    if (
        daily[
            numeric_columns
        ]
        .isna()
        .any()
        .any()
    ):

        raise AssertionError(
            "Missing downloaded weather values."
        )

    DAILY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    daily.to_csv(
        DAILY_FILE,
        index=False,
        date_format="%Y-%m-%d",
    )

    # --------------------------------------------------------
    # EXACT NDCU WEEK AGGREGATION
    # --------------------------------------------------------

    weekly_rows = []

    for district, district_daily in (
        daily.groupby(
            "district"
        )
    ):

        for _, week_row in (
            calendar.iterrows()
        ):

            start = (
                week_row[
                    "start_date"
                ]
            )

            end = (
                week_row[
                    "end_date"
                ]
            )

            subset = (
                district_daily[
                    (
                        district_daily[
                            "date"
                        ]
                        >=
                        start
                    )
                    &
                    (
                        district_daily[
                            "date"
                        ]
                        <=
                        end
                    )
                ]
            )

            if len(
                subset
            ) != 7:

                raise AssertionError(
                    f"{district} week "
                    f"{week_row['week']} "
                    f"has {len(subset)} "
                    f"weather days, expected 7."
                )

            weekly_rows.append(
                {
                    "year":
                    int(
                        end.year
                    ),

                    "week":
                    int(
                        week_row[
                            "week"
                        ]
                    ),

                    "district":
                    district,

                    "start_date":
                    start,

                    "end_date":
                    end,

                    "temperature_mean":
                    float(
                        subset[
                            "temperature_mean"
                        ]
                        .mean()
                    ),

                    "temperature_min":
                    float(
                        subset[
                            "temperature_min"
                        ]
                        .min()
                    ),

                    "temperature_max":
                    float(
                        subset[
                            "temperature_max"
                        ]
                        .max()
                    ),

                    "humidity_mean":
                    float(
                        subset[
                            "humidity_mean"
                        ]
                        .mean()
                    ),

                    "rainfall_sum":
                    float(
                        subset[
                            "rainfall"
                        ]
                        .sum()
                    ),

                    "rain_days":
                    int(
                        (
                            subset[
                                "rainfall"
                            ]
                            >=
                            1.0
                        )
                        .sum()
                    ),

                    "daily_observations":
                    len(
                        subset
                    ),
                }
            )

    weekly = pd.DataFrame(
        weekly_rows
    )

    expected_rows = (
        25
        *
        len(
            calendar
        )
    )

    if len(
        weekly
    ) != expected_rows:

        raise AssertionError(
            f"Expected {expected_rows} "
            f"weekly rows; got "
            f"{len(weekly)}."
        )

    if (
        weekly.groupby(
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
            "A week does not contain "
            "all 25 districts."
        )

    WEEKLY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    weekly.to_csv(
        WEEKLY_FILE,
        index=False,
        date_format="%Y-%m-%d",
    )

    print()
    print(
        "[PASS] Current weather complete"
    )

    print(
        f"[PASS] Daily rows: "
        f"{len(daily):,}"
    )

    print(
        f"[PASS] Weekly rows: "
        f"{len(weekly):,}"
    )

    print(
        f"[PASS] Saved: "
        f"{WEEKLY_FILE}"
    )


if __name__ == "__main__":
    main()