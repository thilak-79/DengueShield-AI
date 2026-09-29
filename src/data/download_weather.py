"""
DengueShield AI
Historical Weather Downloader

Downloads consistent ERA5 historical weather data from
Open-Meteo for 25 Sri Lankan district representative points.

Period:
    2014-01-01 -> 2025-12-31

Why start in 2014?
    Our first modelling year is 2015, but some WER Week 1
    observation windows begin during December of the
    previous calendar year.

Variables:
    Daily:
        temperature_2m_mean
        temperature_2m_min
        temperature_2m_max
        precipitation_sum

    Hourly:
        relative_humidity_2m

Hourly humidity is converted to daily mean humidity.

Output:
    data/raw/weather/daily/<district>.csv
"""

from pathlib import Path
import argparse
import sys
import time

import pandas as pd
import requests


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
# CONFIGURATION
# ============================================================

API_URL = (
    "https://archive-api.open-meteo.com/"
    "v1/archive"
)

COORDINATES_FILE = Path(
    "data/raw/weather/district_coordinates.csv"
)

OUTPUT_DIR = Path(
    "data/raw/weather/daily"
)

SUMMARY_FILE = Path(
    "data/interim/weather_download_summary.csv"
)


START_DATE = "2014-01-01"
END_DATE = "2025-12-31"

TIMEZONE = "Asia/Colombo"

# Use one consistent reanalysis source.
MODEL = "era5"

# Three-year chunks keep individual API responses
# manageable while avoiding hundreds of tiny requests.
CHUNK_YEARS = 3

REQUEST_TIMEOUT = 120

MAX_RETRIES = 6

REQUEST_DELAY_SECONDS = 0.5


DAILY_VARIABLES = [
    "temperature_2m_mean",
    "temperature_2m_min",
    "temperature_2m_max",
    "precipitation_sum",
]

HOURLY_VARIABLES = [
    "relative_humidity_2m",
]


# ============================================================
# HTTP SESSION
# ============================================================

SESSION = requests.Session()

SESSION.headers.update(
    {
        "User-Agent": (
            "DengueShield-AI/"
            "historical-weather-research"
        )
    }
)


# ============================================================
# HELPERS
# ============================================================

def district_slug(
    district,
):
    """
    Convert district name into safe filename.

    Example:
        Nuwara Eliya -> nuwara_eliya
    """

    return (
        district
        .strip()
        .lower()
        .replace(" ", "_")
    )


def make_chunks(
    start_date,
    end_date,
):
    """
    Split full date range into multi-year chunks.

    Example:
        2014-01-01 -> 2016-12-31
        2017-01-01 -> 2019-12-31
        ...
    """

    start = pd.Timestamp(
        start_date
    )

    final_end = pd.Timestamp(
        end_date
    )

    chunks = []

    current = start

    while current <= final_end:

        candidate_end = (
            current
            + pd.DateOffset(
                years=CHUNK_YEARS
            )
            - pd.Timedelta(
                days=1
            )
        )

        chunk_end = min(
            candidate_end,
            final_end,
        )

        chunks.append(
            (
                current.strftime(
                    "%Y-%m-%d"
                ),
                chunk_end.strftime(
                    "%Y-%m-%d"
                ),
            )
        )

        current = (
            chunk_end
            + pd.Timedelta(
                days=1
            )
        )

    return chunks


def expected_day_count():
    """
    Number of calendar days expected in
    START_DATE -> END_DATE inclusive.
    """

    return len(
        pd.date_range(
            START_DATE,
            END_DATE,
            freq="D",
        )
    )


# ============================================================
# API REQUEST
# ============================================================

def request_weather(
    latitude,
    longitude,
    start_date,
    end_date,
):
    """
    Download one date chunk.

    Retries automatically after temporary
    API/server errors.
    """

    params = {
        "latitude": latitude,
        "longitude": longitude,

        "start_date": start_date,
        "end_date": end_date,

        "daily": ",".join(
            DAILY_VARIABLES
        ),

        "hourly": ",".join(
            HOURLY_VARIABLES
        ),

        "timezone": TIMEZONE,

        "temperature_unit": (
            "celsius"
        ),

        "precipitation_unit": (
            "mm"
        ),

        "models": MODEL,
    }

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:

            response = SESSION.get(
                API_URL,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )

        except requests.RequestException as error:

            if attempt == MAX_RETRIES:
                raise

            wait_seconds = (
                2 ** attempt
            )

            print(
                f"          "
                f"[NETWORK RETRY] "
                f"{error}"
            )

            time.sleep(
                wait_seconds
            )

            continue

        # Retry rate-limit/server problems.
        if (
            response.status_code
            == 429
            or
            response.status_code
            >= 500
        ):

            if attempt == MAX_RETRIES:

                response.raise_for_status()

            wait_seconds = (
                2 ** attempt
            )

            print(
                f"          "
                f"[HTTP {response.status_code}] "
                f"retrying..."
            )

            time.sleep(
                wait_seconds
            )

            continue

        response.raise_for_status()

        payload = response.json()

        if payload.get(
            "error",
            False,
        ):

            raise RuntimeError(
                payload.get(
                    "reason",
                    "Unknown Open-Meteo error",
                )
            )

        return payload

    raise RuntimeError(
        "Weather request failed "
        "after all retries."
    )


# ============================================================
# RESPONSE -> DAILY DATAFRAME
# ============================================================

def payload_to_daily_df(
    payload,
):
    """
    Convert Open-Meteo daily + hourly response
    into one daily dataframe.

    Relative humidity is provided hourly,
    so we calculate daily mean humidity.
    """

    daily = payload.get(
        "daily"
    )

    hourly = payload.get(
        "hourly"
    )

    if daily is None:

        raise ValueError(
            "API response has no daily data."
        )

    if hourly is None:

        raise ValueError(
            "API response has no hourly data."
        )

    daily_df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                daily["time"]
            ),

            "temperature_mean": (
                daily[
                    "temperature_2m_mean"
                ]
            ),

            "temperature_min": (
                daily[
                    "temperature_2m_min"
                ]
            ),

            "temperature_max": (
                daily[
                    "temperature_2m_max"
                ]
            ),

            "precipitation_sum": (
                daily[
                    "precipitation_sum"
                ]
            ),
        }
    )

    hourly_df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                hourly["time"]
            ),

            "relative_humidity": (
                hourly[
                    "relative_humidity_2m"
                ]
            ),
        }
    )

    hourly_df[
        "date"
    ] = (
        hourly_df[
            "datetime"
        ]
        .dt.normalize()
    )

    humidity_daily = (
        hourly_df.groupby(
            "date",
            as_index=False,
        )
        .agg(
            humidity_mean=(
                "relative_humidity",
                "mean",
            )
        )
    )

    result = daily_df.merge(
        humidity_daily,
        on="date",
        how="left",
        validate="one_to_one",
    )

    return result


# ============================================================
# EXISTING FILE VALIDATION
# ============================================================

def existing_file_complete(
    output_file,
):
    """
    Return True only if an existing district
    file covers every required day.
    """

    if not output_file.exists():

        return False

    try:

        df = pd.read_csv(
            output_file,
            parse_dates=[
                "date"
            ],
        )

    except Exception:

        return False

    required_columns = {
        "date",
        "district",
        "temperature_mean",
        "temperature_min",
        "temperature_max",
        "humidity_mean",
        "precipitation_sum",
    }

    if not required_columns.issubset(
        df.columns
    ):

        return False

    if (
        df["date"].min()
        != pd.Timestamp(
            START_DATE
        )
    ):

        return False

    if (
        df["date"].max()
        != pd.Timestamp(
            END_DATE
        )
    ):

        return False

    if (
        df["date"].nunique()
        != expected_day_count()
    ):

        return False

    if df[
        "date"
    ].duplicated().any():

        return False

    weather_columns = [
        "temperature_mean",
        "temperature_min",
        "temperature_max",
        "humidity_mean",
        "precipitation_sum",
    ]

    if (
        df[
            weather_columns
        ]
        .isna()
        .any()
        .any()
    ):

        return False

    return True


# ============================================================
# DOWNLOAD ONE DISTRICT
# ============================================================

def download_district(
    district,
    latitude,
    longitude,
    force=False,
):
    """
    Download all weather for one district.
    """

    slug = district_slug(
        district
    )

    output_file = (
        OUTPUT_DIR
        /
        f"{slug}.csv"
    )

    print()
    print(
        "=" * 60
    )

    print(
        f"DISTRICT: {district}"
    )

    print(
        f"Coordinates: "
        f"{latitude}, {longitude}"
    )

    print(
        "=" * 60
    )

    if (
        not force
        and existing_file_complete(
            output_file
        )
    ):

        print(
            "[SKIP] Complete weather "
            "file already exists."
        )

        existing = pd.read_csv(
            output_file
        )

        return {
            "district": district,
            "status": "existing",
            "rows": len(
                existing
            ),
            "start_date": START_DATE,
            "end_date": END_DATE,
            "file": str(
                output_file
            ),
        }

    chunks = make_chunks(
        START_DATE,
        END_DATE,
    )

    frames = []

    for (
        chunk_start,
        chunk_end,
    ) in chunks:

        print(
            f"  [DOWNLOAD] "
            f"{chunk_start} "
            f"-> "
            f"{chunk_end}"
        )

        payload = request_weather(
            latitude=latitude,
            longitude=longitude,
            start_date=chunk_start,
            end_date=chunk_end,
        )

        chunk_df = (
            payload_to_daily_df(
                payload
            )
        )

        frames.append(
            chunk_df
        )

        time.sleep(
            REQUEST_DELAY_SECONDS
        )

    district_df = pd.concat(
        frames,
        ignore_index=True,
    )

    district_df = (
        district_df
        .drop_duplicates(
            subset=[
                "date"
            ],
            keep="last",
        )
        .sort_values(
            "date"
        )
        .reset_index(
            drop=True
        )
    )

    # Add location metadata.
    district_df.insert(
        0,
        "district",
        district,
    )

    district_df.insert(
        1,
        "latitude",
        latitude,
    )

    district_df.insert(
        2,
        "longitude",
        longitude,
    )

    district_df[
        "source_model"
    ] = MODEL

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    expected_rows = (
        expected_day_count()
    )

    actual_rows = len(
        district_df
    )

    if (
        actual_rows
        != expected_rows
    ):

        raise AssertionError(
            f"{district}: "
            f"expected "
            f"{expected_rows} days, "
            f"got "
            f"{actual_rows}"
        )

    if (
        district_df[
            "date"
        ]
        .duplicated()
        .any()
    ):

        raise AssertionError(
            f"{district}: duplicate "
            f"weather dates found."
        )

    weather_columns = [
        "temperature_mean",
        "temperature_min",
        "temperature_max",
        "humidity_mean",
        "precipitation_sum",
    ]

    missing = (
        district_df[
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
            "[WARNING] Missing values:"
        )

        print(
            missing.to_string()
        )

        raise AssertionError(
            f"{district}: missing "
            f"weather values found."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    district_df.to_csv(
        output_file,
        index=False,
        date_format="%Y-%m-%d",
    )

    print(
        f"[PASS] "
        f"{actual_rows} daily rows"
    )

    print(
        f"[SAVED] "
        f"{output_file}"
    )

    return {
        "district": district,
        "status": "downloaded",
        "rows": actual_rows,
        "start_date": (
            district_df[
                "date"
            ]
            .min()
            .strftime(
                "%Y-%m-%d"
            )
        ),
        "end_date": (
            district_df[
                "date"
            ]
            .max()
            .strftime(
                "%Y-%m-%d"
            )
        ),
        "file": str(
            output_file
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--district",
        type=str,
        default=None,
        help=(
            "Download only one district "
            "for testing."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Redownload even when a "
            "complete file exists."
        ),
    )

    args = parser.parse_args()

    print()
    print(
        "=========================================="
    )

    print(
        "DENGUESHIELD AI"
    )

    print(
        "ERA5 HISTORICAL WEATHER DOWNLOADER"
    )

    print(
        "=========================================="
    )

    if not (
        COORDINATES_FILE.exists()
    ):

        raise FileNotFoundError(
            f"Missing coordinate file: "
            f"{COORDINATES_FILE}"
        )

    coordinates = pd.read_csv(
        COORDINATES_FILE
    )

    required = {
        "district",
        "latitude",
        "longitude",
    }

    if not required.issubset(
        coordinates.columns
    ):

        raise ValueError(
            "Coordinate CSV must contain "
            "district, latitude, longitude"
        )

    if (
        coordinates[
            "district"
        ]
        .duplicated()
        .any()
    ):

        raise ValueError(
            "Duplicate districts found "
            "in coordinate CSV."
        )

    if (
        len(
            coordinates
        )
        != 25
    ):

        raise ValueError(
            f"Expected 25 districts, "
            f"found "
            f"{len(coordinates)}"
        )

    # --------------------------------------------
    # Optional one-district test
    # --------------------------------------------

    if args.district:

        mask = (
            coordinates[
                "district"
            ]
            .str.lower()
            ==
            args.district.lower()
        )

        coordinates = (
            coordinates[
                mask
            ]
        )

        if coordinates.empty:

            raise ValueError(
                f"Unknown district: "
                f"{args.district}"
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries = []

    for row in (
        coordinates
        .itertuples(
            index=False
        )
    ):

        result = (
            download_district(
                district=row.district,
                latitude=float(
                    row.latitude
                ),
                longitude=float(
                    row.longitude
                ),
                force=args.force,
            )
        )

        summaries.append(
            result
        )

    summary_df = pd.DataFrame(
        summaries
    )

    SUMMARY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print()
    print(
        "=========================================="
    )

    print(
        "WEATHER DOWNLOAD SUMMARY"
    )

    print(
        "=========================================="
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print(
        "[PASS] Weather download "
        "stage completed."
    )


if __name__ == "__main__":
    main()