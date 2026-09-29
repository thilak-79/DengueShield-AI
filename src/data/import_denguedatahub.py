"""
DengueShield AI
Sri Lanka Historical Dengue Data Importer

Source
------
denguedatahub:
Weekly dengue surveillance data derived from Weekly
Epidemiological Reports published by the Epidemiology Unit,
Ministry of Health, Sri Lanka.

Purpose
-------
1. Download srilanka_weekly_data.rda
2. Read it using Python
3. Keep historical data up to 2025
4. Normalize district names
5. Convert 26 surveillance units -> 25 geographic districts
6. Save clean historical CSV files
7. Compare overlapping 2025 data with our independently
   validated NDCU-derived dataset

We deliberately exclude denguedatahub 2026 observations.
Our own validated NDCU pipeline remains the source for 2026.
"""

from pathlib import Path
import sys

import pandas as pd
import pyreadr
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
# PATHS
# ============================================================

RAW_DIR = Path(
    "data/raw/dengue"
)

INTERIM_DIR = Path(
    "data/interim"
)

PROCESSED_DIR = Path(
    "data/processed"
)


RAW_RDA_FILE = (
    RAW_DIR
    / "denguedatahub_srilanka_weekly.rda"
)

RDHS_OUTPUT_FILE = (
    INTERIM_DIR
    / "denguedatahub_weekly_rdhs.csv"
)

GEOGRAPHIC_OUTPUT_FILE = (
    PROCESSED_DIR
    / "dengue_weekly_2006_2025.csv"
)

OVERLAP_OUTPUT_FILE = (
    INTERIM_DIR
    / "denguedatahub_vs_ndcu_2025_overlap.csv"
)


# Our existing independently validated file.
NDCU_REFERENCE_FILE = (
    PROCESSED_DIR
    / "dengue_weekly_historical.csv"
)


# ============================================================
# EXTERNAL DATA URL
# ============================================================

# Written in parts only for readability.
DATA_URL = (
    "https://"
    "raw.githubusercontent.com/"
    "thiyangt/denguedatahub/"
    "main/data/"
    "srilanka_weekly_data.rda"
)


# ============================================================
# DOWNLOAD
# ============================================================

def download_dataset():
    """
    Download the official package data file from the
    denguedatahub GitHub repository.
    """

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if RAW_RDA_FILE.exists():

        print(
            f"[SKIP] Dataset already exists:"
        )

        print(
            f"       {RAW_RDA_FILE}"
        )

        return

    print(
        "[DOWNLOAD] "
        "Downloading srilanka_weekly_data.rda..."
    )

    response = requests.get(
        DATA_URL,
        timeout=60,
    )

    response.raise_for_status()

    RAW_RDA_FILE.write_bytes(
        response.content
    )

    print(
        f"[PASS] Downloaded "
        f"{len(response.content):,} bytes"
    )


# ============================================================
# READ R DATA
# ============================================================

def load_rda():
    """
    Read .rda file using pyreadr.
    """

    print(
        "\n[READ] Loading R dataset..."
    )

    result = pyreadr.read_r(
        str(
            RAW_RDA_FILE
        )
    )

    print(
        "Objects found:",
        list(
            result.keys()
        ),
    )

    if (
        "srilanka_weekly_data"
        in result
    ):

        df = result[
            "srilanka_weekly_data"
        ].copy()

    elif len(result) == 1:

        df = next(
            iter(
                result.values()
            )
        ).copy()

    else:

        raise RuntimeError(
            "Unable to identify "
            "srilanka_weekly_data object."
        )

    print(
        f"[PASS] Loaded "
        f"{len(df):,} rows"
    )

    return df


# ============================================================
# CLEANING
# ============================================================

def clean_dataset(df):
    """
    Normalize columns, types and district names.
    """

    df = df.copy()

    # --------------------------------------------
    # Normalize column names
    # --------------------------------------------

    df.columns = [
        str(column)
        .strip()
        .lower()
        .replace(".", "_")
        for column
        in df.columns
    ]

    print(
        "\nColumns:"
    )

    print(
        list(
            df.columns
        )
    )

    required = {
        "year",
        "week",
        "district",
        "cases",
    }

    missing = (
        required
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

    # --------------------------------------------
    # Convert types
    # --------------------------------------------

    df["year"] = pd.to_numeric(
        df["year"],
        errors="coerce",
    )

    df["week"] = pd.to_numeric(
        df["week"],
        errors="coerce",
    )

    df["cases"] = pd.to_numeric(
        df["cases"],
        errors="coerce",
    )

    # --------------------------------------------
    # Remove invalid year/week rows
    # --------------------------------------------

    df = df.dropna(
        subset=[
            "year",
            "week",
            "district",
        ]
    )

    df["year"] = (
        df["year"]
        .astype(int)
    )

    df["week"] = (
        df["week"]
        .astype(int)
    )

    # --------------------------------------------
    # IMPORTANT:
    # only use data up to 2025.
    #
    # 2026 remains from our own NDCU pipeline.
    # --------------------------------------------

    df = (
        df[
            df["year"]
            <= 2025
        ]
        .copy()
    )

    # --------------------------------------------
    # Standardize dates
    # --------------------------------------------

    if (
        "start_date"
        in df.columns
    ):

        df["start_date"] = (
            pd.to_datetime(
                df["start_date"],
                errors="coerce",
            )
        )

    if (
        "end_date"
        in df.columns
    ):

        df["end_date"] = (
            pd.to_datetime(
                df["end_date"],
                errors="coerce",
            )
        )

    # --------------------------------------------
    # Standardize district strings
    # --------------------------------------------

    df["district"] = (
        df["district"]
        .astype(str)
        .str.strip()
    )

    district_mapping = {
        "Kalmune": "Kalmunai",
        "Kalmunai": "Kalmunai",

        "Hambanthota": "Hambantota",
        "Hambantota": "Hambantota",

        "NuwaraEliya": "Nuwara Eliya",
        "Nuwara Eliya": "Nuwara Eliya",

        "Rathnapura": "Ratnapura",
        "Ratnapura": "Ratnapura",

        "Kilinochchi": "Kilinochchi",
        "Killinochchi": "Kilinochchi",
    }

    df["district"] = (
        df["district"]
        .replace(
            district_mapping
        )
    )

    # --------------------------------------------
    # Sort
    # --------------------------------------------

    sort_columns = [
        "year",
        "week",
        "district",
    ]

    df = (
        df.sort_values(
            sort_columns
        )
        .reset_index(
            drop=True
        )
    )

    return df


# ============================================================
# VALIDATE RDHS/SURVEILLANCE DATA
# ============================================================

def validate_rdhs(df):

    print(
        "\n=========================================="
    )

    print(
        "VALIDATING DENGUEDATAHUB DATA"
    )

    print(
        "=========================================="
    )

    print(
        f"Years: "
        f"{df['year'].min()} "
        f"- "
        f"{df['year'].max()}"
    )

    print(
        f"Rows: "
        f"{len(df):,}"
    )

    print(
        "\nDistrict/RDHS names:"
    )

    districts = sorted(
        df[
            "district"
        ].unique()
    )

    for district in districts:

        print(
            f"  {district}"
        )

    print(
        f"\nUnique units: "
        f"{len(districts)}"
    )

    # --------------------------------------------
    # Duplicates
    # --------------------------------------------

    duplicates = (
        df.duplicated(
            subset=[
                "year",
                "week",
                "district",
            ]
        )
    )

    print(
        f"Duplicate "
        f"year-week-district rows: "
        f"{duplicates.sum()}"
    )

    # --------------------------------------------
    # Missing case values
    # --------------------------------------------

    missing_cases = (
        df[
            "cases"
        ]
        .isna()
        .sum()
    )

    print(
        f"Missing case values: "
        f"{missing_cases}"
    )

    # --------------------------------------------
    # Invalid negative counts
    # --------------------------------------------

    negative = (
        df[
            "cases"
        ]
        .dropna()
        .lt(0)
        .sum()
    )

    print(
        f"Negative case values: "
        f"{negative}"
    )

    if negative != 0:

        raise ValueError(
            "Negative dengue case "
            "values detected."
        )


# ============================================================
# 26 SURVEILLANCE UNITS -> 25 DISTRICTS
# ============================================================

def build_geographic_dataset(
    df
):
    """
    Ampara geographic district consists of:

        Ampara
        Kalmunai

    We combine them for district-level ML.
    """

    geo = df.copy()

    geo[
        "geographic_district"
    ] = geo[
        "district"
    ]

    geo.loc[
        geo["district"]
        ==
        "Kalmunai",
        "geographic_district"
    ] = "Ampara"

    grouping_columns = [
        "year",
        "week",
        "geographic_district",
    ]

    # Keep dates if available.
    if (
        "start_date"
        in geo.columns
    ):

        grouping_columns.insert(
            2,
            "start_date",
        )

    if (
        "end_date"
        in geo.columns
    ):

        position = (
            3
            if (
                "start_date"
                in geo.columns
            )
            else 2
        )

        grouping_columns.insert(
            position,
            "end_date",
        )

    geo = (
        geo.groupby(
            grouping_columns,
            as_index=False,
            dropna=False,
        )
        .agg(
            cases=(
                "cases",
                lambda values:
                values.sum(
                    min_count=1
                ),
            )
        )
    )

    geo = geo.rename(
        columns={
            "geographic_district":
            "district"
        }
    )

    geo = (
        geo.sort_values(
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

    return geo


# ============================================================
# VALIDATE GEOGRAPHIC DATASET
# ============================================================

def validate_geographic(
    df
):

    print(
        "\n=========================================="
    )

    print(
        "GEOGRAPHIC DATASET SUMMARY"
    )

    print(
        "=========================================="
    )

    print(
        f"Rows: "
        f"{len(df):,}"
    )

    print(
        f"Years: "
        f"{df['year'].min()} "
        f"- "
        f"{df['year'].max()}"
    )

    # Number of districts per year/week
    counts = (
        df.groupby(
            [
                "year",
                "week",
            ]
        )[
            "district"
        ]
        .nunique()
    )

    print(
        "\nDistrict-count frequency:"
    )

    print(
        counts
        .value_counts()
        .sort_index()
        .to_string()
    )

    incomplete = (
        counts[
            counts != 25
        ]
    )

    if incomplete.empty:

        print(
            "\n[PASS] "
            "Every available week has "
            "25 geographic districts"
        )

    else:

        print(
            "\n[WARNING] "
            "Some historical weeks do not "
            "contain 25 districts."
        )

        print(
            incomplete.head(
                30
            ).to_string()
        )


# ============================================================
# COMPARE 2025 OVERLAP WITH OUR NDCU DATA
# ============================================================

def compare_2025_overlap(
    hub_df
):
    """
    Compare the imported historical source with our
    independently validated NDCU-derived 2025 data.

    We do not overwrite either source here.

    The purpose is to understand consistency first.
    """

    print(
        "\n=========================================="
    )

    print(
        "2025 NDCU OVERLAP CHECK"
    )

    print(
        "=========================================="
    )

    if not (
        NDCU_REFERENCE_FILE.exists()
    ):

        print(
            "[WARNING] "
            "NDCU historical reference "
            "file not found."
        )

        return

    ndcu = pd.read_csv(
        NDCU_REFERENCE_FILE
    )

    ndcu = (
        ndcu[
            ndcu[
                "year"
            ]
            == 2025
        ][
            [
                "year",
                "week",
                "district",
                "cases",
            ]
        ]
        .copy()
    )

    # Our NDCU dataset currently contains Weeks 1-37.
    max_ndcu_week = int(
        ndcu[
            "week"
        ].max()
    )

    hub = (
        hub_df[
            (
                hub_df[
                    "year"
                ]
                == 2025
            )
            &
            (
                hub_df[
                    "week"
                ]
                <= max_ndcu_week
            )
        ][
            [
                "year",
                "week",
                "district",
                "cases",
            ]
        ]
        .copy()
    )

    comparison = (
        hub.merge(
            ndcu,
            on=[
                "year",
                "week",
                "district",
            ],
            how="inner",
            suffixes=(
                "_hub",
                "_ndcu",
            ),
        )
    )

    comparison[
        "difference"
    ] = (
        comparison[
            "cases_hub"
        ]
        -
        comparison[
            "cases_ndcu"
        ]
    )

    comparison[
        "exact_match"
    ] = (
        comparison[
            "difference"
        ]
        == 0
    )

    OVERLAP_OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        OVERLAP_OUTPUT_FILE,
        index=False,
    )

    print(
        f"Overlap rows: "
        f"{len(comparison)}"
    )

    exact = int(
        comparison[
            "exact_match"
        ].sum()
    )

    print(
        f"Exact matches: "
        f"{exact}"
    )

    print(
        f"Different rows: "
        f"{len(comparison) - exact}"
    )

    if len(
        comparison
    ) > 0:

        match_percent = (
            exact
            /
            len(comparison)
            *
            100
        )

        print(
            f"Exact-match percentage: "
            f"{match_percent:.2f}%"
        )

        print(
            f"Mean absolute difference: "
            f"{comparison['difference'].abs().mean():.3f}"
        )

        print(
            f"Maximum absolute difference: "
            f"{comparison['difference'].abs().max():.0f}"
        )

    mismatches = (
        comparison[
            ~comparison[
                "exact_match"
            ]
        ]
    )

    if mismatches.empty:

        print(
            "\n[PASS] "
            "All overlapping 2025 values match."
        )

    else:

        print(
            "\nFirst 20 differences:"
        )

        print(
            mismatches[
                [
                    "week",
                    "district",
                    "cases_hub",
                    "cases_ndcu",
                    "difference",
                ]
            ]
            .head(20)
            .to_string(
                index=False
            )
        )

        print(
            f"\nFull comparison saved to:"
        )

        print(
            f"  {OVERLAP_OUTPUT_FILE}"
        )


# ============================================================
# YEAR COVERAGE
# ============================================================

def show_year_coverage(
    df
):

    print(
        "\n=========================================="
    )

    print(
        "YEAR COVERAGE"
    )

    print(
        "=========================================="
    )

    coverage = (
        df.groupby(
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
        )
    )

    print(
        coverage.to_string()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n=========================================="
    )

    print(
        "DENGUESHIELD AI"
    )

    print(
        "IMPORT DENGUEDATAHUB HISTORY"
    )

    print(
        "=========================================="
    )

    # --------------------------------------------
    # Download
    # --------------------------------------------

    download_dataset()

    # --------------------------------------------
    # Read
    # --------------------------------------------

    raw_df = load_rda()

    # --------------------------------------------
    # Clean
    # --------------------------------------------

    rdhs_df = clean_dataset(
        raw_df
    )

    # --------------------------------------------
    # Validate
    # --------------------------------------------

    validate_rdhs(
        rdhs_df
    )

    # --------------------------------------------
    # Save normalized surveillance-level data
    # --------------------------------------------

    INTERIM_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rdhs_df.to_csv(
        RDHS_OUTPUT_FILE,
        index=False,
    )

    print(
        f"\n[PASS] Saved:"
    )

    print(
        f"       {RDHS_OUTPUT_FILE}"
    )

    # --------------------------------------------
    # Build 25 geographic districts
    # --------------------------------------------

    geographic_df = (
        build_geographic_dataset(
            rdhs_df
        )
    )

    validate_geographic(
        geographic_df
    )

    # --------------------------------------------
    # Save geographic historical data
    # --------------------------------------------

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    geographic_df.to_csv(
        GEOGRAPHIC_OUTPUT_FILE,
        index=False,
    )

    print(
        f"\n[PASS] Saved:"
    )

    print(
        f"       {GEOGRAPHIC_OUTPUT_FILE}"
    )

    # --------------------------------------------
    # Historical coverage
    # --------------------------------------------

    show_year_coverage(
        geographic_df
    )

    # --------------------------------------------
    # Compare 2025 against our NDCU pipeline
    # --------------------------------------------

    # compare_2025_overlap(
    # geographic_df
    # )

    print(
        "\n=========================================="
    )

    print(
        "IMPORT COMPLETE"
    )

    print(
        "=========================================="
    )


if __name__ == "__main__":
    main()