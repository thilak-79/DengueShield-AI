"""
DengueShield AI
Historical NDCU Weekly Dengue PDF Parser

Purpose
-------
Extract BOTH the current report year and comparison year from
NDCU Weekly Dengue Update PDFs.

For example, a 2026 Week 37 report contains:

              2025             2026
            W36  W37         W36  W37

Colombo      99   144         222  207

This script converts that into observations such as:

2025,36,Colombo,99
2025,37,Colombo,144
2026,36,Colombo,222
2026,37,Colombo,207

It also:

- handles "Nil" and malformed "Ni" as zero
- handles NDCU asterisk markers
- keeps the newest observation when weeks overlap
- preserves 26 RDHS units
- combines Ampara + Kalmunai into geographic Ampara
- validates 25 geographic districts per week
- compares reconstructed 2026 data against our already
  validated 2026 dataset

Final main output:

data/processed/dengue_weekly_historical.csv
"""

from pathlib import Path
import re

import pandas as pd
import pdfplumber


# ============================================================
# PATHS
# ============================================================

REPORT_DIR = Path(
    "data/raw/dengue/reports"
)

INTERIM_DIR = Path(
    "data/interim"
)

PROCESSED_DIR = Path(
    "data/processed"
)


HISTORICAL_OBSERVATIONS_FILE = (
    INTERIM_DIR
    / "dengue_historical_observations_rdhs.csv"
)

HISTORICAL_RDHS_FILE = (
    PROCESSED_DIR
    / "dengue_weekly_historical_rdhs.csv"
)

HISTORICAL_DISTRICT_FILE = (
    PROCESSED_DIR
    / "dengue_weekly_historical.csv"
)


# Already validated 2026 dataset.
REFERENCE_2026_FILE = (
    PROCESSED_DIR
    / "dengue_weekly_district.csv"
)


# ============================================================
# NDCU RDHS UNITS
# ============================================================

RDHS_UNITS = [
    "Colombo",
    "Gampaha",
    "Kalutara",
    "Kandy",
    "Matale",
    "Nuwara Eliya",
    "Galle",
    "Hambantota",
    "Matara",
    "Jaffna",
    "Kilinochchi",
    "Mannar",
    "Vavuniya",
    "Mullaitivu",
    "Batticaloa",
    "Ampara",
    "Trincomalee",
    "Kalmunai",
    "Kurunegala",
    "Puttalam",
    "Anuradhapura",
    "Polonnaruwa",
    "Badulla",
    "Monaragala",
    "Ratnapura",
    "Kegalle",
]


# ============================================================
# VALUE PATTERNS
# ============================================================

# District values can appear as:
#
# 207
# 71*
# Nil
# Ni
# 1,156
#
VALUE_TOKEN = (
    r"(?:Nil|Ni|[\d,]+\*?)"
)

# For Total rows we remove * first, so we only need:
#
# 207
# Nil
# Ni
#
TOTAL_VALUE_TOKEN = (
    r"(?:Nil|Ni|[\d,]+)"
)


# ============================================================
# BASIC HELPERS
# ============================================================

def extract_report_year_week(
    filename
):
    """
    Example:

        ndcu_2026_week_37.pdf

    Returns:

        (2026, 37)
    """

    match = re.search(
        r"ndcu_(\d{4})_week_(\d+)",
        filename,
        flags=re.IGNORECASE,
    )

    if not match:

        raise ValueError(
            f"Cannot determine "
            f"year/week from: "
            f"{filename}"
        )

    year = int(
        match.group(1)
    )

    week = int(
        match.group(2)
    )

    return year, week


def clean_number(
    value
):
    """
    Convert NDCU text into integer.

    Examples:

        207      -> 207
        71*      -> 71
        1,156    -> 1156
        Nil      -> 0
        Ni       -> 0

    'Ni' occurs because some PDFs truncate 'Nil'
    during text extraction.
    """

    value = str(
        value
    ).strip()

    if value.lower() in {
        "nil",
        "ni",
    }:
        return 0

    value = (
        value
        .replace(",", "")
        .replace("*", "")
        .strip()
    )

    return int(
        value
    )


def get_first_page_text(
    pdf_path
):
    """
    Extract only page 1.

    NDCU dengue case Table 1 is located on page 1.

    Pages 2 and 3 contain MOH, hospital and death tables.
    Those pages are intentionally ignored because they contain
    the same district names and can confuse extraction.
    """

    with pdfplumber.open(
        pdf_path
    ) as pdf:

        if not pdf.pages:

            raise ValueError(
                f"No pages found in "
                f"{pdf_path}"
            )

        text = (
            pdf.pages[0]
            .extract_text()
            or ""
        )

    # Convert line breaks to spaces.
    text = text.replace(
        "\n",
        " "
    )

    # Repair PDF extraction such as:
    #
    # 2441 * 3265
    #
    # We don't actually need the * marker for numeric values.
    text = re.sub(
        r"(?<=\d)\s+\*\s*(?=\d)",
        " ",
        text,
    )

    # Collapse all repeated spaces.
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# DISTRICT ROW EXTRACTION
# ============================================================

def find_district_row(
    text,
    district
):
    """
    Find six values following a district/RDHS name.

    Example:

        Colombo 99 144 222 207 9453 23161

    Meaning for Report 2026 Week 37:

        2025 Week 36        = 99
        2025 Week 37        = 144
        2026 Week 36        = 222
        2026 Week 37        = 207
        2025 cumulative     = 9453
        2026 cumulative     = 23161
    """

    district_pattern = (
        re.escape(
            district
        )
    )

    pattern = (
        rf"(?<![A-Za-z])"
        rf"{district_pattern}"
        rf"\*?"
        rf"\s+({VALUE_TOKEN})"
        rf"\s+({VALUE_TOKEN})"
        rf"\s+({VALUE_TOKEN})"
        rf"\s+({VALUE_TOKEN})"
        rf"\s+({VALUE_TOKEN})"
        rf"\s+({VALUE_TOKEN})"
    )

    matches = list(
        re.finditer(
            pattern,
            text,
            flags=re.IGNORECASE,
        )
    )

    if not matches:
        return None

    if len(matches) > 1:

        print(
            f"          [WARNING] "
            f"{district}: "
            f"{len(matches)} "
            f"candidate rows"
        )

    match = matches[0]

    values = [
        clean_number(
            match.group(i)
        )
        for i in range(
            1,
            7
        )
    ]

    return values


# ============================================================
# NATIONAL TOTAL EXTRACTION
# ============================================================

def find_total_row(
    text
):
    """
    Parse national Total row.

    We strip all asterisks from a copy of the page text first.
    This makes the function robust to:

        1152*
        2441 *
        2441* 3265

    Example:

        Total 521 606 1152 1156 37858 97637
    """

    clean_text = (
        text.replace(
            "*",
            " "
        )
    )

    clean_text = re.sub(
        r"\s+",
        " ",
        clean_text,
    )

    pattern = (
        rf"\bTotal\b"
        rf"\s+({TOTAL_VALUE_TOKEN})"
        rf"\s+({TOTAL_VALUE_TOKEN})"
        rf"\s+({TOTAL_VALUE_TOKEN})"
        rf"\s+({TOTAL_VALUE_TOKEN})"
        rf"\s+({TOTAL_VALUE_TOKEN})"
        rf"\s+({TOTAL_VALUE_TOKEN})"
    )

    matches = list(
        re.finditer(
            pattern,
            clean_text,
            flags=re.IGNORECASE,
        )
    )

    if not matches:
        return None

    # Use first complete numeric Total row.
    match = matches[0]

    values = [
        clean_number(
            match.group(i)
        )
        for i in range(
            1,
            7
        )
    ]

    return values


# ============================================================
# VALIDATION PRINT HELPER
# ============================================================

def compare_total(
    label,
    calculated,
    reported
):
    """
    Compare district/RDHS sum with NDCU printed total.

    Small differences in source reports are preserved and
    reported as SOURCE WARNING rather than treated as a
    parser failure.
    """

    if calculated == reported:

        print(
            f"          [PASS] "
            f"{label} = "
            f"{calculated}"
        )

        return

    difference = (
        calculated
        - reported
    )

    print(
        f"          [SOURCE WARNING] "
        f"{label} RDHS sum="
        f"{calculated}, "
        f"reported total="
        f"{reported}, "
        f"difference="
        f"{difference:+d}"
    )


# ============================================================
# PARSE ONE REPORT
# ============================================================

def parse_report(
    pdf_path
):
    """
    Convert one PDF into historical observations.

    For reports after Week 1:

        previous year / previous week
        previous year / current week
        current year / previous week
        current year / current week

    Week 1 is special.

    We do NOT create Week 0.
    We only retain the Week 1 value for each year.
    """

    (
        report_year,
        report_week,
    ) = extract_report_year_week(
        pdf_path.name
    )

    comparison_year = (
        report_year - 1
    )

    print(
        f"\n[PARSING] "
        f"{report_year} "
        f"Week {report_week:02d}"
    )

    text = get_first_page_text(
        pdf_path
    )

    district_values = {}

    missing = []

    # ========================================================
    # EXTRACT 26 RDHS ROWS
    # ========================================================

    for rdhs in RDHS_UNITS:

        values = (
            find_district_row(
                text,
                rdhs,
            )
        )

        if values is None:

            missing.append(
                rdhs
            )

            continue

        district_values[
            rdhs
        ] = values

    print(
        f"          RDHS rows found: "
        f"{len(district_values)}/26"
    )

    if missing:

        print(
            f"          [WARNING] "
            f"Missing: "
            f"{', '.join(missing)}"
        )

    observations = []

    # ========================================================
    # BUILD HISTORICAL OBSERVATIONS
    # ========================================================

    for (
        rdhs,
        values,
    ) in district_values.items():

        # --------------------------------------------
        # Column mapping
        # --------------------------------------------

        previous_year_previous = (
            values[0]
        )

        previous_year_current = (
            values[1]
        )

        current_year_previous = (
            values[2]
        )

        current_year_current = (
            values[3]
        )

        # These are available for future use,
        # although not needed for weekly forecasting.
        cumulative_previous_year = (
            values[4]
        )

        cumulative_current_year = (
            values[5]
        )

        # --------------------------------------------
        # PREVIOUS YEAR - CURRENT WEEK
        #
        # Example:
        # Report 2026 Week 37
        # -> 2025 Week 37
        # --------------------------------------------

        observations.append(
            {
                "year": (
                    comparison_year
                ),
                "week": (
                    report_week
                ),
                "rdhs": rdhs,
                "cases": (
                    previous_year_current
                ),
                "source_report_year": (
                    report_year
                ),
                "source_report_week": (
                    report_week
                ),
                "observation_type": (
                    "comparison_year_current"
                ),
                "source_pdf": (
                    pdf_path.name
                ),
            }
        )

        # --------------------------------------------
        # CURRENT YEAR - CURRENT WEEK
        #
        # Example:
        # Report 2026 Week 37
        # -> 2026 Week 37
        # --------------------------------------------

        observations.append(
            {
                "year": (
                    report_year
                ),
                "week": (
                    report_week
                ),
                "rdhs": rdhs,
                "cases": (
                    current_year_current
                ),
                "source_report_year": (
                    report_year
                ),
                "source_report_week": (
                    report_week
                ),
                "observation_type": (
                    "current_year_current"
                ),
                "source_pdf": (
                    pdf_path.name
                ),
            }
        )

        # --------------------------------------------
        # PREVIOUS-WEEK VALUES
        #
        # Do not create Week 0 from Week 1.
        # --------------------------------------------

        if report_week > 1:

            # Previous year / previous week
            observations.append(
                {
                    "year": (
                        comparison_year
                    ),
                    "week": (
                        report_week - 1
                    ),
                    "rdhs": rdhs,
                    "cases": (
                        previous_year_previous
                    ),
                    "source_report_year": (
                        report_year
                    ),
                    "source_report_week": (
                        report_week
                    ),
                    "observation_type": (
                        "comparison_year_previous"
                    ),
                    "source_pdf": (
                        pdf_path.name
                    ),
                }
            )

            # Current year / revised previous week
            observations.append(
                {
                    "year": (
                        report_year
                    ),
                    "week": (
                        report_week - 1
                    ),
                    "rdhs": rdhs,
                    "cases": (
                        current_year_previous
                    ),
                    "source_report_year": (
                        report_year
                    ),
                    "source_report_week": (
                        report_week
                    ),
                    "observation_type": (
                        "current_year_revised_previous"
                    ),
                    "source_pdf": (
                        pdf_path.name
                    ),
                }
            )

    # ========================================================
    # NATIONAL TOTAL VALIDATION
    # ========================================================

    total_values = (
        find_total_row(
            text
        )
    )

    if total_values is None:

        print(
            "          [WARNING] "
            "National total row "
            "not found"
        )

        return observations

    if missing:

        print(
            "          [SKIP] "
            "National total validation "
            "because RDHS rows are missing"
        )

        return observations

    # Total column mapping:
    #
    # 0 = previous year previous week
    # 1 = previous year current week
    # 2 = current year previous week
    # 3 = current year current week

    # --------------------------------------------------------
    # Comparison year current week
    # --------------------------------------------------------

    previous_year_current_sum = sum(
        district_values[
            rdhs
        ][1]
        for rdhs
        in RDHS_UNITS
    )

    compare_total(
        (
            f"{comparison_year} "
            f"Week {report_week}"
        ),
        previous_year_current_sum,
        total_values[1],
    )

    # --------------------------------------------------------
    # Current report year current week
    # --------------------------------------------------------

    current_year_current_sum = sum(
        district_values[
            rdhs
        ][3]
        for rdhs
        in RDHS_UNITS
    )

    compare_total(
        (
            f"{report_year} "
            f"Week {report_week}"
        ),
        current_year_current_sum,
        total_values[3],
    )

    # --------------------------------------------------------
    # Previous-week validation
    # --------------------------------------------------------

    if report_week > 1:

        previous_year_previous_sum = sum(
            district_values[
                rdhs
            ][0]
            for rdhs
            in RDHS_UNITS
        )

        compare_total(
            (
                f"{comparison_year} "
                f"Week "
                f"{report_week - 1}"
            ),
            previous_year_previous_sum,
            total_values[0],
        )

        current_year_previous_sum = sum(
            district_values[
                rdhs
            ][2]
            for rdhs
            in RDHS_UNITS
        )

        compare_total(
            (
                f"{report_year} "
                f"Week "
                f"{report_week - 1}"
            ),
            current_year_previous_sum,
            total_values[2],
        )

    return observations


# ============================================================
# LATEST-REVISION DATASET
# ============================================================

def build_latest_rdhs_dataset(
    observations_df
):
    """
    The same year/week/RDHS may occur in multiple reports.

    Example:

        Report 36:
            2025 Week 36

        Report 37:
            2025 Week 36 again

    Keep the observation from the newest available report.
    """

    df = (
        observations_df
        .copy()
    )

    df = df.sort_values(
        [
            "year",
            "week",
            "rdhs",
            "source_report_year",
            "source_report_week",
        ]
    )

    latest = (
        df.drop_duplicates(
            subset=[
                "year",
                "week",
                "rdhs",
            ],
            keep="last",
        )
    )

    latest = (
        latest
        .sort_values(
            [
                "year",
                "week",
                "rdhs",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return latest


# ============================================================
# 26 RDHS -> 25 GEOGRAPHIC DISTRICTS
# ============================================================

def build_geographic_dataset(
    rdhs_df
):
    """
    NDCU divides geographic Ampara into:

        Ampara RDHS
        Kalmunai RDHS

    For district-level ML:

        Ampara =
        Ampara RDHS + Kalmunai RDHS
    """

    df = (
        rdhs_df
        .copy()
    )

    df[
        "district"
    ] = df[
        "rdhs"
    ]

    df.loc[
        df["rdhs"] == "Kalmunai",
        "district"
    ] = "Ampara"

    district_df = (
        df.groupby(
            [
                "year",
                "week",
                "district",
            ],
            as_index=False,
        )[
            "cases"
        ]
        .sum()
    )

    district_df = (
        district_df
        .sort_values(
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

    return district_df


# ============================================================
# BASIC RDHS VALIDATION
# ============================================================

def validate_rdhs_dataset(
    df
):
    """
    Validate historical RDHS-level dataset.
    """

    print(
        "\n=========================================="
    )

    print(
        "VALIDATING HISTORICAL RDHS DATASET"
    )

    print(
        "=========================================="
    )

    # --------------------------------------------------------
    # No negative values
    # --------------------------------------------------------

    assert (
        df["cases"] >= 0
    ).all(), (
        "Negative case count found"
    )

    print(
        "[PASS] No negative case counts"
    )

    # --------------------------------------------------------
    # No duplicates
    # --------------------------------------------------------

    duplicates = (
        df.duplicated(
            subset=[
                "year",
                "week",
                "rdhs",
            ]
        )
    )

    assert not (
        duplicates.any()
    ), (
        "Duplicate RDHS-week rows found"
    )

    print(
        "[PASS] No duplicate "
        "RDHS-week rows"
    )

    # --------------------------------------------------------
    # Every year/week = 26 RDHS units
    # --------------------------------------------------------

    counts = (
        df.groupby(
            [
                "year",
                "week",
            ]
        )[
            "rdhs"
        ]
        .nunique()
    )

    bad = (
        counts[
            counts != 26
        ]
    )

    if bad.empty:

        print(
            "[PASS] Every year/week "
            "has 26 RDHS units"
        )

    else:

        print(
            "[WARNING] Incomplete "
            "year/week RDHS groups:"
        )

        print(
            bad.to_string()
        )


# ============================================================
# DISTRICT VALIDATION
# ============================================================

def validate_district_dataset(
    df
):
    """
    Validate 25-district dataset.
    """

    print(
        "\n=========================================="
    )

    print(
        "VALIDATING HISTORICAL DISTRICT DATASET"
    )

    print(
        "=========================================="
    )

    # --------------------------------------------------------
    # No negatives
    # --------------------------------------------------------

    assert (
        df["cases"] >= 0
    ).all()

    print(
        "[PASS] No negative "
        "district case counts"
    )

    # --------------------------------------------------------
    # No duplicates
    # --------------------------------------------------------

    duplicates = (
        df.duplicated(
            subset=[
                "year",
                "week",
                "district",
            ]
        )
    )

    assert not (
        duplicates.any()
    )

    print(
        "[PASS] No duplicate "
        "district-week rows"
    )

    # --------------------------------------------------------
    # Exactly 25 districts
    # --------------------------------------------------------

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

    bad = (
        counts[
            counts != 25
        ]
    )

    if bad.empty:

        print(
            "[PASS] Every year/week "
            "has 25 geographic districts"
        )

    else:

        print(
            "[WARNING] Incomplete "
            "district groups:"
        )

        print(
            bad.to_string()
        )


# ============================================================
# VALIDATE YEARS + WEEKS
# ============================================================

def validate_year_coverage(
    df
):
    """
    Show and validate week coverage for each year.
    """

    print(
        "\n=========================================="
    )

    print(
        "YEAR COVERAGE"
    )

    print(
        "=========================================="
    )

    years = sorted(
        df[
            "year"
        ].unique()
    )

    print(
        f"Years available: "
        f"{years}"
    )

    for year in years:

        year_df = (
            df[
                df["year"]
                == year
            ]
        )

        weeks = sorted(
            year_df[
                "week"
            ].unique()
        )

        print(
            f"\n{year}:"
        )

        print(
            f"  Weeks: "
            f"{weeks[0]} - "
            f"{weeks[-1]}"
        )

        print(
            f"  Number of weeks: "
            f"{len(weeks)}"
        )

        print(
            f"  District rows: "
            f"{len(year_df)}"
        )

        expected_rows = (
            len(weeks)
            *
            25
        )

        print(
            f"  Expected rows: "
            f"{expected_rows}"
        )

        if (
            len(year_df)
            ==
            expected_rows
        ):

            print(
                "  [PASS] "
                "Year row count complete"
            )

        else:

            print(
                "  [WARNING] "
                "Year row count incomplete"
            )

        # Check contiguous week numbers.
        expected_weeks = set(
            range(
                min(weeks),
                max(weeks) + 1
            )
        )

        missing_weeks = sorted(
            expected_weeks
            -
            set(weeks)
        )

        if not missing_weeks:

            print(
                "  [PASS] "
                "No missing weeks"
            )

        else:

            print(
                f"  [WARNING] "
                f"Missing weeks: "
                f"{missing_weeks}"
            )


# ============================================================
# COMPARE AGAINST VALIDATED 2026 DATA
# ============================================================

def compare_with_existing_2026(
    historical_df
):
    """
    The historical parser should reconstruct 2026 exactly.

    Compare against:

        data/processed/dengue_weekly_district.csv

    which came from our already validated 2026 parser.
    """

    print(
        "\n=========================================="
    )

    print(
        "VERIFYING AGAINST EXISTING 2026 DATA"
    )

    print(
        "=========================================="
    )

    if not (
        REFERENCE_2026_FILE.exists()
    ):

        print(
            "[WARNING] Existing "
            "2026 reference file not found."
        )

        return

    reference = (
        pd.read_csv(
            REFERENCE_2026_FILE
        )
    )

    historical_2026 = (
        historical_df[
            historical_df[
                "year"
            ] == 2026
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

    reference = (
        reference[
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
        historical_2026.merge(
            reference,
            on=[
                "year",
                "week",
                "district",
            ],
            how="outer",
            suffixes=(
                "_historical",
                "_reference",
            ),
            indicator=True,
        )
    )

    comparison[
        "matches"
    ] = (
        comparison[
            "cases_historical"
        ]
        ==
        comparison[
            "cases_reference"
        ]
    )

    problem_rows = (
        comparison[
            (
                comparison["_merge"]
                != "both"
            )
            |
            (
                ~comparison["matches"]
                .fillna(False)
            )
        ]
    )

    if problem_rows.empty:

        print(
            "[PASS] Historical parser "
            "reproduces validated 2026 "
            "dataset exactly"
        )

        print(
            f"[PASS] Compared "
            f"{len(comparison)} "
            f"district-week rows"
        )

    else:

        print(
            "[FAIL] Historical parser "
            "does not exactly reproduce "
            "the 2026 reference dataset"
        )

        print(
            problem_rows.head(
                30
            ).to_string(
                index=False
            )
        )

        raise AssertionError(
            "Historical 2026 data "
            "does not match reference."
        )


# ============================================================
# DATASET SUMMARY
# ============================================================

def print_summary(
    observations_df,
    rdhs_df,
    district_df,
    failed_reports,
):
    """
    Print final parser statistics.
    """

    print(
        "\n=========================================="
    )

    print(
        "HISTORICAL DATASET SUMMARY"
    )

    print(
        "=========================================="
    )

    print(
        f"Observation rows: "
        f"{len(observations_df)}"
    )

    print(
        f"Latest RDHS rows: "
        f"{len(rdhs_df)}"
    )

    print(
        f"Geographic district rows: "
        f"{len(district_df)}"
    )

    years = sorted(
        district_df[
            "year"
        ].unique()
    )

    print(
        f"Years: "
        f"{years}"
    )

    for year in years:

        subset = (
            district_df[
                district_df[
                    "year"
                ] == year
            ]
        )

        print(
            f"{year}: "
            f"{len(subset)} rows, "
            f"{subset['week'].nunique()} "
            f"weeks"
        )

    if failed_reports:

        print(
            "\nFailed reports:"
        )

        for report in failed_reports:

            print(
                f"  - {report}"
            )

    else:

        print(
            "\n[PASS] "
            "No report parsing failures"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n=========================================="
    )

    print(
        "DENGUESHIELD AI - "
        "HISTORICAL NDCU PARSER"
    )

    print(
        "=========================================="
    )

    # --------------------------------------------------------
    # Find all downloaded 2026 reports
    # --------------------------------------------------------

    pdf_files = sorted(
        REPORT_DIR.glob(
            "ndcu_2026_week_*.pdf"
        )
    )

    if not pdf_files:

        raise FileNotFoundError(
            "No NDCU 2026 PDF "
            "reports found."
        )

    print(
        f"\nReports found: "
        f"{len(pdf_files)}"
    )

    all_observations = []

    failed_reports = []

    # --------------------------------------------------------
    # Parse every PDF
    # --------------------------------------------------------

    for pdf_path in pdf_files:

        try:

            rows = (
                parse_report(
                    pdf_path
                )
            )

            all_observations.extend(
                rows
            )

        except Exception as error:

            print(
                f"\n[ERROR] "
                f"{pdf_path.name}: "
                f"{error}"
            )

            failed_reports.append(
                pdf_path.name
            )

    if not all_observations:

        raise RuntimeError(
            "No historical observations "
            "were extracted."
        )

    # --------------------------------------------------------
    # Create folders
    # --------------------------------------------------------

    INTERIM_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Raw historical observations
    # --------------------------------------------------------

    observations_df = (
        pd.DataFrame(
            all_observations
        )
    )

    observations_df = (
        observations_df
        .sort_values(
            [
                "year",
                "week",
                "rdhs",
                "source_report_week",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    observations_df.to_csv(
        HISTORICAL_OBSERVATIONS_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Latest revision per year/week/RDHS
    # --------------------------------------------------------

    rdhs_df = (
        build_latest_rdhs_dataset(
            observations_df
        )
    )

    rdhs_df.to_csv(
        HISTORICAL_RDHS_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # 25 geographic districts
    # --------------------------------------------------------

    district_df = (
        build_geographic_dataset(
            rdhs_df
        )
    )

    # Final ML-friendly file:
    #
    # year,week,district,cases
    district_df = (
        district_df[
            [
                "year",
                "week",
                "district",
                "cases",
            ]
        ]
    )

    district_df.to_csv(
        HISTORICAL_DISTRICT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validate_rdhs_dataset(
        rdhs_df
    )

    validate_district_dataset(
        district_df
    )

    validate_year_coverage(
        district_df
    )

    compare_with_existing_2026(
        district_df
    )

    print_summary(
        observations_df,
        rdhs_df,
        district_df,
        failed_reports,
    )

    # --------------------------------------------------------
    # Expected result for 37 weeks x 2 years
    # --------------------------------------------------------

    expected_rdhs_rows = (
        37
        *
        26
        *
        2
    )

    expected_district_rows = (
        37
        *
        25
        *
        2
    )

    print(
        "\n=========================================="
    )

    print(
        "EXPECTED SIZE CHECK"
    )

    print(
        "=========================================="
    )

    print(
        f"Expected RDHS rows: "
        f"{expected_rdhs_rows}"
    )

    print(
        f"Actual RDHS rows: "
        f"{len(rdhs_df)}"
    )

    print(
        f"Expected district rows: "
        f"{expected_district_rows}"
    )

    print(
        f"Actual district rows: "
        f"{len(district_df)}"
    )

    if (
        len(rdhs_df)
        ==
        expected_rdhs_rows
    ):

        print(
            "[PASS] Historical RDHS "
            "dataset complete"
        )

    else:

        print(
            "[WARNING] Historical RDHS "
            "dataset is not complete"
        )

    if (
        len(district_df)
        ==
        expected_district_rows
    ):

        print(
            "[PASS] Historical district "
            "dataset complete"
        )

    else:

        print(
            "[WARNING] Historical district "
            "dataset is not complete"
        )

    # --------------------------------------------------------
    # Final files
    # --------------------------------------------------------

    print(
        "\nFiles created:"
    )

    print(
        f"  "
        f"{HISTORICAL_OBSERVATIONS_FILE}"
    )

    print(
        f"  "
        f"{HISTORICAL_RDHS_FILE}"
    )

    print(
        f"  "
        f"{HISTORICAL_DISTRICT_FILE}"
    )

    print(
        "\n=========================================="
    )

    print(
        "HISTORICAL PARSING COMPLETE"
    )

    print(
        "=========================================="
    )


if __name__ == "__main__":
    main()