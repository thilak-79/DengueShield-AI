"""
DengueShield AI
NDCU Weekly Dengue PDF Parser

Reads the FIRST PAGE text of each NDCU Weekly Dengue Update.

Each district/RDHS row normally contains six values:

    previous-year previous week
    previous-year current week
    current-year previous week
    current-year current week
    previous-year cumulative
    current-year cumulative

For forecasting we keep:
    - current week's value
    - revised previous week's value

If a later report revises an earlier week,
the newer value wins.

Important:
NDCU uses 26 RDHS rows because Ampara geographic district
is divided into:

    Ampara RDHS
    Kalmunai RDHS

These are later combined into one geographic Ampara district,
giving 25 Sri Lankan districts.
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

OBSERVATIONS_FILE = (
    INTERIM_DIR
    / "dengue_report_observations_rdhs.csv"
)

RDHS_FILE = (
    PROCESSED_DIR
    / "dengue_weekly_rdhs.csv"
)

DISTRICT_FILE = (
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
# BASIC HELPERS
# ============================================================

def extract_report_year_week(filename):
    """
    Extract year and week from filename.

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
            f"Cannot determine year/week "
            f"from filename: {filename}"
        )

    year = int(
        match.group(1)
    )

    week = int(
        match.group(2)
    )

    return year, week


def clean_number(value):
    """
    Convert NDCU PDF values into integers.

    Examples:

        207      -> 207
        71*      -> 71
        1,156    -> 1156
        Nil      -> 0
        Ni       -> 0

    Some NDCU PDFs extract 'Nil' incorrectly as 'Ni'.
    """

    value = str(
        value
    ).strip()

    # --------------------------------------------
    # Handle zero / Nil extraction
    # --------------------------------------------

    if value.lower() in {
        "nil",
        "ni",
    }:
        return 0

    # --------------------------------------------
    # Remove formatting
    # --------------------------------------------

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
    Extract and normalize text from page 1 only.

    Pages 2 and 3 contain MOH, hospital and death tables,
    so they are intentionally ignored.

    This prevents district names from those later tables from
    being mistaken for Table 1 dengue case rows.
    """

    with pdfplumber.open(
        pdf_path
    ) as pdf:

        if len(
            pdf.pages
        ) == 0:

            raise ValueError(
                f"No pages found: "
                f"{pdf_path}"
            )

        text = (
            pdf.pages[0]
            .extract_text()
            or ""
        )

    # --------------------------------------------
    # Replace newlines with spaces
    # --------------------------------------------

    text = text.replace(
        "\n",
        " "
    )

    # --------------------------------------------
    # Repair separated asterisks
    #
    # Some PDFs extract:
    #
    #     2441 * 3265
    #
    # instead of:
    #
    #     2441* 3265
    #
    # This occurred in Week 23.
    # --------------------------------------------

    text = re.sub(
        r"(?<=\d)\s+\*\s*(?=\s|\d)",
        "* ",
        text,
    )

    # --------------------------------------------
    # Collapse repeated whitespace
    # --------------------------------------------

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# DISTRICT ROW PARSER
# ============================================================

# Supported numeric values:
#
# Nil
# Ni
# 207
# 71*
# 1,156
#
VALUE_TOKEN = (
    r"(?:Nil|Ni|[\d,]+\*?)"
)


def find_district_row(
    text,
    district
):
    """
    Find a complete district/RDHS row.

    Example:

        Colombo 99 144 222 207 9453 23161

    Returns:
        [
            previous_year_previous,
            previous_year_current,
            current_year_previous,
            current_year_current,
            previous_year_cumulative,
            current_year_cumulative,
        ]
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

    if len(
        matches
    ) == 0:

        return None

    if len(
        matches
    ) > 1:

        print(
            f"          [WARNING] "
            f"{district}: "
            f"{len(matches)} candidate rows"
        )

    # Full 6-value pattern should correspond to
    # Table 1 on page 1.
    match = matches[0]

    raw_values = [
        match.group(i)
        for i in range(
            1,
            7
        )
    ]

    values = [
        clean_number(
            value
        )
        for value in raw_values
    ]

    return values


# ============================================================
# NATIONAL TOTAL ROW
# ============================================================

def find_total_row(
    text
):
    """
    Extract national total row.

    Example:

        Total* 521 606 1152* 1156 37858 97637
    """

    pattern = (
        rf"\bTotal\*?"
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
# PARSE ONE REPORT
# ============================================================

def parse_report(
    pdf_path
):

    year, report_week = (
        extract_report_year_week(
            pdf_path.name
        )
    )

    print(
        f"\n[PARSING] "
        f"{year} Week "
        f"{report_week:02d}"
    )

    text = (
        get_first_page_text(
            pdf_path
        )
    )

    district_values = {}

    missing = []

    # ========================================================
    # FIND ALL 26 RDHS ROWS
    # ========================================================

    for rdhs in RDHS_UNITS:

        values = (
            find_district_row(
                text,
                rdhs
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
            f"          [WARNING] Missing: "
            f"{', '.join(missing)}"
        )

    observations = []

    # ========================================================
    # CREATE OBSERVATIONS
    # ========================================================

    for (
        rdhs,
        values
    ) in district_values.items():

        # ----------------------------------------------------
        # Six-column mapping
        # ----------------------------------------------------

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

        cumulative_previous_year = (
            values[4]
        )

        cumulative_current_year = (
            values[5]
        )

        # Variables above are kept for clarity and
        # future extensions.

        # ----------------------------------------------------
        # CURRENT WEEK OBSERVATION
        # ----------------------------------------------------

        observations.append(
            {
                "year": year,
                "week": report_week,
                "rdhs": rdhs,
                "cases": (
                    current_year_current
                ),
                "source_report_year": (
                    year
                ),
                "source_report_week": (
                    report_week
                ),
                "observation_type": (
                    "current"
                ),
                "source_pdf": (
                    pdf_path.name
                ),
            }
        )

        # ----------------------------------------------------
        # REVISED PREVIOUS WEEK OBSERVATION
        #
        # Week 1 is special because its previous week
        # belongs to the previous calendar year.
        # ----------------------------------------------------

        if report_week > 1:

            observations.append(
                {
                    "year": year,
                    "week": (
                        report_week - 1
                    ),
                    "rdhs": rdhs,
                    "cases": (
                        current_year_previous
                    ),
                    "source_report_year": (
                        year
                    ),
                    "source_report_week": (
                        report_week
                    ),
                    "observation_type": (
                        "revised_previous"
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
            "National total row not found"
        )

    elif missing:

        print(
            "          [SKIP] "
            "Cannot validate total because "
            "some RDHS rows are missing"
        )

    else:

        # ----------------------------------------------------
        # CURRENT WEEK TOTAL
        # ----------------------------------------------------

        reported_current_total = (
            total_values[3]
        )

        calculated_current_total = sum(
            district_values[
                rdhs
            ][3]
            for rdhs
            in RDHS_UNITS
        )

        if (
            calculated_current_total
            ==
            reported_current_total
        ):

            print(
                f"          [PASS] "
                f"Current total = "
                f"{calculated_current_total}"
            )

        else:

            difference = (
                calculated_current_total
                -
                reported_current_total
            )

            print(
                f"          [SOURCE WARNING] "
                f"Current RDHS sum="
                f"{calculated_current_total}, "
                f"reported total="
                f"{reported_current_total}, "
                f"difference="
                f"{difference:+d}"
            )

        # ----------------------------------------------------
        # PREVIOUS WEEK TOTAL
        # ----------------------------------------------------

        if report_week > 1:

            reported_previous_total = (
                total_values[2]
            )

            calculated_previous_total = sum(
                district_values[
                    rdhs
                ][2]
                for rdhs
                in RDHS_UNITS
            )

            if (
                calculated_previous_total
                ==
                reported_previous_total
            ):

                print(
                    f"          [PASS] "
                    f"Previous total = "
                    f"{calculated_previous_total}"
                )

            else:

                difference = (
                    calculated_previous_total
                    -
                    reported_previous_total
                )

                print(
                    f"          "
                    f"[SOURCE WARNING] "
                    f"RDHS sum="
                    f"{calculated_previous_total}, "
                    f"reported total="
                    f"{reported_previous_total}, "
                    f"difference="
                    f"{difference:+d}"
                )

    return observations


# ============================================================
# CREATE LATEST-REVISION RDHS DATASET
# ============================================================

def build_latest_rdhs_dataset(
    observations_df
):
    """
    A week can appear more than once.

    Example:

        Week 36 appears in Report 36.

        Report 37 then contains a revised
        Week 36 value.

    The newest report wins.
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
# CONVERT 26 RDHS -> 25 GEOGRAPHIC DISTRICTS
# ============================================================

def build_geographic_district_dataset(
    rdhs_df
):
    """
    NDCU divides Ampara geographic district into:

        Ampara RDHS
        Kalmunai RDHS

    For geographic modelling:

        Ampara District =
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

    # Map Kalmunai RDHS into geographic Ampara district.
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
# RDHS VALIDATION
# ============================================================

def validate_rdhs_dataset(
    df
):

    print(
        "\n=========================================="
    )

    print(
        "VALIDATING RDHS DATASET"
    )

    print(
        "=========================================="
    )

    # --------------------------------------------------------
    # No negative case values
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
    # No duplicate week/RDHS combinations
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
    # Every week should contain 26 RDHS units
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

    bad_weeks = (
        counts[
            counts != 26
        ]
    )

    if len(
        bad_weeks
    ) == 0:

        print(
            "[PASS] Every week has "
            "26 RDHS units"
        )

    else:

        print(
            "[WARNING] Weeks with "
            "!= 26 RDHS:"
        )

        print(
            bad_weeks.to_string()
        )


# ============================================================
# DISTRICT VALIDATION
# ============================================================

def validate_district_dataset(
    df
):

    print(
        "\n=========================================="
    )

    print(
        "VALIDATING DISTRICT DATASET"
    )

    print(
        "=========================================="
    )

    # --------------------------------------------------------
    # Every week should contain 25 districts
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

    bad_weeks = (
        counts[
            counts != 25
        ]
    )

    if len(
        bad_weeks
    ) == 0:

        print(
            "[PASS] Every week has "
            "25 geographic districts"
        )

    else:

        print(
            "[WARNING] Weeks with "
            "incorrect district count:"
        )

        print(
            bad_weeks.to_string()
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
    ), (
        "Duplicate district-week rows found"
    )

    print(
        "[PASS] No duplicate "
        "district-week rows"
    )

    # --------------------------------------------------------
    # No negative cases
    # --------------------------------------------------------

    assert (
        df["cases"] >= 0
    ).all(), (
        "Negative district case count found"
    )

    print(
        "[PASS] No negative "
        "district case counts"
    )


# ============================================================
# DATASET SUMMARY
# ============================================================

def print_dataset_summary(
    rdhs_df,
    district_df
):

    print(
        "\n=========================================="
    )

    print(
        "DATASET SUMMARY"
    )

    print(
        "=========================================="
    )

    weeks = sorted(
        district_df[
            "week"
        ].unique()
    )

    print(
        f"Weeks available: "
        f"{weeks[0]} - {weeks[-1]}"
    )

    print(
        f"Number of weeks: "
        f"{len(weeks)}"
    )

    print(
        f"RDHS rows: "
        f"{len(rdhs_df)}"
    )

    print(
        f"District rows: "
        f"{len(district_df)}"
    )

    expected_rdhs_rows = (
        len(weeks)
        *
        26
    )

    expected_district_rows = (
        len(weeks)
        *
        25
    )

    print(
        f"Expected RDHS rows: "
        f"{expected_rdhs_rows}"
    )

    print(
        f"Expected district rows: "
        f"{expected_district_rows}"
    )

    if (
        len(rdhs_df)
        ==
        expected_rdhs_rows
    ):

        print(
            "[PASS] RDHS dataset "
            "row count is complete"
        )

    else:

        print(
            "[WARNING] RDHS dataset "
            "row count is incomplete"
        )

    if (
        len(district_df)
        ==
        expected_district_rows
    ):

        print(
            "[PASS] Geographic district "
            "dataset row count is complete"
        )

    else:

        print(
            "[WARNING] Geographic district "
            "dataset row count is incomplete"
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
        "NDCU PDF PARSER"
    )

    print(
        "=========================================="
    )

    # --------------------------------------------------------
    # Find PDFs
    # --------------------------------------------------------

    pdf_files = sorted(
        REPORT_DIR.glob(
            "ndcu_2026_week_*.pdf"
        )
    )

    if not pdf_files:

        raise FileNotFoundError(
            "No NDCU PDF reports found."
        )

    print(
        f"\nReports found: "
        f"{len(pdf_files)}"
    )

    all_observations = []

    failed_reports = []

    # --------------------------------------------------------
    # Parse reports
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
            "No dengue observations extracted."
        )

    # --------------------------------------------------------
    # Create directories
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
    # Save all observations
    # --------------------------------------------------------

    observations_df = (
        pd.DataFrame(
            all_observations
        )
    )

    observations_df.to_csv(
        OBSERVATIONS_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Latest-revision RDHS dataset
    # --------------------------------------------------------

    rdhs_df = (
        build_latest_rdhs_dataset(
            observations_df
        )
    )

    rdhs_df.to_csv(
        RDHS_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Geographic district dataset
    # --------------------------------------------------------

    district_df = (
        build_geographic_district_dataset(
            rdhs_df
        )
    )

    district_df.to_csv(
        DISTRICT_FILE,
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

    print_dataset_summary(
        rdhs_df,
        district_df
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print(
        "\n=========================================="
    )

    print(
        "PARSING COMPLETE"
    )

    print(
        "=========================================="
    )

    print(
        f"Reports processed: "
        f"{len(pdf_files)}"
    )

    print(
        f"Failed reports: "
        f"{len(failed_reports)}"
    )

    if failed_reports:

        for report in failed_reports:

            print(
                f"    - {report}"
            )

    print(
        f"\nObservation rows: "
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

    print(
        "\nFiles created:"
    )

    print(
        f"  {OBSERVATIONS_FILE}"
    )

    print(
        f"  {RDHS_FILE}"
    )

    print(
        f"  {DISTRICT_FILE}"
    )


if __name__ == "__main__":
    main()