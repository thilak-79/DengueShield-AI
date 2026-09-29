"""
DengueShield AI
Current 2026 Data Readiness Audit

Checks whether current NDCU surveillance data can be used
to construct an experimental operational forecast.

IMPORTANT
---------
Historical model training data uses WER reporting semantics.
Current surveillance data uses NDCU reporting semantics.

Therefore passing this audit means:
    technically ready for experimental inference

It does NOT mean:
    clinically / operationally validated live forecasting.
"""

from pathlib import Path
from datetime import datetime, timedelta, timezone
import json
import re
import sys

import joblib
import numpy as np
import pandas as pd
import pdfplumber


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

CASE_FILE = Path(
    "data/processed/dengue_weekly_historical.csv"
)

REPORT_DIR = Path(
    "data/raw/dengue/reports"
)

REFERENCE_FEATURE_FILE = Path(
    "data/processed/ml_features_2015_2025.csv"
)

MODEL_FILE = Path(
    "models/random_forest/random_forest_1w.joblib"
)

OUTPUT_DIR = Path(
    "data/processed/operational"
)

AUDIT_FILE = (
    OUTPUT_DIR
    / "current_data_audit.csv"
)

CALENDAR_FILE = (
    OUTPUT_DIR
    / "current_week_calendar.csv"
)

METADATA_FILE = (
    OUTPUT_DIR
    / "current_data_metadata.json"
)

SOURCE_BRIDGE_FILE = (
    OUTPUT_DIR
    / "source_bridge_2025.csv"
)


EXPECTED_DISTRICTS = 25
REQUIRED_HISTORY_WEEKS = 5


# ============================================================
# HELPERS
# ============================================================

def add_check(
    rows,
    check,
    status,
    details,
):
    rows.append(
        {
            "check":
            check,

            "status":
            status,

            "details":
            str(
                details
            ),
        }
    )


def find_report(
    year,
    week,
):
    matches = []

    for path in REPORT_DIR.glob(
        "*.pdf"
    ):
        name = path.stem

        pattern = (
            rf"{year}.*week[_\-\s]*0?{week}"
            rf"(?:\D|$)"
        )

        if re.search(
            pattern,
            name,
            flags=re.IGNORECASE,
        ):
            matches.append(
                path
            )

    if not matches:
        return None

    return sorted(
        matches
    )[-1]


def parse_report_window(
    path,
):
    """
    Parse strings such as:

    Week 37 (07th – 13th September 2026)

    or:

    Week 36 (31st August – 06th September 2026)
    """

    with pdfplumber.open(
        path
    ) as pdf:

        text = (
            pdf.pages[
                0
            ]
            .extract_text()
            or
            ""
        )

    text = (
        text
        .replace(
            "–",
            "-"
        )
        .replace(
            "—",
            "-"
        )
        .replace(
            "−",
            "-"
        )
        .replace(
            "ΓÇô",
            "-"
        )
    )

    title_match = re.search(
        r"Week\s+(\d+)\s*"
        r"\(([^)]{5,100})\)",
        text,
        flags=re.IGNORECASE,
    )

    if not title_match:
        raise ValueError(
            f"Cannot find week date range "
            f"in {path.name}"
        )

    week = int(
        title_match.group(
            1
        )
    )

    date_text = (
        title_match.group(
            2
        )
        .strip()
    )

    pattern = (
        r"(\d{1,2})"
        r"(?:st|nd|rd|th)?"
        r"(?:\s+([A-Za-z]+))?"
        r"\s*-\s*"
        r"(\d{1,2})"
        r"(?:st|nd|rd|th)?"
        r"\s+([A-Za-z]+)"
        r"\s+(\d{4})"
    )

    date_match = re.search(
        pattern,
        date_text,
        flags=re.IGNORECASE,
    )

    if not date_match:
        raise ValueError(
            f"Cannot parse '{date_text}' "
            f"from {path.name}"
        )

    start_day = int(
        date_match.group(
            1
        )
    )

    start_month = (
        date_match.group(
            2
        )
    )

    end_day = int(
        date_match.group(
            3
        )
    )

    end_month = (
        date_match.group(
            4
        )
    )

    year = int(
        date_match.group(
            5
        )
    )

    if start_month is None:
        start_month = (
            end_month
        )

    start_date = datetime.strptime(
        (
            f"{start_day} "
            f"{start_month} "
            f"{year}"
        ),
        "%d %B %Y",
    ).date()

    end_date = datetime.strptime(
        (
            f"{end_day} "
            f"{end_month} "
            f"{year}"
        ),
        "%d %B %Y",
    ).date()

    return {
        "week":
        week,

        "start_date":
        start_date,

        "end_date":
        end_date,

        "report_file":
        path.name,
    }


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
        "CURRENT DATA READINESS AUDIT"
    )
    print(
        "=========================================="
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checks = []

    # --------------------------------------------------------
    # INPUT FILES
    # --------------------------------------------------------

    for path, label in [
        (
            CASE_FILE,
            "NDCU case dataset",
        ),
        (
            REFERENCE_FEATURE_FILE,
            "historical feature dataset",
        ),
        (
            MODEL_FILE,
            "1-week RF model",
        ),
    ]:

        if path.exists():

            add_check(
                checks,
                label,
                "PASS",
                path,
            )

        else:

            add_check(
                checks,
                label,
                "FAIL",
                f"Missing: {path}",
            )

    if any(
        row[
            "status"
        ]
        ==
        "FAIL"
        for row
        in checks
    ):

        pd.DataFrame(
            checks
        ).to_csv(
            AUDIT_FILE,
            index=False,
        )

        raise SystemExit(
            "Required files are missing."
        )

    # --------------------------------------------------------
    # CASE DATA
    # --------------------------------------------------------

    cases = pd.read_csv(
        CASE_FILE
    )

    required_columns = {
        "year",
        "week",
        "district",
        "cases",
    }

    missing = (
        required_columns
        -
        set(
            cases.columns
        )
    )

    if missing:
        raise ValueError(
            f"Missing case columns: "
            f"{sorted(missing)}"
        )

    latest_year = int(
        cases[
            "year"
        ]
        .max()
    )

    latest_week = int(
        cases.loc[
            cases[
                "year"
            ]
            ==
            latest_year,
            "week",
        ]
        .max()
    )

    print(
        f"[INFO] Latest source: "
        f"{latest_year} week "
        f"{latest_week}"
    )

    current = (
        cases[
            cases[
                "year"
            ]
            ==
            latest_year
        ]
        .copy()
    )

    latest = (
        current[
            current[
                "week"
            ]
            ==
            latest_week
        ]
    )

    district_count = (
        latest[
            "district"
        ]
        .nunique()
    )

    add_check(
        checks,
        "latest district count",
        (
            "PASS"
            if district_count
            ==
            EXPECTED_DISTRICTS
            else
            "FAIL"
        ),
        district_count,
    )

    duplicate_count = (
        current.duplicated(
            [
                "week",
                "district",
            ]
        )
        .sum()
    )

    add_check(
        checks,
        "duplicate district-week rows",
        (
            "PASS"
            if duplicate_count
            ==
            0
            else
            "FAIL"
        ),
        duplicate_count,
    )

    negative_count = (
        current[
            "cases"
        ]
        .lt(
            0
        )
        .sum()
    )

    add_check(
        checks,
        "negative cases",
        (
            "PASS"
            if negative_count
            ==
            0
            else
            "FAIL"
        ),
        negative_count,
    )

    recent_weeks = list(
        range(
            latest_week
            -
            REQUIRED_HISTORY_WEEKS
            +
            1,
            latest_week
            +
            1,
        )
    )

    recent = (
        current[
            current[
                "week"
            ]
            .isin(
                recent_weeks
            )
        ]
    )

    recent_counts = (
        recent.groupby(
            "week"
        )[
            "district"
        ]
        .nunique()
    )

    complete_recent = (
        len(
            recent_counts
        )
        ==
        REQUIRED_HISTORY_WEEKS
        and
        (
            recent_counts
            ==
            EXPECTED_DISTRICTS
        )
        .all()
    )

    add_check(
        checks,
        "latest 5 weeks complete",
        (
            "PASS"
            if complete_recent
            else
            "FAIL"
        ),
        (
            recent_counts
            .to_dict()
        ),
    )

    latest_total = int(
        latest[
            "cases"
        ]
        .sum()
    )

    add_check(
        checks,
        "latest national case total",
        "PASS",
        latest_total,
    )

    # --------------------------------------------------------
    # EXACT WEEK WINDOWS FROM PDFs
    # --------------------------------------------------------

    calendar_rows = []

    for week in recent_weeks:

        report = find_report(
            latest_year,
            week,
        )

        if report is None:

            add_check(
                checks,
                f"report week {week}",
                "FAIL",
                "PDF not found",
            )

            continue

        try:

            parsed = (
                parse_report_window(
                    report
                )
            )

            calendar_rows.append(
                parsed
            )

            add_check(
                checks,
                f"report week {week}",
                "PASS",
                (
                    f"{parsed['start_date']} "
                    f"to "
                    f"{parsed['end_date']}"
                ),
            )

        except Exception as exc:

            add_check(
                checks,
                f"report week {week}",
                "FAIL",
                exc,
            )

    calendar = pd.DataFrame(
        calendar_rows
    )

    calendar_ok = False

    if len(
        calendar
    ) == REQUIRED_HISTORY_WEEKS:

        calendar = (
            calendar.sort_values(
                "week"
            )
            .reset_index(
                drop=True
            )
        )

        durations = (
            pd.to_datetime(
                calendar[
                    "end_date"
                ]
            )
            -
            pd.to_datetime(
                calendar[
                    "start_date"
                ]
            )
        ).dt.days

        duration_ok = (
            durations
            ==
            6
        ).all()

        continuity_ok = True

        for i in range(
            1,
            len(
                calendar
            ),
        ):

            previous_end = (
                pd.Timestamp(
                    calendar.loc[
                        i - 1,
                        "end_date",
                    ]
                )
            )

            current_start = (
                pd.Timestamp(
                    calendar.loc[
                        i,
                        "start_date",
                    ]
                )
            )

            if (
                current_start
                !=
                previous_end
                +
                pd.Timedelta(
                    days=1
                )
            ):

                continuity_ok = False

        calendar_ok = (
            duration_ok
            and
            continuity_ok
        )

        add_check(
            checks,
            "week windows are 7 days",
            (
                "PASS"
                if duration_ok
                else
                "FAIL"
            ),
            durations.tolist(),
        )

        add_check(
            checks,
            "recent week windows contiguous",
            (
                "PASS"
                if continuity_ok
                else
                "FAIL"
            ),
            (
                "Each week begins one day "
                "after previous week ends."
            ),
        )

        calendar.to_csv(
            CALENDAR_FILE,
            index=False,
        )

    # --------------------------------------------------------
    # MODEL COMPATIBILITY
    # --------------------------------------------------------

    pipeline = joblib.load(
        MODEL_FILE
    )

    if hasattr(
        pipeline,
        "feature_names_in_",
    ):

        model_features = list(
            pipeline.feature_names_in_
        )

    else:

        model_features = []

    add_check(
        checks,
        "model raw input feature count",
        (
            "PASS"
            if len(
                model_features
            )
            >=
            1
            else
            "FAIL"
        ),
        len(
            model_features
        ),
    )

    reference = pd.read_csv(
        REFERENCE_FEATURE_FILE
    )

    missing_model_features = (
        set(
            model_features
        )
        -
        set(
            reference.columns
        )
    )

    add_check(
        checks,
        "model features available in reference",
        (
            "PASS"
            if not
            missing_model_features
            else
            "FAIL"
        ),
        sorted(
            missing_model_features
        ),
    )

    # --------------------------------------------------------
    # SEASONAL ENCODING VALIDATION
    #
    # EXACTLY reproduce src/features/build_features.py:
    #
    # midpoint =
    #     start_date
    #     +
    #     (end_date - start_date) / 2
    #
    # day_of_year = midpoint.dayofyear
    #
    # week_sin =
    #     sin(2*pi*day_of_year / 365.25)
    #
    # week_cos =
    #     cos(2*pi*day_of_year / 365.25)
    #
    # Despite the feature names week_sin/week_cos,
    # this uses the actual surveillance interval dates,
    # not WER or ISO week numbers.
    # --------------------------------------------------------

    seasonal_ready = False

    required_season_columns = {
        "start_date",
        "end_date",
        "week_sin",
        "week_cos",
    }

    if required_season_columns.issubset(
        reference.columns
    ):

        seasonal_reference = (
            reference[
                [
                    "start_date",
                    "end_date",
                    "week_sin",
                    "week_cos",
                ]
            ]
            .dropna()
            .copy()
        )

        seasonal_reference[
            "start_date"
        ] = pd.to_datetime(
            seasonal_reference[
                "start_date"
            ]
        )

        seasonal_reference[
            "end_date"
        ] = pd.to_datetime(
            seasonal_reference[
                "end_date"
            ]
        )

        midpoint = (
            seasonal_reference[
                "start_date"
            ]
            +
            (
                seasonal_reference[
                    "end_date"
                ]
                -
                seasonal_reference[
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

        sin_error = np.abs(
            seasonal_reference[
                "week_sin"
            ]
            .to_numpy(
                dtype=float
            )
            -
            expected_sin.to_numpy(
                dtype=float
            )
        )

        cos_error = np.abs(
            seasonal_reference[
                "week_cos"
            ]
            .to_numpy(
                dtype=float
            )
            -
            expected_cos.to_numpy(
                dtype=float
            )
        )

        max_sin_error = float(
            np.max(
                sin_error
            )
        )

        max_cos_error = float(
            np.max(
                cos_error
            )
        )

        seasonal_ready = (
            max_sin_error
            <
            1e-10
            and
            max_cos_error
            <
            1e-10
        )

        add_check(
            checks,
            "seasonality formula reproduction",
            (
                "PASS"
                if seasonal_ready
                else
                "FAIL"
            ),
            (
                "exact surveillance-midpoint "
                "day-of-year encoding; "
                f"max_sin_error="
                f"{max_sin_error:.12g}; "
                f"max_cos_error="
                f"{max_cos_error:.12g}"
            ),
        )

    else:

        add_check(
            checks,
            "seasonality formula reproduction",
            "FAIL",
            (
                "Required columns missing: "
                "start_date, end_date, "
                "week_sin, week_cos"
            ),
        )

    # --------------------------------------------------------
    # OPTIONAL SOURCE-BRIDGE DIAGNOSTIC
    # --------------------------------------------------------

    if {
        "year",
        "week",
        "district",
        "cases_current",
    }.issubset(
        reference.columns
    ):

        ndcu_2025 = (
            cases[
                cases[
                    "year"
                ]
                ==
                2025
            ][
                [
                    "week",
                    "district",
                    "cases",
                ]
            ]
            .rename(
                columns={
                    "cases":
                    "ndcu_cases"
                }
            )
        )

        wer_2025 = (
            reference[
                reference[
                    "year"
                ]
                ==
                2025
            ][
                [
                    "week",
                    "district",
                    "cases_current",
                ]
            ]
            .rename(
                columns={
                    "cases_current":
                    "wer_cases"
                }
            )
        )

        bridge = (
            ndcu_2025.merge(
                wer_2025,
                on=[
                    "week",
                    "district",
                ],
                how="inner",
            )
        )

        if not bridge.empty:

            bridge[
                "absolute_difference"
            ] = (
                bridge[
                    "ndcu_cases"
                ]
                -
                bridge[
                    "wer_cases"
                ]
            ).abs()

            bridge.to_csv(
                SOURCE_BRIDGE_FILE,
                index=False,
            )

            mae = float(
                bridge[
                    "absolute_difference"
                ]
                .mean()
            )

            correlation = float(
                bridge[
                    [
                        "ndcu_cases",
                        "wer_cases",
                    ]
                ]
                .corr()
                .iloc[
                    0,
                    1,
                ]
            )

            add_check(
                checks,
                "NDCU vs WER source bridge",
                "WARN",
                (
                    f"{len(bridge)} overlapping "
                    f"district-weeks; "
                    f"diagnostic MAE={mae:.3f}; "
                    f"correlation={correlation:.3f}. "
                    f"Week semantics differ, so "
                    f"this is diagnostic only."
                ),
            )

    # --------------------------------------------------------
    # SOURCE-SEMANTIC WARNING
    # --------------------------------------------------------

    add_check(
        checks,
        "surveillance source semantics",
        "WARN",
        (
            "Operational cases come from NDCU, "
            "while the historical ML model was "
            "developed using WER-aligned data. "
            "Do not treat inference as a "
            "validated official live forecast."
        ),
    )

    audit_df = pd.DataFrame(
        checks
    )

    audit_df.to_csv(
        AUDIT_FILE,
        index=False,
    )

    mandatory_failures = (
        audit_df[
            audit_df[
                "status"
            ]
            ==
            "FAIL"
        ]
    )

    latest_calendar = None

    if calendar_ok:

        latest_calendar = (
            calendar[
                calendar[
                    "week"
                ]
                ==
                latest_week
            ]
            .iloc[
                0
            ]
        )

    metadata = {

        "latest_year":
        latest_year,

        "latest_week":
        latest_week,

        "latest_national_cases":
        latest_total,

        "district_count":
        district_count,

        "recent_weeks":
        recent_weeks,

        "latest_week_start":
        (
            str(
                latest_calendar[
                    "start_date"
                ]
            )
            if latest_calendar
            is not None
            else
            None
        ),

        "latest_week_end":
        (
            str(
                latest_calendar[
                    "end_date"
                ]
            )
            if latest_calendar
            is not None
            else
            None
        ),

        "forecast_target_start":
        (
            str(
                pd.Timestamp(
                    latest_calendar[
                        "end_date"
                    ]
                )
                +
                pd.Timedelta(
                    days=1
                )
            )[
                :10
            ]
            if latest_calendar
            is not None
            else
            None
        ),

        "forecast_target_end":
        (
            str(
                pd.Timestamp(
                    latest_calendar[
                        "end_date"
                    ]
                )
                +
                pd.Timedelta(
                    days=7
                )
            )[
                :10
            ]
            if latest_calendar
            is not None
            else
            None
        ),

        "data_freshness_date":
        (
            str(
                latest_calendar[
                    "end_date"
                ]
            )
            if latest_calendar
            is not None
            else
            None
        ),

        "case_source":
        "NDCU Weekly Dengue Update",

        "training_source":
        "WER-aligned historical surveillance",

        "operational_status":
        (
            "READY_FOR_EXPERIMENTAL_INFERENCE"
            if mandatory_failures.empty
            else
            "NOT_READY"
        ),

        "source_semantics_warning":
        True,

        "generated_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),
    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    print()
    print(
        audit_df.to_string(
            index=False
        )
    )

    print()
    print(
        "=========================================="
    )

    if mandatory_failures.empty:

        print(
            "[PASS] CORE CURRENT DATA "
            "AUDIT PASSED"
        )

        print(
            "[WARN] Source semantics differ "
            "between NDCU and WER."
        )

        print(
            "[STATUS] Experimental inference "
            "only."
        )

    else:

        print(
            "[FAIL] CURRENT DATA AUDIT FAILED"
        )

        print(
            mandatory_failures.to_string(
                index=False
            )
        )

    print(
        "=========================================="
    )

    print()
    print(
        f"Saved: {AUDIT_FILE}"
    )
    print(
        f"Saved: {CALENDAR_FILE}"
    )
    print(
        f"Saved: {METADATA_FILE}"
    )


if __name__ == "__main__":
    main()