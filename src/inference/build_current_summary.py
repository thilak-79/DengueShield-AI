"""
DengueShield AI
Build Current Operational Summary

Creates dashboard-ready summary information for the
experimental 2026 current inference.

This is not an official public-health alert summary.
"""

from pathlib import Path
import json

import pandas as pd


ACTIVITY_FILE = Path(
    "data/processed/operational/"
    "current_activity_2026.csv"
)

METADATA_FILE = Path(
    "data/processed/operational/"
    "current_data_metadata.json"
)

SUMMARY_FILE = Path(
    "data/processed/operational/"
    "current_activity_summary_2026.csv"
)

STATUS_FILE = Path(
    "data/processed/operational/"
    "current_status_2026.json"
)


def main():

    if not ACTIVITY_FILE.exists():
        raise FileNotFoundError(
            ACTIVITY_FILE
        )

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            METADATA_FILE
        )

    df = pd.read_csv(
        ACTIVITY_FILE
    )

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        metadata = json.load(
            file
        )

    if len(df) != 25:
        raise AssertionError(
            "Expected 25 districts."
        )

    level_order = [
        "LOW",
        "ELEVATED",
        "HIGH",
        "VERY HIGH",
    ]

    level_counts = (
        df[
            "relative_activity"
        ]
        .value_counts()
        .reindex(
            level_order,
            fill_value=0,
        )
    )

    trend_order = [
        "INCREASING",
        "STABLE",
        "DECREASING",
        "UNKNOWN",
    ]

    trend_counts = (
        df[
            "forecast_trend"
        ]
        .value_counts()
        .reindex(
            trend_order,
            fill_value=0,
        )
    )

    current_total = float(
        df[
            "current_cases"
        ]
        .sum()
    )

    forecast_total = float(
        df[
            "forecast_cases_1w"
        ]
        .sum()
    )

    net_change = (
        forecast_total
        -
        current_total
    )

    percent_change = (
        (
            net_change
            /
            current_total
            *
            100.0
        )
        if current_total != 0
        else None
    )

    summary_rows = []

    for level in level_order:

        subset = (
            df[
                df[
                    "relative_activity"
                ]
                ==
                level
            ]
        )

        summary_rows.append(
            {
                "activity_level":
                level,

                "districts":
                int(
                    level_counts[
                        level
                    ]
                ),

                "mean_current_cases":
                (
                    float(
                        subset[
                            "current_cases"
                        ]
                        .mean()
                    )
                    if not subset.empty
                    else
                    0.0
                ),

                "mean_forecast_cases":
                (
                    float(
                        subset[
                            "forecast_cases_1w"
                        ]
                        .mean()
                    )
                    if not subset.empty
                    else
                    0.0
                ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    status = {

        "mode":
        "EXPERIMENTAL_CURRENT_INFERENCE",

        "latest_year":
        metadata[
            "latest_year"
        ],

        "latest_week":
        metadata[
            "latest_week"
        ],

        "surveillance_window_start":
        metadata[
            "latest_week_start"
        ],

        "surveillance_window_end":
        metadata[
            "latest_week_end"
        ],

        "forecast_target_start":
        metadata[
            "forecast_target_start"
        ],

        "forecast_target_end":
        metadata[
            "forecast_target_end"
        ],

        "data_freshness_date":
        metadata[
            "data_freshness_date"
        ],

        "district_count":
        int(
            df[
                "district"
            ]
            .nunique()
        ),

        "current_total_cases":
        current_total,

        "forecast_total_cases":
        forecast_total,

        "forecast_net_change":
        net_change,

        "forecast_percent_change":
        percent_change,

        "activity_counts":
        {
            level:
            int(
                level_counts[
                    level
                ]
            )
            for level
            in level_order
        },

        "trend_counts":
        {
            trend:
            int(
                trend_counts[
                    trend
                ]
            )
            for trend
            in trend_order
        },

        "forecast_model":
        "Random Forest full-history refit",

        "case_source":
        "NDCU",

        "training_source":
        "WER-aligned historical surveillance",

        "source_semantics_warning":
        True,

        "official_public_health_alert":
        False,

        "disclaimer":
        (
            "Experimental current inference. "
            "NDCU operational surveillance is being "
            "used with a model trained on WER-aligned "
            "historical data. This is not an official "
            "Ministry of Health forecast or alert."
        ),
    }

    with open(
        STATUS_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            status,
            file,
            indent=2,
        )

    print(
        "[PASS] Current operational summary created"
    )

    print(
        f"[PASS] Current total: "
        f"{current_total:.0f}"
    )

    print(
        f"[PASS] Forecast total: "
        f"{forecast_total:.2f}"
    )

    print(
        f"[PASS] Net change: "
        f"{net_change:.2f}"
    )

    print(
        f"[PASS] Saved: {SUMMARY_FILE}"
    )

    print(
        f"[PASS] Saved: {STATUS_FILE}"
    )


if __name__ == "__main__":
    main()