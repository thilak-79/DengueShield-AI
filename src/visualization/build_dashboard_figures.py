"""
DengueShield AI
Dashboard Visualization Builder

Creates reusable figures for:
1. Final model comparison
2. Walk-forward validation
3. Latest district current vs forecast cases
4. Latest relative activity levels

Important:
Activity levels are statistical categories based on
district-specific historical percentiles.
They are NOT official public-health alert thresholds.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
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

MODEL_COMPARISON_FILE = Path(
    "models/comparison/"
    "final_model_comparison.csv"
)

WALK_FORWARD_METRICS_FILE = Path(
    "models/walk_forward/"
    "walk_forward_metrics.csv"
)

RISK_FILE = Path(
    "data/processed/"
    "latest_district_risk.csv"
)

OUTPUT_DIR = Path(
    "figures/dashboard"
)


# ============================================================
# VALIDATION
# ============================================================

def check_inputs():

    required = [
        MODEL_COMPARISON_FILE,
        WALK_FORWARD_METRICS_FILE,
        RISK_FILE,
    ]

    for file in required:

        if not file.exists():

            raise FileNotFoundError(
                f"Missing required file: {file}"
            )

    print(
        "[PASS] Required visualization "
        "inputs found"
    )


# ============================================================
# FIGURE 1
# MODEL COMPARISON
# ============================================================

def plot_model_comparison():

    df = pd.read_csv(
        MODEL_COMPARISON_FILE
    )

    pivot = (
        df.pivot(
            index="horizon_weeks",
            columns="model",
            values="mae",
        )
        .sort_index()
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(10, 6),
    )

    ax.set_title(
        "Dengue Forecast Model Comparison"
    )

    ax.set_xlabel(
        "Forecast Horizon (weeks)"
    )

    ax.set_ylabel(
        "Mean Absolute Error (MAE)"
    )

    ax.legend(
        title="Model"
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    output = (
        OUTPUT_DIR
        /
        "01_model_comparison_mae.png"
    )

    plt.savefig(
        output,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"[PASS] Saved {output}"
    )


# ============================================================
# FIGURE 2
# WALK-FORWARD VALIDATION
# ============================================================

def plot_walk_forward():

    df = pd.read_csv(
        WALK_FORWARD_METRICS_FILE
    )

    # Focus on the one-week horizon because
    # this is the main ML forecast used
    # by the prototype.
    data = (
        df[
            df[
                "horizon_weeks"
            ]
            ==
            1
        ]
        .copy()
    )

    pivot = (
        data.pivot(
            index="test_year",
            columns="model",
            values="mae",
        )
        .sort_index()
    )

    ax = pivot.plot(
        marker="o",
        figsize=(10, 6),
    )

    ax.set_title(
        "1-Week Walk-Forward Forecast Performance"
    )

    ax.set_xlabel(
        "Test Year"
    )

    ax.set_ylabel(
        "Mean Absolute Error (MAE)"
    )

    ax.legend(
        title="Model"
    )

    ax.grid(
        alpha=0.3
    )

    plt.tight_layout()

    output = (
        OUTPUT_DIR
        /
        "02_walk_forward_1week_mae.png"
    )

    plt.savefig(
        output,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"[PASS] Saved {output}"
    )


# ============================================================
# FIGURE 3
# CURRENT VS FORECAST
# ============================================================

def plot_district_forecasts():

    df = pd.read_csv(
        RISK_FILE
    )

    data = (
        df[
            [
                "district",
                "current_cases",
                "forecast_cases_1w",
            ]
        ]
        .copy()
        .sort_values(
            "forecast_cases_1w",
            ascending=True,
        )
    )

    y = np.arange(
        len(data)
    )

    height = 0.38

    fig, ax = plt.subplots(
        figsize=(12, 10)
    )

    ax.barh(
        y - height / 2,
        data[
            "current_cases"
        ],
        height,
        label="Current cases",
    )

    ax.barh(
        y + height / 2,
        data[
            "forecast_cases_1w"
        ],
        height,
        label="1-week RF forecast",
    )

    ax.set_yticks(
        y
    )

    ax.set_yticklabels(
        data[
            "district"
        ]
    )

    ax.set_xlabel(
        "Weekly Dengue Cases"
    )

    ax.set_title(
        "Current Cases vs 1-Week Forecast"
    )

    ax.legend()

    plt.tight_layout()

    output = (
        OUTPUT_DIR
        /
        "03_district_current_vs_forecast.png"
    )

    plt.savefig(
        output,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"[PASS] Saved {output}"
    )


# ============================================================
# FIGURE 4
# RELATIVE ACTIVITY LEVELS
# ============================================================

def plot_activity_levels():

    df = pd.read_csv(
        RISK_FILE
    )

    order = [
        "LOW",
        "ELEVATED",
        "HIGH",
        "VERY HIGH",
    ]

    counts = (
        df[
            "activity_level"
        ]
        .value_counts()
        .reindex(
            order,
            fill_value=0,
        )
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    bars = ax.bar(
        counts.index,
        counts.values,
    )

    ax.set_title(
        "Latest Relative Dengue Activity Categories"
    )

    ax.set_xlabel(
        "Relative Activity Level"
    )

    ax.set_ylabel(
        "Number of Districts"
    )

    for bar, value in zip(
        bars,
        counts.values,
    ):

        ax.text(
            bar.get_x()
            +
            bar.get_width() / 2,
            bar.get_height()
            +
            0.1,
            str(
                int(
                    value
                )
            ),
            ha="center",
        )

    plt.tight_layout()

    output = (
        OUTPUT_DIR
        /
        "04_latest_activity_levels.png"
    )

    plt.savefig(
        output,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"[PASS] Saved {output}"
    )


# ============================================================
# OPTIONAL ACTIVITY TABLE
# ============================================================

def save_dashboard_table():

    df = pd.read_csv(
        RISK_FILE
    )

    columns = [
        "district",
        "current_cases",
        "forecast_cases_1w",
        "persistence_forecast_1w",
        "forecast_percent_change",
        "forecast_trend",
        "activity_level",
        "top_driver_1",
        "top_driver_2",
        "top_driver_3",
    ]

    dashboard = (
        df[
            columns
        ]
        .sort_values(
            "forecast_cases_1w",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    output = (
        OUTPUT_DIR
        /
        "latest_dashboard_table.csv"
    )

    dashboard.to_csv(
        output,
        index=False,
    )

    print(
        f"[PASS] Saved {output}"
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
        "DASHBOARD VISUALIZATIONS"
    )

    print(
        "=========================================="
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    check_inputs()

    plot_model_comparison()

    plot_walk_forward()

    plot_district_forecasts()

    plot_activity_levels()

    save_dashboard_table()

    print()
    print(
        "=========================================="
    )

    print(
        "VISUALIZATION STAGE COMPLETE"
    )

    print(
        "=========================================="
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Activity levels are statistical "
        "relative categories."
    )

    print(
        "They are not official dengue "
        "alert thresholds."
    )


if __name__ == "__main__":
    main()