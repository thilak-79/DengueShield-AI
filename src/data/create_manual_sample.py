from pathlib import Path

import pandas as pd


# ============================================================
# DengueShield AI
# Manual dengue dataset from NDCU Weekly Dengue Updates
# Weeks 33-37, 2026
#
# IMPORTANT:
# NDCU sometimes revises a week's numbers in the next report.
# Therefore, for each week below, we use the latest available
# revision within the five reports inspected.
#
# "Nil" values from NDCU reports are represented as 0.
# ============================================================


DISTRICTS_RDHS = [
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


# ------------------------------------------------------------
# Latest available values found within NDCU Reports 33-37.
#
# Week 33 -> revised values in Week 34 report
# Week 34 -> revised values in Week 35 report
# Week 35 -> revised values in Week 36 report
# Week 36 -> revised values in Week 37 report
# Week 37 -> current values in Week 37 report
# ------------------------------------------------------------

WEEKLY_DATA = {
    33: {
        "source": "NDCU Week 34 Report",
        "cases": [
            625,  # Colombo
            605,  # Gampaha
            165,  # Kalutara
            273,  # Kandy
            31,   # Matale
            31,   # Nuwara Eliya
            120,  # Galle
            51,   # Hambantota
            92,   # Matara
            9,    # Jaffna
            8,    # Kilinochchi
            1,    # Mannar
            9,    # Vavuniya
            3,    # Mullaitivu
            32,   # Batticaloa
            19,   # Ampara
            15,   # Trincomalee
            59,   # Kalmunai
            68,   # Kurunegala
            65,   # Puttalam
            37,   # Anuradhapura
            19,   # Polonnaruwa
            65,   # Badulla
            25,   # Monaragala
            85,   # Ratnapura
            101,  # Kegalle
        ],
    },

    34: {
        "source": "NDCU Week 35 Report",
        "cases": [
            441,  # Colombo
            369,  # Gampaha
            137,  # Kalutara
            187,  # Kandy
            15,   # Matale
            18,   # Nuwara Eliya
            92,   # Galle
            45,   # Hambantota
            56,   # Matara
            16,   # Jaffna
            1,    # Kilinochchi
            2,    # Mannar
            1,    # Vavuniya
            2,    # Mullaitivu
            24,   # Batticaloa
            11,   # Ampara
            8,    # Trincomalee
            29,   # Kalmunai
            49,   # Kurunegala
            34,   # Puttalam
            42,   # Anuradhapura
            15,   # Polonnaruwa
            42,   # Badulla
            25,   # Monaragala
            58,   # Ratnapura
            83,   # Kegalle
        ],
    },

    35: {
        "source": "NDCU Week 36 Report",
        "cases": [
            295,  # Colombo
            269,  # Gampaha
            94,   # Kalutara
            150,  # Kandy
            11,   # Matale
            10,   # Nuwara Eliya
            54,   # Galle
            15,   # Hambantota
            43,   # Matara
            17,   # Jaffna
            0,    # Kilinochchi - Nil
            1,    # Mannar
            3,    # Vavuniya
            0,    # Mullaitivu - Nil
            17,   # Batticaloa
            24,   # Ampara
            3,    # Trincomalee
            31,   # Kalmunai
            42,   # Kurunegala
            38,   # Puttalam
            15,   # Anuradhapura
            4,    # Polonnaruwa
            25,   # Badulla
            17,   # Monaragala
            54,   # Ratnapura
            63,   # Kegalle
        ],
    },

    36: {
        "source": "NDCU Week 37 Report",
        "cases": [
            222,  # Colombo
            224,  # Gampaha
            71,   # Kalutara
            158,  # Kandy
            10,   # Matale
            18,   # Nuwara Eliya
            47,   # Galle
            21,   # Hambantota
            41,   # Matara
            19,   # Jaffna
            0,    # Kilinochchi - Nil
            0,    # Mannar - Nil
            0,    # Vavuniya - Nil
            0,    # Mullaitivu - Nil
            21,   # Batticaloa
            12,   # Ampara
            5,    # Trincomalee
            32,   # Kalmunai
            29,   # Kurunegala
            37,   # Puttalam
            30,   # Anuradhapura
            8,    # Polonnaruwa
            23,   # Badulla
            11,   # Monaragala
            55,   # Ratnapura
            58,   # Kegalle
        ],
    },

    37: {
        "source": "NDCU Week 37 Report",
        "cases": [
            207,  # Colombo
            218,  # Gampaha
            73,   # Kalutara
            174,  # Kandy
            12,   # Matale
            14,   # Nuwara Eliya
            69,   # Galle
            22,   # Hambantota
            38,   # Matara
            22,   # Jaffna
            3,    # Kilinochchi
            0,    # Mannar - Nil
            1,    # Vavuniya
            1,    # Mullaitivu
            13,   # Batticaloa
            10,   # Ampara
            5,    # Trincomalee
            31,   # Kalmunai
            38,   # Kurunegala
            33,   # Puttalam
            11,   # Anuradhapura
            12,   # Polonnaruwa
            25,   # Badulla
            11,   # Monaragala
            44,   # Ratnapura
            69,   # Kegalle
        ],
    },
}


# Official/latest totals corresponding to the data above.
EXPECTED_TOTALS = {
    33: 2613,
    34: 1802,
    35: 1295,
    36: 1152,
    37: 1156,
}


def create_dataset():
    rows = []

    for week, week_data in WEEKLY_DATA.items():

        case_values = week_data["cases"]

        # Make sure every week contains all 26 District/RDHS rows.
        if len(case_values) != len(DISTRICTS_RDHS):
            raise ValueError(
                f"Week {week} has {len(case_values)} values, "
                f"but expected {len(DISTRICTS_RDHS)}."
            )

        for district, cases in zip(DISTRICTS_RDHS, case_values):

            rows.append(
                {
                    "year": 2026,
                    "week": week,
                    "district": district,
                    "cases": cases,
                    "source": week_data["source"],
                }
            )

    return pd.DataFrame(rows)


def validate_dataset(df):

    print("\n==========================================")
    print("VALIDATING MANUAL DENGUE DATASET")
    print("==========================================\n")

    # --------------------------------------------------------
    # 1. Total number of rows
    # 5 weeks x 26 District/RDHS units = 130
    # --------------------------------------------------------

    assert len(df) == 130, (
        f"Expected 130 rows but found {len(df)}"
    )

    print(f"[PASS] Total rows: {len(df)}")


    # --------------------------------------------------------
    # 2. Check weeks
    # --------------------------------------------------------

    expected_weeks = {33, 34, 35, 36, 37}

    actual_weeks = set(df["week"].unique())

    assert actual_weeks == expected_weeks

    print(
        f"[PASS] Weeks available: "
        f"{sorted(actual_weeks)}"
    )


    # --------------------------------------------------------
    # 3. Check 26 District/RDHS records per week
    # --------------------------------------------------------

    counts = df.groupby("week")["district"].nunique()

    for week, count in counts.items():

        assert count == 26, (
            f"Week {week} contains only "
            f"{count} District/RDHS units"
        )

    print(
        "[PASS] Every week contains "
        "26 District/RDHS rows"
    )


    # --------------------------------------------------------
    # 4. Check duplicates
    # --------------------------------------------------------

    duplicates = df.duplicated(
        subset=[
            "year",
            "week",
            "district",
        ]
    )

    assert not duplicates.any()

    print("[PASS] No duplicate district-week rows")


    # --------------------------------------------------------
    # 5. Check negative case counts
    # --------------------------------------------------------

    assert (df["cases"] >= 0).all()

    print("[PASS] No negative case counts")


    # --------------------------------------------------------
    # 6. Check weekly totals
    # --------------------------------------------------------

    calculated_totals = (
        df.groupby("week")["cases"]
        .sum()
        .to_dict()
    )

    print("\nWeekly total validation:\n")

    for week, expected_total in EXPECTED_TOTALS.items():

        calculated = calculated_totals[week]

        status = (
            "PASS"
            if calculated == expected_total
            else "FAIL"
        )

        print(
            f"Week {week}: "
            f"calculated={calculated}, "
            f"expected={expected_total} "
            f"[{status}]"
        )

        assert calculated == expected_total, (
            f"Week {week} total mismatch"
        )

    print("\n[PASS] All weekly totals validated")


def save_dataset(df):

    output_path = Path(
        "data/raw/dengue/manual_sample.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nDataset saved successfully:\n"
        f"{output_path.resolve()}"
    )


def main():

    df = create_dataset()

    validate_dataset(df)

    save_dataset(df)

    print("\nFirst 10 rows:\n")

    print(
        df.head(10).to_string(
            index=False
        )
    )

    print(
        "\n=========================================="
    )
    print(
        "MANUAL SAMPLE CREATED SUCCESSFULLY"
    )
    print(
        "=========================================="
    )


if __name__ == "__main__":
    main()