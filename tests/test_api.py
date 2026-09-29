"""
DengueShield AI
API Integration Tests

Tests both:

1. Historical Evaluation API
2. Experimental Current 2026 API

These tests do not start an external Uvicorn server.
FastAPI TestClient runs the application directly.
"""

from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


# ============================================================
# BASIC API
# ============================================================

def test_root():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["project"] == "DengueShield AI"


def test_health():
    response = client.get(
        "/api/health"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] in {
        "healthy",
        "degraded",
    }


# ============================================================
# HISTORICAL API
# ============================================================

def test_historical_district_count():
    response = client.get(
        "/api/districts"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["count"] == 25
    assert len(
        data["districts"]
    ) == 25


def test_historical_activity():
    response = client.get(
        "/api/activity/latest"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["count"] == 25
    assert len(
        data["data"]
    ) == 25


def test_historical_total_cases():
    response = client.get(
        "/api/activity/latest"
    )

    rows = response.json()[
        "data"
    ]

    total = sum(
        float(
            row[
                "current_cases"
            ]
        )
        for row in rows
    )

    assert total == 1201.0


def test_historical_colombo():
    response = client.get(
        "/api/district/Colombo"
    )

    assert response.status_code == 200

    data = response.json()[
        "district"
    ]

    assert data[
        "district"
    ] == "Colombo"

    assert data[
        "current_cases"
    ] == 318.0

    assert abs(
        data[
            "forecast_cases_1w"
        ]
        -
        295.1549999122
    ) < 1e-5

    assert data[
        "activity_level"
    ] == "HIGH"


def test_historical_unknown_district():
    response = client.get(
        "/api/district/"
        "THIS_DISTRICT_DOES_NOT_EXIST"
    )

    assert response.status_code == 404


# ============================================================
# CURRENT 2026 API
# ============================================================

def test_current_status():
    response = client.get(
        "/api/current/status"
    )

    assert response.status_code == 200

    payload = response.json()

    assert (
        payload["mode"]
        ==
        "EXPERIMENTAL_CURRENT_INFERENCE"
    )

    data = payload["data"]

    assert data[
        "latest_year"
    ] == 2026

    assert data[
        "latest_week"
    ] == 37

    assert data[
        "district_count"
    ] == 25

    assert data[
        "data_freshness_date"
    ] == "2026-09-13"

    assert (
        data[
            "official_public_health_alert"
        ]
        is False
    )


def test_current_activity_count():
    response = client.get(
        "/api/current/activity"
    )

    assert response.status_code == 200

    data = response.json()

    assert data[
        "count"
    ] == 25

    assert len(
        data[
            "data"
        ]
    ) == 25


def test_current_total_cases():
    response = client.get(
        "/api/current/activity"
    )

    rows = response.json()[
        "data"
    ]

    total = sum(
        float(
            row[
                "current_cases"
            ]
        )
        for row in rows
    )

    assert total == 1156.0


def test_current_no_missing_forecasts():
    response = client.get(
        "/api/current/activity"
    )

    rows = response.json()[
        "data"
    ]

    for row in rows:

        assert (
            row[
                "forecast_cases_1w"
            ]
            is not None
        )


def test_current_no_negative_forecasts():
    response = client.get(
        "/api/current/activity"
    )

    rows = response.json()[
        "data"
    ]

    for row in rows:

        assert (
            float(
                row[
                    "forecast_cases_1w"
                ]
            )
            >=
            0
        )


def test_current_activity_counts_total_25():
    response = client.get(
        "/api/current/status"
    )

    data = response.json()[
        "data"
    ]

    counts = data[
        "activity_counts"
    ]

    assert sum(
        counts.values()
    ) == 25

    assert counts[
        "VERY HIGH"
    ] == 1

    assert counts[
        "HIGH"
    ] == 5

    assert counts[
        "ELEVATED"
    ] == 13

    assert counts[
        "LOW"
    ] == 6


def test_current_trend_counts_total_25():
    response = client.get(
        "/api/current/status"
    )

    data = response.json()[
        "data"
    ]

    counts = data[
        "trend_counts"
    ]

    assert sum(
        counts.values()
    ) == 25

    assert counts[
        "INCREASING"
    ] == 7

    assert counts[
        "STABLE"
    ] == 7

    assert counts[
        "DECREASING"
    ] == 10

    assert counts[
        "UNKNOWN"
    ] == 1


def test_current_colombo():
    response = client.get(
        "/api/current/district/Colombo"
    )

    assert response.status_code == 200

    data = response.json()[
        "district"
    ]

    assert data[
        "district"
    ] == "Colombo"

    assert data[
        "current_cases"
    ] == 207.0

    assert abs(
        data[
            "forecast_cases_1w"
        ]
        -
        176.717084
    ) < 1e-3

    assert data[
        "forecast_trend"
    ] == "DECREASING"

    assert data[
        "relative_activity"
    ] == "ELEVATED"


def test_current_kandy():
    response = client.get(
        "/api/current/district/Kandy"
    )

    assert response.status_code == 200

    data = response.json()[
        "district"
    ]

    assert data[
        "current_cases"
    ] == 174.0

    assert abs(
        data[
            "forecast_cases_1w"
        ]
        -
        137.282937
    ) < 1e-3

    assert data[
        "forecast_trend"
    ] == "DECREASING"

    assert data[
        "relative_activity"
    ] == "HIGH"


def test_current_unknown_district():
    response = client.get(
        "/api/current/district/"
        "THIS_DISTRICT_DOES_NOT_EXIST"
    )

    assert response.status_code == 404


# ============================================================
# EXPLAINABILITY API
# ============================================================

def test_historical_explanation_colombo():
    response = client.get(
        "/api/explanation/"
        "district/Colombo"
    )

    assert response.status_code == 200

    data = response.json()

    assert data[
        "district"
    ] == "Colombo"

    assert len(
        data[
            "drivers"
        ]
    ) > 0


def test_current_explanation_colombo():
    response = client.get(
        "/api/current/explanation/"
        "district/Colombo"
    )

    assert response.status_code == 200

    data = response.json()

    assert data[
        "district"
    ] == "Colombo"

    assert len(
        data[
            "data"
        ]
    ) > 0


def test_current_global_explanation():
    response = client.get(
        "/api/current/explanation/global"
    )

    assert response.status_code == 200

    data = response.json()

    assert data[
        "count"
    ] > 0


# ============================================================
# HISTORICAL / CURRENT SEPARATION
# ============================================================

def test_historical_current_are_not_same_dataset():

    historical = client.get(
        "/api/district/Colombo"
    ).json()[
        "district"
    ]

    current = client.get(
        "/api/current/district/Colombo"
    ).json()[
        "district"
    ]

    assert (
        historical[
            "current_cases"
        ]
        !=
        current[
            "current_cases"
        ]
    )

    assert (
        historical[
            "forecast_cases_1w"
        ]
        !=
        current[
            "forecast_cases_1w"
        ]
    )

    assert (
        historical[
            "current_cases"
        ]
        ==
        318.0
    )

    assert (
        current[
            "current_cases"
        ]
        ==
        207.0
    )