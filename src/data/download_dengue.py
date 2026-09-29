"""
DengueShield AI
Automatic NDCU Weekly Dengue Report Downloader

This script:
1. Visits the NDCU weekly-report archive.
2. Finds Weekly Dengue Update PDF links.
3. Extracts year and week number.
4. Downloads the PDFs.
5. Avoids downloading files that already exist.
6. Creates a metadata CSV for later parsing.

Official source:
https://www.dengue.health.gov.lk/weekly-report/
"""

from pathlib import Path
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup


# ============================================================
# Configuration
# ============================================================

BASE_URL = "https://www.dengue.health.gov.lk"

ARCHIVE_PAGES = [
    "https://www.dengue.health.gov.lk/weekly-report/",
    "https://www.dengue.health.gov.lk/weekly-report/page/2/",
]

OUTPUT_DIR = Path("data/raw/dengue/reports")

METADATA_FILE = Path(
    "data/raw/dengue/report_metadata.csv"
)

TARGET_YEAR = 2026

REQUEST_TIMEOUT = 30


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}


# ============================================================
# Helper functions
# ============================================================

def create_output_directory():
    """Create report directory if it does not exist."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"[INFO] Report directory: "
        f"{OUTPUT_DIR.resolve()}"
    )


def get_page_html(url):
    """Download HTML from an archive page."""

    print(f"[INFO] Reading archive page:")
    print(f"       {url}")

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    return response.text


def extract_week_and_year(url):
    """
    Extract year and week number from a PDF URL.

    Example:
    .../2026/09/Weekly-Dengue-Update-2026-Week-37.pdf
    """

    pattern = (
        r"Weekly-Dengue-Update-"
        r"(\d{4})-Week-(\d+)"
    )

    match = re.search(
        pattern,
        url,
        re.IGNORECASE
    )

    if not match:
        return None, None

    year = int(match.group(1))
    week = int(match.group(2))

    return year, week


def discover_pdf_links():
    """
    Find all 2026 Weekly Dengue Update PDFs
    from the archive pages.
    """

    reports = {}

    for archive_url in ARCHIVE_PAGES:

        html = get_page_html(archive_url)

        soup = BeautifulSoup(
            html,
            "html.parser"
        )

        for link in soup.find_all(
            "a",
            href=True
        ):

            href = link["href"].strip()

            # Convert relative URL to full URL.
            if href.startswith("/"):
                href = BASE_URL + href

            # Ignore non-PDF links.
            if ".pdf" not in href.lower():
                continue

            # Only interested in weekly dengue updates.
            if "weekly-dengue-update" not in href.lower():
                continue

            year, week = extract_week_and_year(
                href
            )

            if year is None:
                continue

            # Only collect the selected year.
            if year != TARGET_YEAR:
                continue

            reports[week] = {
                "year": year,
                "week": week,
                "url": href,
            }

    # Sort by epidemiological week.
    sorted_reports = [
        reports[week]
        for week in sorted(reports)
    ]

    return sorted_reports


def download_pdf(report):
    """Download a single PDF."""

    year = report["year"]
    week = report["week"]
    url = report["url"]

    filename = (
        f"ndcu_{year}_week_{week:02d}.pdf"
    )

    output_path = (
        OUTPUT_DIR / filename
    )

    # Avoid unnecessary repeat downloads.
    if output_path.exists():

        print(
            f"[SKIP] Week {week:02d} "
            f"already downloaded"
        )

        report["filename"] = filename
        report["status"] = "existing"

        return report

    print(
        f"[DOWNLOAD] "
        f"Week {week:02d}"
    )

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    # Basic check to make sure we got a PDF.
    content_type = (
        response.headers
        .get("Content-Type", "")
        .lower()
    )

    if (
        "pdf" not in content_type
        and not response.content.startswith(
            b"%PDF"
        )
    ):

        raise ValueError(
            f"Week {week} did not return "
            f"a valid PDF."
        )

    output_path.write_bytes(
        response.content
    )

    size_kb = (
        output_path.stat().st_size / 1024
    )

    print(
        f"           Saved: {filename}"
    )

    print(
        f"           Size : "
        f"{size_kb:.1f} KB"
    )

    report["filename"] = filename
    report["status"] = "downloaded"

    return report


def save_metadata(reports):
    """Save report metadata as CSV."""

    df = pd.DataFrame(reports)

    df = df[
        [
            "year",
            "week",
            "filename",
            "url",
            "status",
        ]
    ]

    df = df.sort_values(
        ["year", "week"]
    )

    METADATA_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        METADATA_FILE,
        index=False
    )

    print(
        "\n[INFO] Metadata saved:"
    )

    print(
        f"       {METADATA_FILE.resolve()}"
    )


def validate_reports(reports):
    """Perform simple validation."""

    weeks = sorted(
        report["week"]
        for report in reports
    )

    print(
        "\n=========================================="
    )

    print("REPORT DISCOVERY SUMMARY")

    print(
        "=========================================="
    )

    print(
        f"Year: {TARGET_YEAR}"
    )

    print(
        f"Reports discovered: "
        f"{len(weeks)}"
    )

    print(
        f"Weeks discovered:"
    )

    print(weeks)

    if not weeks:
        raise RuntimeError(
            "No reports were discovered."
        )

    # Identify any gaps.
    expected = set(
        range(
            min(weeks),
            max(weeks) + 1
        )
    )

    missing = sorted(
        expected - set(weeks)
    )

    if missing:

        print(
            f"\n[WARNING] Missing weeks: "
            f"{missing}"
        )

    else:

        print(
            "\n[PASS] No missing weeks "
            "between first and latest report."
        )

    return weeks


def main():

    print(
        "\n=========================================="
    )

    print(
        "DENGUESHIELD AI - NDCU DOWNLOADER"
    )

    print(
        "==========================================\n"
    )

    create_output_directory()

    reports = discover_pdf_links()

    validate_reports(reports)

    downloaded_reports = []

    print(
        "\n=========================================="
    )

    print("DOWNLOADING REPORTS")

    print(
        "==========================================\n"
    )

    for report in reports:

        try:

            result = download_pdf(
                report.copy()
            )

            downloaded_reports.append(
                result
            )

            # Be polite to the government server.
            time.sleep(0.5)

        except Exception as error:

            print(
                f"[ERROR] Week "
                f"{report['week']}: "
                f"{error}"
            )

            failed_report = (
                report.copy()
            )

            failed_report[
                "filename"
            ] = ""

            failed_report[
                "status"
            ] = "failed"

            downloaded_reports.append(
                failed_report
            )

    save_metadata(
        downloaded_reports
    )

    success_count = sum(
        report["status"]
        in {"downloaded", "existing"}
        for report in downloaded_reports
    )

    print(
        "\n=========================================="
    )

    print("DOWNLOAD COMPLETE")

    print(
        "=========================================="
    )

    print(
        f"Available reports: "
        f"{success_count}"
    )

    print(
        f"Stored in: "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()