from __future__ import annotations

import io
import re
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd
from openpyxl import load_workbook


SHEET_ID = re.compile(r"/spreadsheets/d/([A-Za-z0-9_-]+)")
GID = re.compile(r"(?:[?#&]gid=)(\d+)")
MAX_CSV_BYTES = 8_000_000


def export_url(sheet_url: str) -> str:
    match = SHEET_ID.search(sheet_url)
    if not match:
        raise ValueError("Enter a Google Sheets link containing /spreadsheets/d/.")
    gid = GID.search(sheet_url)
    return (
        f"https://docs.google.com/spreadsheets/d/{match.group(1)}/export"
        f"?format=csv&gid={gid.group(1) if gid else '0'}"
    )


def read_sheet(sheet_url: str) -> pd.DataFrame:
    request = Request(export_url(sheet_url), headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=20) as response:
        content = response.read(MAX_CSV_BYTES + 1)
    if len(content) > MAX_CSV_BYTES:
        raise ValueError("The sheet is too large to load.")
    return parse_csv(content)


def parse_csv(content: bytes) -> pd.DataFrame:
    try:
        raw = pd.read_csv(io.BytesIO(content), usecols=["date", "mine", "production"])
    except (UnicodeError, ValueError) as error:
        raise ValueError("Expected CSV columns: date, mine, production.") from error

    return validate_records(raw)


def read_workbook(path: str | Path) -> pd.DataFrame:
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        rows = []
        for date, mine, production in workbook["Data"].iter_rows(min_row=2, max_col=3, values_only=True):
            if date is None:
                break
            rows.append((date, mine, production))
    finally:
        workbook.close()
    return validate_records(pd.DataFrame(rows, columns=["date", "mine", "production"]))


def validate_records(raw: pd.DataFrame) -> pd.DataFrame:

    raw = raw.dropna(how="all", subset=["date", "mine", "production"])
    if raw.empty:
        raise ValueError("The sheet contains no production records.")

    date = pd.to_datetime(raw["date"], errors="coerce")
    mine = raw["mine"].astype("string").str.strip()
    production = pd.to_numeric(raw["production"].astype("string").str.replace(",", ".", regex=False), errors="coerce")
    invalid = date.isna() | mine.isna() | mine.eq("") | mine.eq("All mines") | production.isna() | production.lt(0)
    if invalid.any():
        raise ValueError(f"Invalid production records: {int(invalid.sum())}.")

    data = pd.DataFrame({"date": date.dt.normalize(), "mine": mine, "production": production})
    if data.duplicated(["date", "mine"]).any():
        raise ValueError("The sheet contains duplicate mine/date records.")
    return data.sort_values(["date", "mine"], kind="stable").reset_index(drop=True)
