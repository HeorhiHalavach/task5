from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from analysis import Settings, daily_series, detect, statistics
from charts import production_chart
from data_source import read_sheet, read_workbook
from report import build_report


st.set_page_config(page_title="Mine production", page_icon="⛏️", layout="wide")
st.title("Mine production dashboard")
st.caption("Daily production from the linked Google Sheet")


@st.cache_data(ttl=60, show_spinner=False)
def load(url: str) -> pd.DataFrame:
    return read_sheet(url)


@st.cache_data(ttl=60, show_spinner=False)
def load_local(path: str, modified_ns: int) -> pd.DataFrame:
    return read_workbook(path)


with st.sidebar:
    st.header("Source")
    sheet_url = st.text_input("Google Sheet link (optional)", value=os.getenv("SHEET_URL", ""))
    st.caption("Leave blank to use generator.xlsx from this folder.")
    if st.button("Refresh data", width="stretch"):
        load.clear()
        load_local.clear()
    st.header("Anomaly detection")
    methods = st.multiselect("Methods", ["IQR", "Z-score", "Moving average", "Grubbs"], default=["IQR", "Z-score", "Moving average", "Grubbs"])
    iqr_multiplier = st.number_input("IQR fence multiplier", min_value=0.1, value=1.5, step=0.1)
    z_threshold = st.number_input("Z-score threshold", min_value=0.1, value=3.0, step=0.1)
    window = st.number_input("Moving average days", min_value=2, value=7, step=1)
    percent_threshold = st.number_input("Moving average difference (%)", min_value=1.0, value=25.0, step=1.0)
    grubbs_alpha = st.number_input("Grubbs significance", min_value=0.001, max_value=0.2, value=0.05, step=0.001, format="%.3f")

try:
    if sheet_url:
        data = load(sheet_url)
    else:
        workbook_path = Path(__file__).with_name("generator.xlsx")
        data = load_local(str(workbook_path), workbook_path.stat().st_mtime_ns)
except Exception as error:
    st.error(f"Could not load production data: {error}")
    st.stop()

series = daily_series(data)
stats = statistics(series)
settings = Settings(iqr_multiplier, z_threshold, int(window), percent_threshold, grubbs_alpha)
findings = detect(series, settings, methods)

left, middle, right = st.columns(3)
left.metric("Mines", data["mine"].nunique())
middle.metric("Days", data["date"].nunique())
right.metric("Anomalies", len(findings.drop_duplicates(["date", "mine"])))
st.subheader("Daily statistics")
st.dataframe(stats.rename(columns={"mine": "Mine", "count": "Days", "mean": "Mean", "std": "Std. deviation", "median": "Median", "iqr": "IQR"}), width="stretch", hide_index=True)

st.subheader("Production over time")
names = stats["mine"].tolist()
selected = st.selectbox("Series", names, index=names.index("All mines"))
mode = st.segmented_control("Chart", ["Line", "Bar", "Stacked"], default="Line")
degree = st.slider("Trendline degree", min_value=1, max_value=4, value=2)
st.altair_chart(production_chart(series, findings, [selected], mode, degree), width="stretch")

st.subheader("Anomaly details")
st.dataframe(findings, width="stretch", hide_index=True)
report_key = (int(pd.util.hash_pandas_object(series, index=False).sum()), settings, tuple(methods))
if st.session_state.get("report_key") != report_key:
    st.session_state.pop("report_pdf", None)
if st.button("Generate detailed PDF"):
    st.session_state["report_pdf"] = build_report(series, stats, findings)
    st.session_state["report_key"] = report_key
if "report_pdf" in st.session_state:
    st.download_button(
        "Download PDF",
        data=st.session_state["report_pdf"],
        file_name="mine_production_report.pdf",
        mime="application/pdf",
    )
