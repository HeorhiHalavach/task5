from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd


def production_chart(
    series: pd.DataFrame,
    findings: pd.DataFrame,
    selected: list[str],
    mode: str,
    degree: int,
) -> alt.Chart:
    if mode == "Stacked":
        visible = series[series["mine"] != "All mines"]
        trend_source = series[series["mine"] == "All mines"]
        chart = alt.Chart(visible).mark_area().encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("production:Q", stack="zero", title="Production"),
            color=alt.Color("mine:N", title="Mine"),
            tooltip=["date:T", "mine:N", alt.Tooltip("production:Q", format=",.1f")],
        )
    else:
        visible = series[series["mine"].isin(selected)]
        trend_source = visible.groupby("date", as_index=False)["production"].sum()
        mark = alt.Chart(visible).mark_line(point=False) if mode == "Line" else alt.Chart(visible).mark_bar()
        chart = mark.encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("production:Q", title="Production"),
            color=alt.Color("mine:N", title="Mine"),
            tooltip=["date:T", "mine:N", alt.Tooltip("production:Q", format=",.1f")],
        )
    chart = chart.properties(height=410)

    marked = findings[findings["mine"].isin(selected if mode != "Stacked" else ["All mines"])]
    if not marked.empty:
        points = marked.drop_duplicates(["date", "mine"])
        overlay = alt.Chart(points).mark_point(size=115, color="#e63946", filled=True).encode(
            x="date:T", y="production:Q", tooltip=["date:T", "mine:N", "method:N", "detail:N"]
        )
        chart = chart + overlay

    if len(trend_source) > degree:
        trend_source = trend_source.sort_values("date")
        x = (trend_source["date"] - trend_source["date"].min()).dt.days.to_numpy(dtype=float)
        y = trend_source["production"].to_numpy(dtype=float)
        scale = max(x.max(), 1.0)
        fitted = np.polynomial.Polynomial.fit(x / scale, y, degree)(x / scale)
        trend = pd.DataFrame({"date": trend_source["date"], "trend": fitted})
        line = alt.Chart(trend).mark_line(color="#f5a623", strokeDash=[7, 4], strokeWidth=2).encode(
            x="date:T", y=alt.Y("trend:Q"), tooltip=["date:T", alt.Tooltip("trend:Q", format=",.1f")]
        )
        chart = chart + line
    return chart
