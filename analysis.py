from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import t


METHODS = {
    "IQR": "Outside the interquartile range fence",
    "Z-score": "Far from the series mean",
    "Moving average": "Far from the previous days' average",
    "Grubbs": "Statistically unusual under a normal distribution",
}


@dataclass(frozen=True)
class Settings:
    iqr_multiplier: float = 1.5
    z_threshold: float = 3.0
    window: int = 7
    percent_threshold: float = 25.0
    grubbs_alpha: float = 0.05


def daily_series(data: pd.DataFrame) -> pd.DataFrame:
    total = data.groupby("date", sort=False, as_index=False)["production"].sum()
    total["mine"] = "All mines"
    return pd.concat([data, total[data.columns]], ignore_index=True)


def statistics(series: pd.DataFrame) -> pd.DataFrame:
    grouped = series.groupby("mine", sort=True)["production"]
    result = grouped.agg(count="count", mean="mean", std="std", median="median")
    result["iqr"] = grouped.quantile(0.75) - grouped.quantile(0.25)
    return result.fillna({"std": 0}).reset_index()


def grubbs_mask(values: np.ndarray, alpha: float) -> np.ndarray:
    remaining = np.arange(len(values))
    marked = np.zeros(len(values), dtype=bool)
    while len(remaining) >= 3:
        sample = values[remaining]
        deviation = np.abs(sample - sample.mean())
        std = sample.std(ddof=1)
        if std == 0:
            break
        index = int(deviation.argmax())
        n = len(sample)
        critical_t = t.ppf(1 - alpha / (2 * n), n - 2)
        critical_g = (n - 1) / np.sqrt(n) * np.sqrt(critical_t**2 / (n - 2 + critical_t**2))
        if deviation[index] / std <= critical_g:
            break
        marked[remaining[index]] = True
        remaining = np.delete(remaining, index)
    return marked


def detect(series: pd.DataFrame, settings: Settings, methods: list[str]) -> pd.DataFrame:
    findings: list[pd.DataFrame] = []
    for mine, group in series.groupby("mine", sort=False):
        group = group.sort_values("date")
        values = group["production"].to_numpy(dtype=float)
        if "IQR" in methods:
            q1, q3 = np.quantile(values, [0.25, 0.75])
            spread = q3 - q1
            mask = (values < q1 - settings.iqr_multiplier * spread) | (values > q3 + settings.iqr_multiplier * spread)
            findings.extend(_rows(group, mask, "IQR", q1 - settings.iqr_multiplier * spread, q3 + settings.iqr_multiplier * spread))
        if "Z-score" in methods:
            std = values.std(ddof=1) if len(values) > 1 else 0
            score = np.abs(values - values.mean()) / std if std > 0 else np.zeros(len(values))
            findings.extend(_rows(group, score > settings.z_threshold, "Z-score", score))
        if "Moving average" in methods:
            baseline = group["production"].shift(1).rolling(settings.window, min_periods=settings.window).mean().to_numpy()
            distance = np.divide(np.abs(values - baseline), np.abs(baseline), out=np.full(len(values), np.nan), where=baseline != 0) * 100
            distance[(baseline == 0) & (values != 0)] = np.inf
            distance[(baseline == 0) & (values == 0)] = 0
            findings.extend(_rows(group, distance > settings.percent_threshold, "Moving average", distance))
        if "Grubbs" in methods:
            findings.extend(_rows(group, grubbs_mask(values, settings.grubbs_alpha), "Grubbs"))
    if not findings:
        return pd.DataFrame(columns=["date", "mine", "production", "method", "detail"])
    return pd.concat(findings, ignore_index=True).sort_values(["date", "mine", "method"])


def _rows(group: pd.DataFrame, mask: np.ndarray, method: str, *details: object) -> list[pd.DataFrame]:
    if not np.any(mask):
        return []
    result = group.loc[mask, ["date", "mine", "production"]].copy()
    result["method"] = method
    if method == "IQR":
        result["detail"] = f"Expected range: {details[0]:.1f}-{details[1]:.1f}"
    elif details:
        result["detail"] = [f"Score: {value:.1f}" for value in details[0][mask]]
    else:
        result["detail"] = METHODS[method]
    return [result]
