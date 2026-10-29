from __future__ import annotations

from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Flowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _report_fonts() -> tuple[str, str]:
    candidates = [
        (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/arialbd.ttf")),
        (Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"), Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")),
    ]
    for regular, bold in candidates:
        if regular.is_file() and bold.is_file():
            pdfmetrics.registerFont(TTFont("ReportSans", str(regular)))
            pdfmetrics.registerFont(TTFont("ReportSans-Bold", str(bold)))
            pdfmetrics.registerFontFamily("ReportSans", normal="ReportSans", bold="ReportSans-Bold")
            return "ReportSans", "ReportSans-Bold"
    return "Helvetica", "Helvetica-Bold"


FONT, FONT_BOLD = _report_fonts()


class SeriesChart(Flowable):
    def __init__(self, series: pd.DataFrame, title: str, width: float = 178 * mm, height: float = 62 * mm):
        super().__init__()
        self.series = series.sort_values("date")
        self.title = title
        self.width = width
        self.height = height

    def draw(self) -> None:
        canvas = self.canv
        canvas.setFont(FONT_BOLD, 10)
        canvas.drawString(0, self.height - 12, self.title)
        left, bottom, top = 42, 24, self.height - 28
        right = self.width - 8
        canvas.setStrokeColor(colors.HexColor("#9aa9b9"))
        canvas.line(left, bottom, right, bottom)
        canvas.line(left, bottom, left, top)
        values = self.series["production"].to_numpy(dtype=float)
        if len(values) == 0:
            return
        low, high = min(0.0, values.min()), values.max()
        span = max(high - low, 1.0)
        canvas.setFont(FONT, 8)
        canvas.drawRightString(left - 5, top - 2, f"{high:,.0f}")
        canvas.drawRightString(left - 5, bottom - 2, f"{low:,.0f}")
        canvas.drawString(left, bottom - 15, self.series["date"].iloc[0].strftime("%Y-%m-%d"))
        canvas.drawRightString(right, bottom - 15, self.series["date"].iloc[-1].strftime("%Y-%m-%d"))
        canvas.setStrokeColor(colors.HexColor("#2274a5"))
        canvas.setLineWidth(1.4)
        step = (right - left) / max(len(values) - 1, 1)
        points = [(left + i * step, bottom + (value - low) / span * (top - bottom)) for i, value in enumerate(values)]
        path = canvas.beginPath()
        path.moveTo(*points[0])
        for point in points[1:]:
            path.lineTo(*point)
        canvas.drawPath(path)


def build_report(series: pd.DataFrame, stats: pd.DataFrame, findings: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=18 * mm, bottomMargin=18 * mm)
    styles = getSampleStyleSheet()
    styles["Normal"].fontName = FONT
    for name in ("Title", "Heading1", "Heading2", "Heading3"):
        styles[name].fontName = FONT_BOLD
    styles.add(ParagraphStyle(name="CenteredTitle", parent=styles["Title"], alignment=TA_CENTER, textColor=colors.HexColor("#143a52")))
    story: list = [Paragraph("Mine production report", styles["CenteredTitle"]), Spacer(1, 5 * mm)]
    start, end = series["date"].min(), series["date"].max()
    story.append(Paragraph(f"Period: {start:%Y-%m-%d} to {end:%Y-%m-%d} | Mines: {len(stats) - 1}", styles["Normal"]))
    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph("Daily production statistics", styles["Heading2"]))
    rows = [["Mine", "Days", "Mean", "Std. dev.", "Median", "IQR"]]
    rows += [[row.mine, str(row.count), f"{row.mean:,.1f}", f"{row.std:,.1f}", f"{row.median:,.1f}", f"{row.iqr:,.1f}"] for row in stats.itertuples(index=False)]
    table = Table(rows, repeatRows=1, colWidths=[47 * mm, 16 * mm, 30 * mm, 28 * mm, 28 * mm, 27 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#143a52")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f5f8")]),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
        ("FONTNAME", (0, 1), (-1, -1), FONT),
    ]))
    story += [table, Spacer(1, 8 * mm)]
    for mine, group in series.groupby("mine", sort=True):
        story += [SeriesChart(group, f"Daily output: {mine}"), Spacer(1, 4 * mm)]
    observations = findings.groupby(["date", "mine", "production"], sort=True, dropna=False)
    story += [Paragraph(f"Detected anomalies: {observations.ngroups}", styles["Heading2"])]
    if findings.empty:
        story.append(Paragraph("No observations crossed the selected thresholds.", styles["Normal"]))
    else:
        for number, ((date, mine, production), detected) in enumerate(observations, 1):
            story.append(Paragraph(f"Anomaly {number}: {escape(str(mine))}, {date:%Y-%m-%d}", styles["Heading3"]))
            methods = "; ".join(f"{row.method} ({row.detail})" for row in detected.itertuples(index=False))
            story.append(Paragraph(f"Output: {production:,.1f} | {escape(methods)}", styles["Normal"]))
            story.append(Spacer(1, 2 * mm))
    document.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buffer.getvalue()


def _footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#d7e2ea"))
    canvas.line(15 * mm, 13 * mm, A4[0] - 15 * mm, 13 * mm)
    canvas.setFont(FONT, 8)
    canvas.setFillColor(colors.HexColor("#617585"))
    canvas.drawString(15 * mm, 9 * mm, "Weyland-Yutani | Production analysis")
    canvas.drawRightString(A4[0] - 15 * mm, 9 * mm, f"Page {document.page}")
    canvas.restoreState()
