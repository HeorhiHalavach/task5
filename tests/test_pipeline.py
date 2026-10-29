import io
import unittest
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

from analysis import Settings, daily_series, detect, statistics
from charts import production_chart
from data_source import export_url, parse_csv, read_workbook
from report import build_report


class PipelineTests(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range("2026-01-01", periods=20)
        self.data = pd.DataFrame(
            [(day, mine, 200.0 if mine == "A" and index == 19 else 100.0)
             for index, day in enumerate(dates) for mine in ("A", "B")],
            columns=["date", "mine", "production"],
        )
        self.series = daily_series(self.data)

    def test_normalized_input_and_aggregate(self):
        csv = self.data.to_csv(index=False).encode()
        parsed = parse_csv(csv)
        self.assertEqual(len(parsed), 40)
        total = daily_series(parsed).query("mine == 'All mines'")
        self.assertEqual(total["production"].iloc[-1], 300)
        self.assertEqual(total["production"].iloc[0], 200)
        stats = statistics(daily_series(parsed))
        self.assertEqual(stats.loc[stats["mine"] == "All mines", "mean"].iloc[0], 205)

    def test_local_workbook_has_valid_records(self):
        workbook = Path(__file__).resolve().parents[1] / "generator.xlsx"
        data = read_workbook(workbook)
        self.assertGreater(len(data), 0)
        self.assertEqual(data.columns.tolist(), ["date", "mine", "production"])
        self.assertFalse(data.duplicated(["date", "mine"]).any())

    def test_bad_records_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_csv(b"date,mine,production\n2026-01-01,A,1\n2026-01-01,A,2\n")
        with self.assertRaisesRegex(ValueError, "Invalid"):
            parse_csv(b"date,mine,production\n2026-01-01,A,-1\n")

    def test_csv_decimal_separators(self):
        content = b'date,mine,production\n2026-01-01,A,"95,5"\n2026-01-01,B,96.5\n'
        parsed = parse_csv(content)
        self.assertEqual(parsed["production"].tolist(), [95.5, 96.5])

    def test_anomaly_methods_reject_spike_and_keep_baseline(self):
        settings = Settings(window=4, percent_threshold=30)
        for method in ("IQR", "Z-score", "Moving average", "Grubbs"):
            with self.subTest(method=method):
                findings = detect(self.series, settings, [method])
                dates = findings.loc[findings["mine"] == "A", "date"]
                self.assertIn(pd.Timestamp("2026-01-20"), dates.to_list())
                self.assertNotIn(pd.Timestamp("2026-01-01"), dates.to_list())

    def test_chart_and_pdf(self):
        findings = detect(self.series, Settings(), ["IQR"])
        for mode in ("Line", "Bar", "Stacked"):
            for degree in range(1, 5):
                with self.subTest(mode=mode, degree=degree):
                    chart = production_chart(self.series, findings, ["All mines"], mode, degree)
                    self.assertIn("layer", chart.to_dict())
        pdf = build_report(self.series, statistics(self.series), findings)
        self.assertTrue(pdf.startswith(b"%PDF-"))
        text = "\n".join(page.extract_text() for page in PdfReader(io.BytesIO(pdf)).pages)
        self.assertIn("Anomaly 1", text)
        self.assertIn("Daily output: All mines", text)

    def test_pdf_keeps_cyrillic_mine_names(self):
        series = self.series.copy()
        series.loc[series["mine"] == "A", "mine"] = "Шахта Север"
        pdf = build_report(series, statistics(series), detect(series, Settings(), ["IQR"]))
        text = "\n".join(page.extract_text() for page in PdfReader(io.BytesIO(pdf)).pages)
        self.assertIn("Шахта Север", text)

    def test_url_is_reconstructed_for_google(self):
        self.assertEqual(
            export_url("https://docs.google.com/spreadsheets/d/test_123/edit#gid=42"),
            "https://docs.google.com/spreadsheets/d/test_123/export?format=csv&gid=42",
        )
        with self.assertRaises(ValueError):
            export_url("http://localhost/private")


if __name__ == "__main__":
    unittest.main()
