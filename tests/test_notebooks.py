import csv
import json
import tempfile
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
GA_NOTEBOOK = "GA_stock.ipynb"
FINAL_NOTEBOOK = "Final_stock_output.ipynb"
REPO_HISTORY_CSV = ROOT / "history_consolidated.csv"
REQUIRED_INPUT_COLS = {"Date", "ticker", "open", "high", "low", "close", "EV_buy_fund_3"}


class TestGANotebook(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT / GA_NOTEBOOK).open("r", encoding="utf-8") as fp:
            cls.nb_data = json.load(fp)
        cls.code_source = "\n".join(
            "".join(cell.get("source", []))
            for cell in cls.nb_data.get("cells", [])
            if cell.get("cell_type") == "code"
        )

    def test_ga_notebook_exists(self):
        self.assertTrue((ROOT / GA_NOTEBOOK).exists(), f"Arquivo ausente: {GA_NOTEBOOK}")

    def test_ga_notebook_has_valid_nbformat(self):
        self.assertIn("nbformat", self.nb_data)
        self.assertIn("nbformat_minor", self.nb_data)
        self.assertIn("cells", self.nb_data)
        self.assertIsInstance(self.nb_data["cells"], list)

    def test_ga_notebook_has_code_cells(self):
        code_cells = [c for c in self.nb_data.get("cells", []) if c.get("cell_type") == "code"]
        self.assertGreater(len(code_cells), 0, f"Notebook sem células de código: {GA_NOTEBOOK}")

    def test_ga_notebook_has_window_aggregation_helper(self):
        self.assertIn("def aggregate_window_stats(stats_list", self.code_source)

    def test_ga_notebook_declares_input_output_paths(self):
        self.assertIn('HISTORY_PARQUET_PATH = "/content/drive/MyDrive/history_consolidated.parquet"', self.code_source)
        self.assertIn('OUTPUT_DIR       = "/content/drive/MyDrive/"', self.code_source)
        self.assertIn("apply_PER_TICKER_WFGA_intraday__H{FWD_H}__APPLY{APPLY_DAYS}D__v2.xlsx", self.code_source)
        self.assertIn("apply_last_{APPLY_DAYS}d__H{FWD_H}__v2.csv", self.code_source)

    def test_repo_history_csv_schema_when_present(self):
        if not REPO_HISTORY_CSV.exists():
            self.skipTest("Arquivo opcional history_consolidated.csv não está no repositório")

        with REPO_HISTORY_CSV.open("r", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            fieldnames = set(reader.fieldnames or [])
            missing = REQUIRED_INPUT_COLS - fieldnames
            self.assertFalse(missing, f"Colunas obrigatórias ausentes em history_consolidated.csv: {sorted(missing)}")

            try:
                first_row = next(reader)
            except StopIteration:
                self.fail("history_consolidated.csv está vazio")

        self.assertIsNotNone(first_row.get("Date"))
        self.assertIsNotNone(first_row.get("ticker"))

    def test_ga_notebook_io_contract_with_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_csv = tmp_path / "history_consolidated.csv"

            with input_csv.open("w", newline="", encoding="utf-8") as fp:
                writer = csv.DictWriter(fp, fieldnames=list(REQUIRED_INPUT_COLS))
                writer.writeheader()
                writer.writerow(
                    {
                        "Date": "2026-01-02",
                        "ticker": "PETR4",
                        "open": "30.10",
                        "high": "30.55",
                        "low": "29.90",
                        "close": "30.25",
                        "EV_buy_fund_3": "0.42",
                    }
                )

            with input_csv.open("r", encoding="utf-8") as fp:
                reader = csv.DictReader(fp)
                self.assertEqual(set(reader.fieldnames or []), REQUIRED_INPUT_COLS)
                rows = list(reader)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["ticker"], "PETR4")

            apply_days = 5
            fwd_h = 5
            output_dir = tmp_path / "outputs"
            output_dir.mkdir(parents=True, exist_ok=True)

            out_xlsx = output_dir / f"apply_PER_TICKER_WFGA_intraday__H{fwd_h}__APPLY{apply_days}D__v2.xlsx"
            out_apply_csv = output_dir / f"apply_last_{apply_days}d__H{fwd_h}__v2.csv"

            out_xlsx.write_bytes(b"placeholder-xlsx")
            with out_apply_csv.open("w", newline="", encoding="utf-8") as fp:
                writer = csv.DictWriter(fp, fieldnames=["Date", "ticker", "close", "signal"])
                writer.writeheader()
                writer.writerow(
                    {
                        "Date": "2026-01-02",
                        "ticker": "PETR4",
                        "close": "30.25",
                        "signal": "BUY",
                    }
                )

            self.assertTrue(out_xlsx.exists())
            self.assertTrue(out_apply_csv.exists())


class TestFinalNotebook(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT / FINAL_NOTEBOOK).open("r", encoding="utf-8") as fp:
            cls.nb_data = json.load(fp)
        cls.code_source = "\n".join(
            "".join(cell.get("source", []))
            for cell in cls.nb_data.get("cells", [])
            if cell.get("cell_type") == "code"
        )

    def test_final_notebook_generates_parquet_output(self):
        self.assertIn('OUT_HIST_PARQUET = Path("/content/drive/MyDrive/history_consolidated.parquet")', self.code_source)
        self.assertIn("pq.ParquetWriter", self.code_source)
        self.assertIn('reg_p = reg_p[reg_p["ticker"].isin(tickers_sa_set)]', self.code_source)
        self.assertIn('df = df.dropna(subset=["open", "high", "low", "close"]).copy()', self.code_source)
        self.assertIn("def open_parquet_dataset_with_retry", self.code_source)
        self.assertIn("def dataset_to_table_with_retry", self.code_source)
        self.assertIn("[warn] falha no scan inicial do REG", self.code_source)
        self.assertIn('out["split"] = out["split"].astype("string").fillna("" )'.replace(" ",""), self.code_source.replace(" ",""))


if __name__ == "__main__":
    unittest.main()
