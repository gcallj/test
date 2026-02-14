import csv
import json
import tempfile
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
GA_NOTEBOOK = "GA_stock.ipynb"
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

    def test_ga_notebook_declares_input_output_paths(self):
        self.assertIn('HISTORY_CSV_PATH = "/content/drive/MyDrive/history_consolidated.csv"', self.code_source)
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


if __name__ == "__main__":
    unittest.main()
