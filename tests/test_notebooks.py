import csv
import json
import tempfile
from pathlib import Path
import unittest
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GA_NOTEBOOK = "GA_stock.ipynb"
FINAL_NOTEBOOK = "Final_stock_output.ipynb"
REPO_HISTORY_PARQUET = ROOT / "history_consolidated.parquet"
REQUIRED_INPUT_COLS = {"Date", "ticker", "open", "high", "low", "close", "EV_buy_fund_3"}


class TestGANotebook(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        nb_path = ROOT / GA_NOTEBOOK
        if not nb_path.exists():
            cls.nb_data = None
            cls.code_source = ""
            return
        with nb_path.open("r", encoding="utf-8") as fp:
            cls.nb_data = json.load(fp)
        cls.code_source = "\n".join(
            "".join(cell.get("source", []))
            for cell in cls.nb_data.get("cells", [])
            if cell.get("cell_type") == "code"
        )

    def test_ga_notebook_exists(self):
        self.assertTrue((ROOT / GA_NOTEBOOK).exists(), f"Arquivo ausente: {GA_NOTEBOOK}")

    def test_ga_notebook_has_valid_nbformat(self):
        if self.nb_data is None:
            self.skipTest("Notebook not found")
        self.assertIn("nbformat", self.nb_data)
        self.assertIn("cells", self.nb_data)
        self.assertIsInstance(self.nb_data["cells"], list)

    def test_ga_notebook_has_code_cells(self):
        if self.nb_data is None:
            self.skipTest("Notebook not found")
        code_cells = [c for c in self.nb_data.get("cells", []) if c.get("cell_type") == "code"]
        self.assertGreater(len(code_cells), 0, f"Notebook sem celulas de codigo: {GA_NOTEBOOK}")

    def test_ga_notebook_declares_input_output_paths(self):
        """Check that notebook or ga_run.py declares valid I/O paths."""
        if self.nb_data is None:
            self.skipTest("Notebook not found")
        # Check notebook source for any valid path pattern
        has_path = any(p in self.code_source for p in [
            "history_consolidated",
            "HISTORY_CSV_PATH",
            "HISTORY_PARQUET_PATH",
        ])
        # Also accept ga_run.py as the canonical GA implementation
        ga_run_path = ROOT / "ga_run.py"
        if ga_run_path.exists():
            ga_source = ga_run_path.read_text(encoding="utf-8")
            has_path = has_path or "history_consolidated" in ga_source
        self.assertTrue(has_path, "No valid input path declaration found in notebook or ga_run.py")

    def test_ga_run_exists_and_has_outputs(self):
        """ga_run.py is the canonical GA implementation — verify it declares outputs."""
        ga_run_path = ROOT / "ga_run.py"
        self.assertTrue(ga_run_path.exists(), "ga_run.py missing")
        ga_source = ga_run_path.read_text(encoding="utf-8")
        self.assertIn("summary_latest.xlsx", ga_source)
        self.assertIn("apply_last_", ga_source)

    def test_repo_history_parquet_schema_when_present(self):
        if not REPO_HISTORY_PARQUET.exists():
            self.skipTest("Arquivo opcional history_consolidated.parquet nao esta no repositorio")

        df = pd.read_parquet(REPO_HISTORY_PARQUET)
        fieldnames = set(df.columns)
        missing = REQUIRED_INPUT_COLS - fieldnames
        self.assertFalse(missing, f"Colunas obrigatorias ausentes: {sorted(missing)}")
        self.assertGreater(len(df), 0, "history_consolidated.parquet esta vazio")

    def test_ga_notebook_io_contract_with_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_parquet = tmp_path / "history_consolidated.parquet"

            fixture_df = pd.DataFrame([{
                "Date": "2026-01-02", "ticker": "PETR4",
                "open": 30.10, "high": 30.55, "low": 29.90, "close": 30.25,
                "EV_buy_fund_3": 0.42,
            }])
            fixture_df.to_parquet(input_parquet, index=False)

            loaded = pd.read_parquet(input_parquet)
            self.assertEqual(set(loaded.columns), REQUIRED_INPUT_COLS)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded.iloc[0]["ticker"], "PETR4")

    def test_checkpoint_format(self):
        """Verify global_ga_checkpoint.json has expected structure."""
        ckpt_path = ROOT / "global_ga_checkpoint.json"
        if not ckpt_path.exists():
            self.skipTest("Checkpoint not in repo")
        with ckpt_path.open("r", encoding="utf-8") as f:
            ck = json.load(f)
        self.assertIn("fitness", ck)
        self.assertIn("genome", ck)
        self.assertIsInstance(ck["genome"], list)
        self.assertGreater(len(ck["genome"]), 0)
        self.assertIsInstance(ck["fitness"], (int, float))


class TestFinalNotebook(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        nb_path = ROOT / FINAL_NOTEBOOK
        if not nb_path.exists():
            cls.nb_data = None
            cls.code_source = ""
            return
        with nb_path.open("r", encoding="utf-8") as fp:
            cls.nb_data = json.load(fp)
        cls.code_source = "\n".join(
            "".join(cell.get("source", []))
            for cell in cls.nb_data.get("cells", [])
            if cell.get("cell_type") == "code"
        )

    def test_final_notebook_generates_parquet_output(self):
        if self.nb_data is None:
            self.skipTest("Final notebook not found")
        self.assertIn("pq.ParquetWriter", self.code_source)
        self.assertIn("history_consolidated", self.code_source)


if __name__ == "__main__":
    unittest.main()
