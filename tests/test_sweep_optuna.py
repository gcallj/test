import os
import tempfile
import unittest
from pathlib import Path

from analysis.sweep_optuna import (
    BACKEND_GRID,
    BACKEND_OPTUNA,
    SWEEP_BACKEND_ENV,
    build_search_space,
    get_active_backend,
    run_optuna_study,
    should_use_optuna,
)


# Toy specs mirroring the (name, lo, hi, step, is_int) shape of ga_run.GLOBAL_PARAM_SPECS.
TOY_SPECS = [
    ("x", 0.0, 10.0, 0.5, False),
    ("y", -5.0, 5.0, 0.5, False),
    ("z", 1.0, 4.0, 1.0, True),
    ("noise_gene", 0.0, 1.0, 0.1, False),  # unused, present to confirm filtering
]


def _quadratic_objective(params):
    """Maximum at (x=5.0, y=0.0)."""
    return -((params["x"] - 5.0) ** 2 + (params["y"]) ** 2)


class GetActiveBackend(unittest.TestCase):
    def setUp(self):
        self._old = os.environ.pop(SWEEP_BACKEND_ENV, None)

    def tearDown(self):
        if self._old is not None:
            os.environ[SWEEP_BACKEND_ENV] = self._old
        else:
            os.environ.pop(SWEEP_BACKEND_ENV, None)

    def test_default_is_grid(self):
        self.assertEqual(get_active_backend(), BACKEND_GRID)

    def test_env_var_picked_up(self):
        os.environ[SWEEP_BACKEND_ENV] = BACKEND_OPTUNA
        self.assertEqual(get_active_backend(), BACKEND_OPTUNA)
        self.assertTrue(should_use_optuna())

    def test_invalid_value_falls_back_to_grid(self):
        os.environ[SWEEP_BACKEND_ENV] = "tpe_funky"
        self.assertEqual(get_active_backend(), BACKEND_GRID)

    def test_override_wins(self):
        os.environ[SWEEP_BACKEND_ENV] = BACKEND_OPTUNA
        self.assertEqual(get_active_backend(BACKEND_GRID), BACKEND_GRID)


class BuildSearchSpace(unittest.TestCase):
    def test_filters_to_focus_genes(self):
        out = build_search_space(["x", "z"], TOY_SPECS)
        names = [spec[0] for spec in out]
        self.assertEqual(names, ["x", "z"])

    def test_order_follows_focus_genes_arg(self):
        out = build_search_space(["z", "x"], TOY_SPECS)
        names = [spec[0] for spec in out]
        self.assertEqual(names, ["z", "x"])

    def test_unknown_focus_is_silently_dropped_when_others_present(self):
        out = build_search_space(["x", "nope", "y"], TOY_SPECS)
        names = [spec[0] for spec in out]
        self.assertEqual(names, ["x", "y"])

    def test_all_unknown_raises(self):
        with self.assertRaises(ValueError):
            build_search_space(["a", "b"], TOY_SPECS)


class RunOptunaStudy(unittest.TestCase):
    def test_finds_quadratic_optimum(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = run_optuna_study(
                _quadratic_objective,
                focus_genes=["x", "y"],
                all_specs=TOY_SPECS,
                n_trials=40,
                study_name="quadratic_test_unique",
                studies_dir=Path(tmp),
                seed=42,
            )
            self.assertEqual(res["n_trials"], 40)
            # With 40 TPE trials on a 21x21 discrete grid, the best should land within 1 step of optimum.
            self.assertAlmostEqual(res["best_params"]["x"], 5.0, delta=0.5)
            self.assertAlmostEqual(res["best_params"]["y"], 0.0, delta=0.5)
            self.assertGreater(res["best_value"], -0.5)

    def test_study_resumes_when_called_again_with_same_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            first = run_optuna_study(
                _quadratic_objective,
                focus_genes=["x", "y"],
                all_specs=TOY_SPECS,
                n_trials=10,
                study_name="resume_test",
                studies_dir=tmp_path,
                seed=1,
            )
            self.assertEqual(first["n_trials"], 10)
            # Second call with same study_name + storage path resumes.
            second = run_optuna_study(
                _quadratic_objective,
                focus_genes=["x", "y"],
                all_specs=TOY_SPECS,
                n_trials=5,
                study_name="resume_test",
                studies_dir=tmp_path,
            )
            self.assertEqual(second["n_trials"], 15)

    def test_respects_int_spec(self):
        with tempfile.TemporaryDirectory() as tmp:
            captured = []

            def objective(p):
                captured.append(p["z"])
                return -abs(p["z"] - 2.0)

            run_optuna_study(
                objective,
                focus_genes=["z"],
                all_specs=TOY_SPECS,
                n_trials=8,
                study_name="int_test",
                studies_dir=Path(tmp),
                seed=7,
            )
            # All sampled values should be valid integers in [1, 4].
            self.assertTrue(all(float(v).is_integer() for v in captured))
            self.assertTrue(all(1.0 <= v <= 4.0 for v in captured))


if __name__ == "__main__":
    unittest.main()
