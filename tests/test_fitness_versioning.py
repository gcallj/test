import unittest

from analysis.fitness_versioning import (
    FITNESS_VERSION,
    PAYLOAD_KEY,
    VERSION_MISMATCH_REASON,
    get_fitness_version,
    get_payload_version,
    is_compatible,
    tag_payload,
    versioned_regression_check,
)


def _strict_checker(_cand, _best):
    """Reference checker that always rejects (so we can prove the version short-circuit)."""
    return False, "would_have_rejected"


def _lenient_checker(_cand, _best):
    return True, "ok"


class FitnessVersionConstant(unittest.TestCase):
    def test_get_fitness_version_returns_constant(self):
        self.assertEqual(get_fitness_version(), FITNESS_VERSION)

    def test_constant_is_non_empty_string(self):
        self.assertIsInstance(FITNESS_VERSION, str)
        self.assertTrue(FITNESS_VERSION.startswith("v"))


class TagPayload(unittest.TestCase):
    def test_tag_stamps_current_version_by_default(self):
        payload = {"fitness": 18.45, "genome": [1, 2, 3]}
        out = tag_payload(payload)
        self.assertEqual(out[PAYLOAD_KEY], FITNESS_VERSION)
        # Original keys preserved
        self.assertEqual(out["fitness"], 18.45)
        self.assertEqual(out["genome"], [1, 2, 3])

    def test_tag_does_not_mutate_input(self):
        payload = {"fitness": 18.45}
        _ = tag_payload(payload)
        self.assertNotIn(PAYLOAD_KEY, payload)

    def test_tag_accepts_explicit_version(self):
        out = tag_payload({}, version="v6.0")
        self.assertEqual(out[PAYLOAD_KEY], "v6.0")


class GetPayloadVersion(unittest.TestCase):
    def test_returns_version_when_present(self):
        self.assertEqual(get_payload_version({PAYLOAD_KEY: "v5.2"}), "v5.2")

    def test_returns_none_for_legacy_payload(self):
        self.assertIsNone(get_payload_version({}))
        self.assertIsNone(get_payload_version({"fitness": 18.0}))

    def test_returns_none_for_none_input(self):
        self.assertIsNone(get_payload_version(None))


class IsCompatible(unittest.TestCase):
    def test_both_none_is_compatible(self):
        self.assertTrue(is_compatible(None, None))

    def test_either_none_is_compatible_legacy_fallback(self):
        self.assertTrue(is_compatible("v5.2", None))
        self.assertTrue(is_compatible(None, "v6.0"))

    def test_same_version_compatible(self):
        self.assertTrue(is_compatible("v5.2", "v5.2"))

    def test_different_versions_incompatible(self):
        self.assertFalse(is_compatible("v5.2", "v6.0"))
        self.assertFalse(is_compatible("v6.0", "v5.2"))


class VersionedRegressionCheck(unittest.TestCase):
    def test_same_version_delegates_to_base_checker(self):
        ok, reason = versioned_regression_check(
            {}, {},
            base_checker=_strict_checker,
            cand_version="v5.2", best_version="v5.2",
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "would_have_rejected")

    def test_different_versions_short_circuit_to_accept(self):
        ok, reason = versioned_regression_check(
            {}, {},
            base_checker=_strict_checker,  # would have rejected if called
            cand_version="v6.0", best_version="v5.2",
        )
        self.assertTrue(ok)
        self.assertTrue(reason.startswith(VERSION_MISMATCH_REASON))
        self.assertIn("v6.0", reason)
        self.assertIn("v5.2", reason)

    def test_legacy_against_versioned_passes_through_to_base(self):
        # Legacy (None) vs versioned -> compatible -> base_checker decides.
        ok, _ = versioned_regression_check(
            {}, {},
            base_checker=_strict_checker,
            cand_version=None, best_version="v5.2",
        )
        self.assertFalse(ok)

    def test_passthrough_when_base_accepts(self):
        ok, reason = versioned_regression_check(
            {}, {},
            base_checker=_lenient_checker,
            cand_version="v5.2", best_version="v5.2",
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "ok")


if __name__ == "__main__":
    unittest.main()
