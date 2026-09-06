import unittest
import csv
import tempfile
from pathlib import Path

import numpy as np

from scripts.train_paper_multihorizon_teacher import (build_sampler_weights,
                                                       split_protocol_from_manifest)
from utils.paper_cv import (is_derived_source_alias, physical_source_family,
                            stratified_family_folds)


class PaperFamilySplitTests(unittest.TestCase):
    def test_known_multicam_aliases_share_one_physical_family(self):
        derived = "fall::MultiFallEvents/MulticamTrain_chute17"
        canonical = "fall::MulticamTrain/train/Fall/chute17/cam4.avi"
        self.assertEqual(physical_source_family(derived), physical_source_family(canonical))
        self.assertTrue(is_derived_source_alias(derived))
        self.assertFalse(is_derived_source_alias(canonical))

    def test_family_split_is_deterministic_disjoint_and_stratified(self):
        families = np.repeat([f"fall-{i}" for i in range(10)] + [f"normal-{i}" for i in range(10)], 2)
        is_fall = np.repeat([1] * 10 + [0] * 10, 2)
        datasets = np.repeat(["source-a"] * 5 + ["source-b"] * 5 + ["source-a"] * 5 + ["source-b"] * 5, 2)
        first = stratified_family_folds(families, is_fall, datasets, folds=5, seed=42)
        second = stratified_family_folds(families, is_fall, datasets, folds=5, seed=42)
        np.testing.assert_array_equal(first, second)
        for family in np.unique(families):
            self.assertEqual(len(set(first[families == family])), 1)
        for fold in range(5):
            self.assertIn(1, set(is_fall[first == fold]))
            self.assertIn(0, set(is_fall[first == fold]))


class PaperWeightApplicationTests(unittest.TestCase):
    def test_loss_only_does_not_apply_window_weights_to_sampler(self):
        groups = np.asarray(["a", "a", "b"])
        window_weights = np.asarray([3.0, 3.0, 1.0])
        actual = build_sampler_weights(groups, window_weights, True, "loss_only")
        np.testing.assert_allclose(actual, [0.5, 0.5, 1.0])

    def test_legacy_mode_preserves_double_application(self):
        groups = np.asarray(["a", "a", "b"])
        window_weights = np.asarray([3.0, 3.0, 1.0])
        actual = build_sampler_weights(groups, window_weights, True, "sampler_and_loss")
        np.testing.assert_allclose(actual, [1.5, 1.5, 1.0])

    def test_v2_protocol_is_inferred_from_manifest_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "renamed.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("group", "physical_source_family", "outer_fold"),
                )
                writer.writeheader()
                writer.writerow(
                    {"group": "g", "physical_source_family": "f", "outer_fold": 0}
                )
            self.assertEqual(split_protocol_from_manifest(path), "paper_family_disjoint_v2")


if __name__ == "__main__":
    unittest.main()
