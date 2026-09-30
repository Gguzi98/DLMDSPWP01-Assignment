"""Unit tests for the analysis program.

Run with: python -m unittest test_main.py -v
"""

import unittest

import numpy as np
import pandas as pd

from analysis import (
    DataValidationError,
    IdealFunctionSelector,
    MappingError,
    ModelAnalyzer,
    TestDataMapper,
    load_csv,
    validate_dataset,
)


def build_training_data() -> pd.DataFrame:
    """Return a small training dataset with two columns that follow y = x."""
    return pd.DataFrame({
        "x": [1.0, 2.0, 3.0],
        "y1": [1.0, 2.0, 3.0],
        "y2": [2.0, 4.0, 6.0],
        "y3": [1.0, 2.0, 3.0],
        "y4": [2.0, 4.0, 6.0],
    })


def build_ideal_data() -> pd.DataFrame:
    """Return a small ideal dataset where y1 matches x and y2 matches 2x."""
    return pd.DataFrame({
        "x": [1.0, 2.0, 3.0],
        "y1": [1.0, 2.0, 3.0],
        "y2": [2.0, 4.0, 6.0],
        "y3": [9.0, 9.0, 9.0],
    })


class TestLoading(unittest.TestCase):
    """Tests for loading and validating the datasets."""

    def test_missing_file_raises_error(self):
        """A file that does not exist is reported as a validation error."""
        with self.assertRaises(DataValidationError):
            load_csv("data/does_not_exist.csv")

    def test_missing_column_raises_error(self):
        """A dataset without a required column is rejected."""
        data = pd.DataFrame({"x": [1.0], "y1": [1.0]})
        with self.assertRaises(DataValidationError):
            validate_dataset("Training", data, ["x", "y1", "y2"])

    def test_missing_value_raises_error(self):
        """A dataset with an empty cell is rejected."""
        data = pd.DataFrame({"x": [1.0, 2.0], "y1": [1.0, np.nan]})
        with self.assertRaises(DataValidationError):
            validate_dataset("Training", data, ["x", "y1"])

    def test_text_column_raises_error(self):
        """A dataset with a non numeric column is rejected."""
        data = pd.DataFrame({"x": [1.0], "y1": ["text"]})
        with self.assertRaises(DataValidationError):
            validate_dataset("Training", data, ["x", "y1"])

    def test_valid_dataset_passes(self):
        """A complete numeric dataset passes validation."""
        data = pd.DataFrame({"x": [1.0, 2.0], "y1": [1.0, 2.0]})
        self.assertIsNone(validate_dataset("Training", data, ["x", "y1"]))


class TestModelAnalyzer(unittest.TestCase):
    """Tests for the SSE calculation in the base class."""

    def setUp(self):
        """Create an analyzer with the small datasets."""
        self.analyzer = ModelAnalyzer(build_training_data(), build_ideal_data())

    def test_identical_series_have_zero_sse(self):
        """Two identical series produce an SSE of zero."""
        values = pd.Series([1.0, 2.0, 3.0])
        self.assertEqual(self.analyzer.calculate_sse(values, values), 0.0)

    def test_known_sse_is_correct(self):
        """Differences of 1, 2 and 3 produce an SSE of 14."""
        training = pd.Series([1.0, 2.0, 3.0])
        ideal = pd.Series([0.0, 0.0, 0.0])
        self.assertEqual(self.analyzer.calculate_sse(training, ideal), 14.0)


class TestIdealFunctionSelector(unittest.TestCase):
    """Tests for the selection of the ideal functions and the thresholds."""

    def setUp(self):
        """Create a selector with the small datasets."""
        self.selector = IdealFunctionSelector(build_training_data(), build_ideal_data())

    def test_best_ideal_is_the_matching_column(self):
        """Training y1 follows the ideal y1 exactly, so y1 is selected."""
        best_ideals = self.selector.find_best_ideals()
        self.assertEqual(best_ideals["y1"]["ideal_column"], "y1")
        self.assertEqual(best_ideals["y2"]["ideal_column"], "y2")

    def test_threshold_is_max_deviation_times_root_two(self):
        """The threshold equals the largest deviation multiplied by sqrt(2)."""
        self.selector.find_best_ideals()
        thresholds = self.selector.calculate_thresholds()
        expected = thresholds["y1"]["max_deviation"] * np.sqrt(2)
        self.assertAlmostEqual(thresholds["y1"]["threshold"], expected)

    def test_thresholds_before_selection_raise_error(self):
        """Calculating thresholds without a selection is not allowed."""
        with self.assertRaises(MappingError):
            self.selector.calculate_thresholds()


class TestTestDataMapper(unittest.TestCase):
    """Tests for the mapping of the test points."""

    def setUp(self):
        """Create a mapper with one point inside and one outside the band."""
        test_data = pd.DataFrame({"x": [1.0, 2.0], "y": [1.2, 50.0]})
        self.mapper = TestDataMapper(build_training_data(), build_ideal_data(), test_data)

    def test_inherits_methods_from_the_selector(self):
        """The mapper can select ideal functions without defining the method."""
        best_ideals = self.mapper.find_best_ideals()
        self.assertEqual(best_ideals["y1"]["ideal_column"], "y1")

    def test_lookup_of_unknown_x_raises_error(self):
        """Looking up an x value that is not in the ideal data is an error."""
        with self.assertRaises(MappingError):
            self.mapper.ideal_value_at("y1", 99.0)

    def test_mapping_before_thresholds_raises_error(self):
        """Mapping without calculated thresholds is not allowed."""
        with self.assertRaises(MappingError):
            self.mapper.map_test_points()

    def test_point_outside_the_band_stays_unassigned(self):
        """A point far from every ideal function is kept without an assignment."""
        self.mapper.find_best_ideals()
        self.mapper.thresholds = {
            "y1": {"ideal_column": "y1", "max_deviation": 0.5, "threshold": 0.5},
            "y2": {"ideal_column": "y2", "max_deviation": 0.5, "threshold": 0.5},
            "y3": {"ideal_column": "y1", "max_deviation": 0.5, "threshold": 0.5},
            "y4": {"ideal_column": "y2", "max_deviation": 0.5, "threshold": 0.5},
        }
        results = self.mapper.map_test_points()
        self.assertEqual(results.loc[0, "ideal_function"], "y1")
        self.assertAlmostEqual(results.loc[0, "delta_y"], 0.2)
        self.assertTrue(pd.isna(results.loc[1, "ideal_function"]))
        self.assertTrue(pd.isna(results.loc[1, "delta_y"]))


if __name__ == "__main__":
    unittest.main()
