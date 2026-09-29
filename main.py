"""
DLMDSPWP01: Programming with Python

Selects, for each of the four training functions, the ideal function with the
smallest sum of squared errors, then maps the test data to those ideal
functions using the sqrt(2) criterion.

Pipeline:
1. Load the three CSV files
2. Inspect and validate them
3. Select the best ideal function per training function (SSE)
4. Calculate the sqrt(2) threshold per selected function
5. Map the test data point by point
6. Save everything to SQLite with SQLAlchemy
7. Visualise the result with Bokeh
"""

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from bokeh.plotting import figure, output_file, save
from bokeh.layouts import gridplot

TRAIN_PATH = "data/train.csv"
IDEAL_PATH = "data/ideal.csv"
TEST_PATH = "data/test.csv"
DB_PATH = "results.db"
PLOT_PATH = "results.html"

TRAINING_COLUMNS = ["x", "y1", "y2", "y3", "y4"]
IDEAL_COLUMNS = ["x"] + [f"y{n}" for n in range(1, 51)]
TEST_COLUMNS = ["x", "y"]


# ==========================================
# Section 1: Exceptions
# ==========================================

class DataValidationError(Exception):
    """Raised when a dataset does not have the expected structure."""


class MappingError(Exception):
    """Raised when a test point cannot be located on an ideal function."""


# ==========================================
# Section 2: Loading, inspection, validation
# ==========================================

def load_csv(path: str) -> pd.DataFrame:
    """Read one CSV file into a DataFrame.

    Args:
        path: Location of the CSV file.

    Returns:
        The file contents as a DataFrame.

    Raises:
        DataValidationError: If the file does not exist or cannot be parsed.
    """
    try:
        return pd.read_csv(path)
    except FileNotFoundError as error:
        raise DataValidationError(f"File not found: {path}") from error
    except pd.errors.ParserError as error:
        raise DataValidationError(f"File could not be parsed: {path}") from error


def inspect_dataset(name: str, data: pd.DataFrame) -> None:
    """Print the shape, columns and missing value count of a dataset."""
    print(f"\n{name}")
    print("Shape:", data.shape)
    print("Columns:", list(data.columns))
    print("Missing values:", int(data.isna().sum().sum()))


def validate_dataset(name: str, data: pd.DataFrame, expected_columns: list) -> None:
    """Check that a dataset is complete, numeric and has the expected columns.

    Args:
        name: Dataset name, used in the error messages.
        data: The dataset to check.
        expected_columns: Column names the dataset must contain.

    Raises:
        DataValidationError: If any check fails.
    """
    if data.empty:
        raise DataValidationError(f"{name} dataset is empty.")

    missing_columns = [col for col in expected_columns if col not in data.columns]
    if missing_columns:
        raise DataValidationError(f"{name} dataset is missing columns: {missing_columns}")

    if data.isna().any().any():
        raise DataValidationError(f"{name} dataset contains missing values.")

    for column in data.columns:
        if not pd.api.types.is_numeric_dtype(data[column]):
            raise DataValidationError(f"Column {column} in {name} is not numeric.")


# ==========================================
# Section 3: Base class
# ==========================================

class ModelAnalyzer:
    """Base class holding the datasets and the SSE calculation.

    Attributes:
        training_data: DataFrame with columns x, y1 to y4.
        ideal_data: DataFrame with columns x, y1 to y50.
    """

    def __init__(self, training_data: pd.DataFrame, ideal_data: pd.DataFrame):
        """Store the two datasets used by every analysis step."""
        self.training_data = training_data
        self.ideal_data = ideal_data

    def calculate_sse(self, training_values: pd.Series, ideal_values: pd.Series) -> float:
        """Calculate the sum of squared errors between two series.

        Formula: SSE = sum((training value - ideal value) ** 2)

        Args:
            training_values: Values of one training function.
            ideal_values: Values of one ideal function.

        Returns:
            The sum of squared errors as a float.
        """
        differences = training_values - ideal_values
        return float((differences ** 2).sum())


# ==========================================
# Section 4: Selecting the ideal functions
# ==========================================

class IdealFunctionSelector(ModelAnalyzer):
    """Child class that selects the best ideal function per training function.

    Inherits the datasets and the SSE calculation from ModelAnalyzer.

    Attributes:
        best_ideals: Chosen ideal column and its SSE per training column.
        thresholds: Maximum deviation and sqrt(2) threshold per training column.
    """

    def __init__(self, training_data: pd.DataFrame, ideal_data: pd.DataFrame):
        """Initialise the parent class and prepare the result dictionaries."""
        super().__init__(training_data, ideal_data)
        self.best_ideals = {}
        self.thresholds = {}

    def find_best_ideals(self) -> dict:
        """Find the ideal function with the smallest SSE for each training function.

        Compares each of the four training functions against all fifty ideal
        functions, which is 200 SSE calculations in total.

        Returns:
            Dictionary of the form
            {'y1': {'ideal_column': 'y13', 'sse': 34.08}, ...}
        """
        self.best_ideals = {}

        for training_column in ["y1", "y2", "y3", "y4"]:
            sse_per_ideal = {}

            for ideal_column in self.ideal_data.columns:
                if ideal_column == "x":
                    continue
                sse_per_ideal[ideal_column] = self.calculate_sse(
                    self.training_data[training_column],
                    self.ideal_data[ideal_column],
                )

            best_column = min(sse_per_ideal, key=sse_per_ideal.get)
            self.best_ideals[training_column] = {
                "ideal_column": best_column,
                "sse": sse_per_ideal[best_column],
            }

        return self.best_ideals

    def calculate_thresholds(self) -> dict:
        """Calculate the sqrt(2) threshold for each selected ideal function.

        The largest deviation between a training function and its selected
        ideal function is multiplied by sqrt(2). A test point may be assigned
        to that ideal function only if its deviation stays below this value.

        Returns:
            Dictionary of the form
            {'y1': {'ideal_column': 'y13', 'max_deviation': 0.4992,
                    'threshold': 0.7060}, ...}

        Raises:
            MappingError: If find_best_ideals has not been called yet.
        """
        if not self.best_ideals:
            raise MappingError("Call find_best_ideals before calculate_thresholds.")

        self.thresholds = {}

        for training_column in ["y1", "y2", "y3", "y4"]:
            ideal_column = self.best_ideals[training_column]["ideal_column"]
            deviations = (
                self.training_data[training_column] - self.ideal_data[ideal_column]
            ).abs()
            max_deviation = float(deviations.max())

            self.thresholds[training_column] = {
                "ideal_column": ideal_column,
                "max_deviation": max_deviation,
                "threshold": max_deviation * np.sqrt(2),
            }

        return self.thresholds


# ==========================================
# Section 5: Mapping the test data
# ==========================================

class TestDataMapper(IdealFunctionSelector):
    """Child class that assigns the test points to the selected ideal functions.

    Inherits the selection of the ideal functions and the threshold
    calculation, because a test point can only be mapped once both are known.

    Attributes:
        test_data: DataFrame with columns x and y.
        test_results: List of one dictionary per test point.
    """

    def __init__(self, training_data: pd.DataFrame, ideal_data: pd.DataFrame,
                 test_data: pd.DataFrame):
        """Initialise the parent class and store the test dataset."""
        super().__init__(training_data, ideal_data)
        self.test_data = test_data
        self.test_results = []

    def ideal_value_at(self, ideal_column: str, x_value: float) -> float:
        """Return the value of one ideal function at a given x.

        Args:
            ideal_column: Name of the ideal function column, for example y13.
            x_value: The x value to look up.

        Returns:
            The y value of that ideal function at x_value.

        Raises:
            MappingError: If x_value does not appear in the ideal dataset.
        """
        matching_rows = self.ideal_data[self.ideal_data["x"] == x_value]
        if matching_rows.empty:
            raise MappingError(f"x value {x_value} is not present in the ideal dataset.")
        return float(matching_rows[ideal_column].iloc[0])

    def map_test_points(self) -> pd.DataFrame:
        """Assign every test point to an ideal function where the criterion holds.

        The test data is processed line by line. For each point the deviation
        from each of the four selected ideal functions is calculated. The point
        is assigned to the function with the smallest deviation, as long as that
        deviation stays below the sqrt(2) threshold of that function. Points
        that meet no threshold are kept with empty values.

        Returns:
            DataFrame with the columns x, y, ideal_function and delta_y.

        Raises:
            MappingError: If the thresholds have not been calculated yet.
        """
        if not self.thresholds:
            raise MappingError("Call calculate_thresholds before map_test_points.")

        self.test_results = []

        for _, test_row in self.test_data.iterrows():
            x_value = test_row["x"]
            y_value = test_row["y"]
            best_column = None
            best_deviation = None

            for training_column in ["y1", "y2", "y3", "y4"]:
                ideal_column = self.thresholds[training_column]["ideal_column"]
                threshold = self.thresholds[training_column]["threshold"]

                deviation = abs(y_value - self.ideal_value_at(ideal_column, x_value))

                if deviation <= threshold:
                    if best_deviation is None or deviation < best_deviation:
                        best_column = ideal_column
                        best_deviation = deviation

            self.test_results.append({
                "x": x_value,
                "y": y_value,
                "ideal_function": best_column,
                "delta_y": best_deviation,
            })

        return pd.DataFrame(self.test_results)


# ==========================================
# Section 6: Database persistence
# ==========================================

def persist_to_database(training_data: pd.DataFrame, ideal_data: pd.DataFrame,
                        test_results: pd.DataFrame, db_path: str = DB_PATH) -> None:
    """Write the three result tables to a SQLite database with SQLAlchemy.

    The column labels follow the format required by the assignment, while the
    DataFrames keep their short names inside the program.

    Args:
        training_data: The training dataset.
        ideal_data: The ideal function dataset.
        test_results: The mapped test points.
        db_path: File name of the SQLite database.
    """
    engine = create_engine(f"sqlite:///{db_path}")

    training_labels = {"x": "X"}
    for number in range(1, 5):
        training_labels[f"y{number}"] = f"Y{number} (training func)"

    ideal_labels = {"x": "X"}
    for number in range(1, 51):
        ideal_labels[f"y{number}"] = f"Y{number} (ideal func)"

    test_labels = {
        "x": "X (test func)",
        "y": "Y (test func)",
        "delta_y": "Delta Y (test func)",
        "ideal_function": "No. of ideal func",
    }

    training_data.rename(columns=training_labels).to_sql(
        "training_data", engine, if_exists="replace", index=False)
    ideal_data.rename(columns=ideal_labels).to_sql(
        "ideal_functions", engine, if_exists="replace", index=False)
    test_results.rename(columns=test_labels).to_sql(
        "test_results", engine, if_exists="replace", index=False)

    print(f"Database written to {db_path}")


# ==========================================
# Section 7: Visualisation
# ==========================================

def create_visualisation(training_data: pd.DataFrame, ideal_data: pd.DataFrame,
                         test_results: pd.DataFrame, thresholds: dict,
                         output_path: str = PLOT_PATH) -> None:
    """Draw one Bokeh panel per training function and save them as a grid.

    Each panel shows the training points, the selected ideal function, the
    sqrt(2) tolerance band around that ideal function, and the test points
    that were assigned to it.

    Args:
        training_data: The training dataset.
        ideal_data: The ideal function dataset.
        test_results: The mapped test points.
        thresholds: Result of IdealFunctionSelector.calculate_thresholds.
        output_path: File name of the HTML output.
    """
    output_file(output_path)
    panels = []

    for training_column in ["y1", "y2", "y3", "y4"]:
        ideal_column = thresholds[training_column]["ideal_column"]
        threshold = thresholds[training_column]["threshold"]

        panel = figure(
            title=f"Training {training_column} and ideal {ideal_column}",
            x_axis_label="x",
            y_axis_label="y",
            width=450,
            height=350,
        )

        panel.varea(
            x=ideal_data["x"],
            y1=ideal_data[ideal_column] - threshold,
            y2=ideal_data[ideal_column] + threshold,
            fill_alpha=0.2,
            fill_color="gray",
            legend_label="Tolerance band",
        )

        panel.scatter(
            training_data["x"],
            training_data[training_column],
            size=4,
            color="blue",
            alpha=0.5,
            legend_label=f"Training {training_column}",
        )

        panel.line(
            ideal_data["x"],
            ideal_data[ideal_column],
            color="green",
            line_width=2,
            legend_label=f"Ideal {ideal_column}",
        )

        assigned_points = test_results[test_results["ideal_function"] == ideal_column]
        panel.scatter(
            assigned_points["x"],
            assigned_points["y"],
            size=8,
            color="red",
            legend_label="Assigned test points",
        )

        panel.legend.location = "top_left"
        panel.legend.click_policy = "hide"
        panels.append(panel)

    save(gridplot([[panels[0], panels[1]], [panels[2], panels[3]]]))
    print(f"Visualisation written to {output_path}")


# ==========================================
# Section 8: Main execution
# ==========================================

def main() -> None:
    """Run the full analysis from loading the data to writing the output files."""
    try:
        training_data = load_csv(TRAIN_PATH)
        ideal_data = load_csv(IDEAL_PATH)
        test_data = load_csv(TEST_PATH)

        inspect_dataset("Training data", training_data)
        inspect_dataset("Ideal data", ideal_data)
        inspect_dataset("Test data", test_data)

        validate_dataset("Training", training_data, TRAINING_COLUMNS)
        validate_dataset("Ideal", ideal_data, IDEAL_COLUMNS)
        validate_dataset("Test", test_data, TEST_COLUMNS)

        mapper = TestDataMapper(training_data, ideal_data, test_data)
        mapper.find_best_ideals()
        mapper.calculate_thresholds()
        test_results = mapper.map_test_points()

    except DataValidationError as error:
        print("Data validation failed:", error)
        return
    except MappingError as error:
        print("Mapping failed:", error)
        return

    print("\nSelected ideal functions:")
    for training_column in ["y1", "y2", "y3", "y4"]:
        result = mapper.best_ideals[training_column]
        print(f"  {training_column} -> {result['ideal_column']}  SSE: {result['sse']:.4f}")

    print("\nTolerance thresholds:")
    for training_column in ["y1", "y2", "y3", "y4"]:
        result = mapper.thresholds[training_column]
        print(f"  {training_column}: max deviation {result['max_deviation']:.4f}, "
              f"threshold {result['threshold']:.4f}")

    mapped = int(test_results["ideal_function"].notna().sum())
    print(f"\nTest points: {len(test_results)} total, {mapped} mapped, "
          f"{len(test_results) - mapped} unmapped")
    print("\nAssignments per ideal function:")
    print(test_results["ideal_function"].value_counts().to_string())

    persist_to_database(training_data, ideal_data, test_results)
    create_visualisation(training_data, ideal_data, test_results, mapper.thresholds)


if __name__ == "__main__":
    main()
