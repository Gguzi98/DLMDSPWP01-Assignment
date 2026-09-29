"""DLMDSPWP01: Programming with Python

Runs the full analysis: for each of the four training functions the ideal
function with the smallest sum of squared errors is selected, and the test data
is then mapped to those functions using the sqrt(2) criterion.

Pipeline:
1. Load the three CSV files
2. Inspect and validate them
3. Select the best ideal function per training function (SSE)
4. Calculate the sqrt(2) threshold per selected function
5. Map the test data point by point
6. Save everything to SQLite with SQLAlchemy
7. Visualise the result with Bokeh
"""

from analysis import (
    IDEAL_COLUMNS,
    IDEAL_PATH,
    TEST_COLUMNS,
    TEST_PATH,
    TRAINING_COLUMNS,
    TRAIN_PATH,
    DataValidationError,
    MappingError,
    TestDataMapper,
    inspect_dataset,
    load_csv,
    validate_dataset,
)
from output import create_visualisation, persist_to_database


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
