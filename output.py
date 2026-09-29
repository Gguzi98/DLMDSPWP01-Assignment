"""Output layer for the DLMDSPWP01 assignment.

Writes the results to a SQLite database with SQLAlchemy and draws the Bokeh
charts. These functions receive finished DataFrames, so the module contains no
analysis logic of its own.
"""

import pandas as pd
from sqlalchemy import create_engine
from bokeh.plotting import figure, output_file, save
from bokeh.layouts import gridplot

DB_PATH = "results.db"
PLOT_PATH = "results.html"


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
