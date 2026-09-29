# DLMDSPWP01 - Programming with Python

Selects the ideal function with the smallest sum of squared errors for each of
the four training functions, then maps the test data to those functions using
the sqrt(2) criterion. Results are written to a SQLite database and an
interactive Bokeh chart.

## Usage

    pip install -r requirements.txt
    python main.py
    python -m unittest test_main.py -v

Written for Python 3.14. Outputs `results.db` and `results.html`.
