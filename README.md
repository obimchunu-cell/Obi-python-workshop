# Obi-python-workshop
my first attempt at a python data exploration
https://obi-python-workshop-f4hxen6rnm5n35czcguayz.streamlit.app/
# Universal Data Explorer

A Streamlit app that takes **any** CSV or Excel upload and runs robust,
automatic exploratory data analysis on it — no hardcoded columns, no fixed
target variable. It works the same whether you upload sales data, a music
industry dataset, medical records, or anything else with rows and columns.

## What it does

- **Upload** a `.csv`, `.xlsx`, or `.xls` file (multi-sheet Excel supported)
- **Overview metrics**: row/column counts, numeric vs. categorical split, missing %, duplicates
- **Summary statistics & missing values**: per-column stats plus a missing-value chart
- **Correlation heatmap**: auto-computed across numeric columns, with top relationships called out
- **Distributions**: histograms for numeric columns, bar charts for categorical columns, time series for dates — auto-detected per column
- **Target variable analysis**: the app *infers* a likely target column from naming patterns, position, and cardinality, and lets you override it via dropdown. Handles both classification-shaped (class balance, boxplots vs. features) and regression-shaped (distribution, correlation with target) targets.

This app is **exploration-only** 




## Project structure

```
.
├── app.py                  # the whole app (single file, sectioned with PART comments)
├── requirements.txt        # dependencies for Streamlit Cloud / local installs
├── .streamlit/
│   └── config.toml         # theme (dark, purple accent)
└── README.md
```

## Notes on robustness

- CSV loading retries multiple encodings (`utf-8`, `utf-8-sig`, `latin1`, `cp1252`) and separators (`,` `;` tab) before failing
- Fully-empty rows/columns are dropped; duplicate column names are de-duplicated automatically
- Constant columns are excluded from correlation (they'd produce undefined values)
- High-cardinality categorical columns are capped at the top 15 categories in bar charts, with a note on how many were hidden
- Every chart/stat function checks for the specific edge case that would otherwise crash it (empty selection, single unique value, no numeric columns, etc.) and shows a friendly message instead
