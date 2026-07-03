
# -*- coding: utf-8 -*-
"""
Universal Data Explorer
========================
A Streamlit app for uploading any tabular dataset (CSV or Excel) and running
robust, automatic exploratory data analysis on it:

  - Summary statistics & missing value profiling
  - Correlation heatmap
  - Per-column distribution plots
  - Target variable analysis (with an auto-inferred suggested target,
    which the user can always override)

The app makes NO assumptions about column names, schema, or domain -- every
uploaded file gets profiled fresh. All charting/statistics code defends
against the common ways real-world data breaks a naive EDA script:
empty files, all-null columns, single-value columns, high-cardinality
categoricals, non-numeric-only or numeric-only datasets, duplicate rows,
and mixed/inconsistent dtypes.

# ---------------------------------------------------------------------------
# PART 1: IMPORTS
# ---------------------------------------------------------------------------
"""

import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# PART 2: PAGE CONFIG & AESTHETIC STYLING
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Universal Data Explorer",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS on top of the .streamlit/config.toml theme -- gives the app a
# polished, "designed" feel rather than default Streamlit chrome.
CUSTOM_CSS = """
<style>
    /* ---- Global type ---- */
    html, body, [class*="css"] {
        font-family: "Inter", "Segoe UI", sans-serif;
    }

    /* ---- Header ---- */
    .hero {
        padding: 1.6rem 2rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #6C5CE7 0%, #341f97 100%);
        margin-bottom: 1.6rem;
        box-shadow: 0 8px 24px rgba(108, 92, 231, 0.25);
    }
    .hero h1 {
        color: white;
        font-size: 2.1rem;
        margin-bottom: 0.2rem;
        font-weight: 700;
    }
    .hero p {
        color: rgba(255,255,255,0.85);
        font-size: 1.02rem;
        margin: 0;
    }

    /* ---- Metric cards ---- */
    div[data-testid="stMetric"] {
        background-color: #1A1D27;
        border: 1px solid rgba(255,255,255,0.06);
        padding: 1rem 1.1rem;
        border-radius: 14px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.25);
    }
    div[data-testid="stMetricLabel"] {
        color: #A9ADC1;
        font-weight: 500;
    }

    /* ---- Section headers ---- */
    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin-top: 0.6rem;
        margin-bottom: 0.6rem;
        color: #F4F4F6;
        border-left: 4px solid #6C5CE7;
        padding-left: 0.6rem;
    }

    /* ---- Tabs ---- */
    button[data-baseweb="tab"] {
        font-weight: 600;
        font-size: 0.95rem;
    }

    /* ---- Dataframe corners ---- */
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    .footnote {
        color: #7C8195;
        font-size: 0.82rem;
        margin-top: 2rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PLOTLY_TEMPLATE = "plotly_dark"
ACCENT = "#6C5CE7"
ACCENT_SEQUENCE = ["#6C5CE7", "#00CEC9", "#FD79A8", "#FDCB6E", "#55EFC4", "#74B9FF", "#E17055"]


# ---------------------------------------------------------------------------
# PART 3: DATA LOADING (robust to CSV / Excel, bad encodings, multi-sheet files)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def read_csv_robust(file_bytes: bytes) -> pd.DataFrame:
    """Try a sequence of common encodings/separators before giving up."""
    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
    seps = [None, ",", ";", "\t"]  # None -> let pandas sniff it

    last_error = None
    for enc in encodings:
        for sep in seps:
            try:
                buf = io.BytesIO(file_bytes)
                df = pd.read_csv(buf, encoding=enc, sep=sep, engine="python")
                if df.shape[1] > 1 or sep is not None:
                    return df
            except Exception as e:  # noqa: BLE001 - intentionally broad, we retry
                last_error = e
                continue
    raise ValueError(f"Could not parse CSV with any known encoding/separator. Last error: {last_error}")


@st.cache_data(show_spinner=False)
def get_excel_sheet_names(file_bytes: bytes) -> list:
    buf = io.BytesIO(file_bytes)
    xls = pd.ExcelFile(buf)
    return xls.sheet_names


@st.cache_data(show_spinner=False)
def read_excel_sheet(file_bytes: bytes, sheet_name: str) -> pd.DataFrame:
    buf = io.BytesIO(file_bytes)
    return pd.read_excel(buf, sheet_name=sheet_name)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Light, safe cleanup that does not change the meaning of the data."""
    df = df.copy()

    # De-duplicate any repeated column names (common after messy exports)
    if df.columns.duplicated().any():
        cols = pd.Series(df.columns)
        for dup in cols[cols.duplicated()].unique():
            dup_idx = cols[cols == dup].index
            for i, idx in enumerate(dup_idx):
                cols[idx] = f"{dup}_{i}" if i != 0 else dup
        df.columns = cols

    # Strip whitespace from string column names
    df.columns = [str(c).strip() for c in df.columns]

    # Drop fully-empty rows/columns (common export artifact), but keep everything else intact
    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")

    # Try to parse obvious date-like object columns, without forcing anything
    for col in df.select_dtypes(include="object").columns:
        if df[col].dropna().empty:
            continue
        sample = df[col].dropna().astype(str).head(20)
        looks_datey = sample.str.match(r"^\d{4}-\d{2}-\d{2}").mean() > 0.7
        if looks_datey:
            try:
                df[col] = pd.to_datetime(df[col], errors="ignore")
            except Exception:  # noqa: BLE001
                pass

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# PART 4: TARGET VARIABLE INFERENCE (universal -- no hardcoded column names)
# ---------------------------------------------------------------------------

TARGET_NAME_HINTS = [
    "target", "label", "class", "y", "outcome", "result", "trend",
    "churn", "default", "survived", "diagnosis", "category", "status",
]


def infer_target_column(df: pd.DataFrame) -> str:
    """
    Heuristically guess the most likely target column, purely from structure
    and naming -- never assumes a specific domain or dataset.

    Scoring combines:
      - name similarity to common target keywords
      - being positioned last in the dataframe (common export convention)
      - "reasonable" cardinality (a good target is neither constant nor
        almost-unique like an ID column)
    """
    if df.shape[1] == 0:
        return None

    n_rows = max(len(df), 1)
    scores = {}

    for i, col in enumerate(df.columns):
        score = 0.0
        col_lower = str(col).lower()

        # Name-based signal
        if any(hint in col_lower for hint in TARGET_NAME_HINTS):
            score += 5

        # Positional signal: last columns are conventionally targets
        score += (i / max(df.shape[1] - 1, 1)) * 1.5

        nunique = df[col].nunique(dropna=True)
        unique_ratio = nunique / n_rows

        # Penalize obvious ID-like columns (almost all unique values)
        if unique_ratio > 0.9:
            score -= 4
        # Penalize constant columns (no predictive signal at all)
        if nunique <= 1:
            score -= 10

        # Reward classification-shaped targets (few distinct classes)
        if 2 <= nunique <= 20:
            score += 2
        # Mild reward for numeric columns too (regression targets)
        if pd.api.types.is_numeric_dtype(df[col]) and nunique > 20:
            score += 0.5

        # Penalize obvious identifier columns by name
        if col_lower in {"id", "index", "unnamed: 0"} or col_lower.endswith("_id"):
            score -= 6

        scores[col] = score

    return max(scores, key=scores.get)


# ---------------------------------------------------------------------------
# PART 5: EDA COMPONENTS
# ---------------------------------------------------------------------------

def render_overview_metrics(df: pd.DataFrame):
    n_rows, n_cols = df.shape
    n_numeric = df.select_dtypes(include=np.number).shape[1]
    n_categorical = df.select_dtypes(include=["object", "category", "bool"]).shape[1]
    n_missing_cells = int(df.isna().sum().sum())
    missing_pct = (n_missing_cells / (n_rows * n_cols) * 100) if n_rows * n_cols else 0
    n_duplicates = int(df.duplicated().sum())

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Rows", f"{n_rows:,}")
    c2.metric("Columns", f"{n_cols:,}")
    c3.metric("Numeric cols", n_numeric)
    c4.metric("Categorical cols", n_categorical)
    c5.metric("Missing cells", f"{missing_pct:.1f}%")
    c6.metric("Duplicate rows", f"{n_duplicates:,}")


def render_summary_and_missing(df: pd.DataFrame):
    st.markdown('<div class="section-title">Summary Statistics</div>', unsafe_allow_html=True)

    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] > 0:
        st.dataframe(numeric_df.describe().T.style.format(precision=3), use_container_width=True)
    else:
        st.info("No numeric columns detected in this dataset.")

    cat_df = df.select_dtypes(include=["object", "category", "bool"])
    if cat_df.shape[1] > 0:
        with st.expander("Categorical column summary"):
            cat_summary = pd.DataFrame({
                "unique_values": cat_df.nunique(),
                "most_frequent": cat_df.mode().iloc[0] if not cat_df.empty else None,
                "missing": cat_df.isna().sum(),
            })
            st.dataframe(cat_summary, use_container_width=True)

    st.markdown('<div class="section-title">Missing Values</div>', unsafe_allow_html=True)
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        st.success("No missing values detected. Clean dataset!")
        return

    missing_pct = (missing / len(df) * 100).round(2)
    missing_table = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
    st.dataframe(missing_table, use_container_width=True)

    fig = px.bar(
        missing_table.reset_index().rename(columns={"index": "column"}),
        x="column", y="missing_pct",
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=[ACCENT],
        labels={"missing_pct": "% missing", "column": "Column"},
        title="Missing values by column (%)",
    )
    fig.update_layout(margin=dict(t=50, l=10, r=10, b=10), height=380)
    st.plotly_chart(fig, use_container_width=True)


def render_correlation(df: pd.DataFrame):
    st.markdown('<div class="section-title">Correlation Heatmap</div>', unsafe_allow_html=True)

    numeric_df = df.select_dtypes(include=np.number)
    # Drop constant numeric columns -- they produce undefined (NaN) correlations
    numeric_df = numeric_df.loc[:, numeric_df.nunique(dropna=True) > 1]

    if numeric_df.shape[1] < 2:
        st.info("Need at least two non-constant numeric columns to compute correlations.")
        return

    corr = numeric_df.corr(numeric_only=True)

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        template=PLOTLY_TEMPLATE,
        aspect="auto",
    )
    fig.update_layout(margin=dict(t=20, l=10, r=10, b=10), height=min(180 + 40 * len(corr), 800))
    st.plotly_chart(fig, use_container_width=True)

    # Surface the strongest pairwise relationships as a quick callout
    corr_pairs = (
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .rename("correlation")
        .reset_index()
    )
    corr_pairs.columns = ["Feature 1", "Feature 2", "Correlation"]
    corr_pairs["abs_corr"] = corr_pairs["Correlation"].abs()
    top_pairs = corr_pairs.sort_values("abs_corr", ascending=False).head(5).drop(columns="abs_corr")

    if not top_pairs.empty:
        with st.expander("Strongest relationships"):
            st.dataframe(top_pairs.style.format({"Correlation": "{:.3f}"}), use_container_width=True)


def render_distributions(df: pd.DataFrame):
    st.markdown('<div class="section-title">Column Distributions</div>', unsafe_allow_html=True)

    all_cols = list(df.columns)
    if not all_cols:
        st.info("No columns to display.")
        return

    selected_cols = st.multiselect(
        "Choose columns to visualize (defaults to the first 6)",
        options=all_cols,
        default=all_cols[:6],
    )

    if not selected_cols:
        st.info("Select at least one column above to see its distribution.")
        return

    MAX_CATEGORIES = 15  # cap bars shown for high-cardinality categoricals

    cols_per_row = 2
    for i in range(0, len(selected_cols), cols_per_row):
        row_cols = st.columns(cols_per_row)
        for j, col_name in enumerate(selected_cols[i:i + cols_per_row]):
            with row_cols[j]:
                series = df[col_name]

                if pd.api.types.is_numeric_dtype(series):
                    clean_series = series.dropna()
                    if clean_series.nunique() <= 1:
                        st.info(f"'{col_name}' has a single unique value -- nothing to plot.")
                        continue
                    fig = px.histogram(
                        clean_series, x=col_name if col_name in df.columns else None,
                        nbins=min(40, max(10, clean_series.nunique())),
                        template=PLOTLY_TEMPLATE,
                        color_discrete_sequence=[ACCENT],
                        title=col_name,
                    )
                    fig.update_layout(margin=dict(t=40, l=10, r=10, b=10), height=320, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

                elif pd.api.types.is_datetime64_any_dtype(series):
                    counts = series.dropna().dt.to_period("M").astype(str).value_counts().sort_index()
                    if counts.empty:
                        st.info(f"'{col_name}' has no valid dates to plot.")
                        continue
                    fig = px.line(
                        x=counts.index, y=counts.values,
                        template=PLOTLY_TEMPLATE,
                        color_discrete_sequence=[ACCENT],
                        labels={"x": "Period", "y": "Count"},
                        title=f"{col_name} (records over time)",
                    )
                    fig.update_layout(margin=dict(t=40, l=10, r=10, b=10), height=320)
                    st.plotly_chart(fig, use_container_width=True)

                else:
                    value_counts = series.astype(str).value_counts(dropna=True)
                    if value_counts.empty:
                        st.info(f"'{col_name}' has no values to plot.")
                        continue
                    truncated = value_counts.head(MAX_CATEGORIES)
                    if len(value_counts) > MAX_CATEGORIES:
                        st.caption(f"Showing top {MAX_CATEGORIES} of {len(value_counts)} categories.")
                    fig = px.bar(
                        x=truncated.index, y=truncated.values,
                        template=PLOTLY_TEMPLATE,
                        color_discrete_sequence=[ACCENT],
                        labels={"x": col_name, "y": "Count"},
                        title=col_name,
                    )
                    fig.update_layout(margin=dict(t=40, l=10, r=10, b=10), height=320, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)


def render_target_analysis(df: pd.DataFrame, target_col: str):
    st.markdown('<div class="section-title">Target Variable Analysis</div>', unsafe_allow_html=True)

    if target_col is None or target_col not in df.columns:
        st.info("No target column selected.")
        return

    target_series = df[target_col]
    is_numeric_target = pd.api.types.is_numeric_dtype(target_series)
    # Treat low-cardinality numeric columns (e.g. 0/1 flags) as classification-shaped too
    is_classification_shaped = (not is_numeric_target) or target_series.nunique(dropna=True) <= 15

    if is_classification_shaped:
        counts = target_series.astype(str).value_counts(dropna=True)
        col1, col2 = st.columns([1.3, 1])

        with col1:
            fig = px.bar(
                x=counts.index, y=counts.values,
                template=PLOTLY_TEMPLATE,
                color=counts.index,
                color_discrete_sequence=ACCENT_SEQUENCE,
                labels={"x": target_col, "y": "Count"},
                title=f"Class balance — {target_col}",
            )
            fig.update_layout(margin=dict(t=40, l=10, r=10, b=10), height=380, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig_pie = px.pie(
                values=counts.values, names=counts.index,
                template=PLOTLY_TEMPLATE,
                color_discrete_sequence=ACCENT_SEQUENCE,
                hole=0.45,
                title="Proportion",
            )
            fig_pie.update_layout(margin=dict(t=40, l=10, r=10, b=10), height=380)
            st.plotly_chart(fig_pie, use_container_width=True)

        # Flag class imbalance -- genuinely useful, data-driven observation
        if len(counts) >= 2:
            imbalance_ratio = counts.max() / counts.min()
            if imbalance_ratio > 3:
                st.warning(
                    f"Class imbalance detected: the majority class ('{counts.idxmax()}') is "
                    f"{imbalance_ratio:.1f}x larger than the minority class ('{counts.idxmin()}'). "
                    "Consider this when interpreting any modeling built on this target."
                )

        # Numeric features vs. target (grouped distributions)
        numeric_cols = [c for c in df.select_dtypes(include=np.number).columns if c != target_col]
        if numeric_cols:
            feature = st.selectbox("Compare a numeric feature across target classes", numeric_cols)
            plot_df = df[[target_col, feature]].dropna()
            if not plot_df.empty:
                fig_box = px.box(
                    plot_df, x=target_col, y=feature,
                    template=PLOTLY_TEMPLATE,
                    color=target_col,
                    color_discrete_sequence=ACCENT_SEQUENCE,
                    title=f"{feature} by {target_col}",
                )
                fig_box.update_layout(margin=dict(t=40, l=10, r=10, b=10), height=400, showlegend=False)
                st.plotly_chart(fig_box, use_container_width=True)

    else:
        # Numeric / continuous target
        clean_series = target_series.dropna()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean", f"{clean_series.mean():.3g}")
        c2.metric("Median", f"{clean_series.median():.3g}")
        c3.metric("Std dev", f"{clean_series.std():.3g}")
        c4.metric("Skew", f"{clean_series.skew():.3g}")

        fig = px.histogram(
            clean_series, x=target_col,
            marginal="box",
            template=PLOTLY_TEMPLATE,
            color_discrete_sequence=[ACCENT],
            title=f"Distribution of {target_col}",
        )
        fig.update_layout(margin=dict(t=40, l=10, r=10, b=10), height=420)
        st.plotly_chart(fig, use_container_width=True)

        # Correlation of other numeric features with this target
        numeric_df = df.select_dtypes(include=np.number)
        if target_col in numeric_df.columns and numeric_df.shape[1] > 1:
            corr_with_target = (
                numeric_df.corr(numeric_only=True)[target_col]
                .drop(labels=[target_col])
                .sort_values(key=abs, ascending=False)
            )
            if not corr_with_target.empty:
                fig_corr = px.bar(
                    x=corr_with_target.values, y=corr_with_target.index,
                    orientation="h",
                    template=PLOTLY_TEMPLATE,
                    color=corr_with_target.values,
                    color_continuous_scale="RdBu_r",
                    range_color=[-1, 1],
                    labels={"x": "Correlation", "y": "Feature"},
                    title=f"Feature correlation with {target_col}",
                )
                fig_corr.update_layout(margin=dict(t=40, l=10, r=10, b=10), height=380, coloraxis_showscale=False)
                st.plotly_chart(fig_corr, use_container_width=True)


# ---------------------------------------------------------------------------
# PART 6: MAIN APP FLOW
# ---------------------------------------------------------------------------

def main():
    st.markdown(
        """
        <div class="hero">
            <h1>✨ Universal Data Explorer</h1>
            <p>Upload any CSV or Excel file and get instant, robust exploratory data analysis —
            no assumptions about your schema, no fixed target variable.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("📁 Upload data")
        uploaded_file = st.file_uploader(
            "CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            help="Every upload is profiled fresh -- no dataset-specific assumptions.",
        )
        st.caption("Max file size: 200 MB")

    if uploaded_file is None:
        st.info("⬅️ Upload a CSV or Excel file from the sidebar to get started.")
        st.markdown(
            '<p class="footnote">Tip: this app auto-suggests a target variable based on column '
            "names, position, and cardinality -- you can always change it once your data loads.</p>",
            unsafe_allow_html=True,
        )
        return

    file_bytes = uploaded_file.getvalue()

    try:
        if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
            sheet_names = get_excel_sheet_names(file_bytes)
            with st.sidebar:
                sheet = st.selectbox("Sheet", sheet_names) if len(sheet_names) > 1 else sheet_names[0]
            raw_df = read_excel_sheet(file_bytes, sheet)
        else:
            raw_df = read_csv_robust(file_bytes)
    except Exception as e:  # noqa: BLE001
        st.error(
            "Couldn't read this file. It may be corrupted, password-protected, or in an "
            f"unsupported format.\n\nDetails: {e}"
        )
        return

    if raw_df.empty or raw_df.shape[1] == 0:
        st.error("This file loaded but contains no usable data (no rows or no columns).")
        return

    df = clean_dataframe(raw_df)

    if df.empty:
        st.error("After removing fully-empty rows/columns, no data remained. Please check the file.")
        return

    st.success(f"Loaded **{uploaded_file.name}** — {df.shape[0]:,} rows × {df.shape[1]:,} columns")

    with st.sidebar:
        st.header("🎯 Target variable")
        suggested_target = infer_target_column(df)
        columns_with_none = ["(none)"] + list(df.columns)
        default_idx = columns_with_none.index(suggested_target) if suggested_target in columns_with_none else 0
        target_col = st.selectbox(
            "Select target (auto-suggested)",
            options=columns_with_none,
            index=default_idx,
            help="Inferred automatically from column names, position, and cardinality. Override anytime.",
        )
        target_col = None if target_col == "(none)" else target_col

        with st.expander("Preview raw data"):
            st.dataframe(df.head(20), use_container_width=True)

    render_overview_metrics(df)
    st.write("")

    tab_labels = ["📊 Summary & Missing Values", "🔗 Correlations", "📈 Distributions"]
    if target_col:
        tab_labels.append("🎯 Target Analysis")

    tabs = st.tabs(tab_labels)

    with tabs[0]:
        render_summary_and_missing(df)
    with tabs[1]:
        render_correlation(df)
    with tabs[2]:
        render_distributions(df)
    if target_col:
        with tabs[3]:
            render_target_analysis(df, target_col)

    st.markdown(
        '<p class="footnote">Universal Data Explorer — exploration only, no modeling. '
        "Every chart adapts to whatever columns and dtypes are present in your file.</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
