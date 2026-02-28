from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# -----------------------------
# Config / Helpers
# -----------------------------

@dataclass
class LoadOptions:
    separator: str
    encoding: str
    na_values: Tuple[str, ...]
    sample_rows: int


DEFAULT_NA_VALUES = ("", "NA", "N/A", "na", "n/a", "null", "NULL", "None", "none", "nan", "NaN")


def human_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    v = float(num_bytes)
    for u in units:
        if v < 1024.0 or u == units[-1]:
            return f"{v:.2f} {u}"
        v /= 1024.0
    return f"{v:.2f} TB"


def safe_read_csv(file_bytes: bytes, opts: LoadOptions) -> pd.DataFrame:
    # Use BytesIO so pandas can read it
    buf = io.BytesIO(file_bytes)
    df = pd.read_csv(
        buf,
        sep=opts.separator,
        encoding=opts.encoding,
        na_values=list(opts.na_values),
        keep_default_na=True,
        low_memory=False,
    )
    return df


def infer_separator(filename: str) -> str:
    # Very light guess; user can override in UI
    if filename.lower().endswith(".tsv"):
        return "\t"
    return ","


def get_basic_overview(df: pd.DataFrame) -> dict:
    mem = int(df.memory_usage(deep=True).sum())
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "memory": human_bytes(mem),
        "duplicates": int(df.duplicated().sum()),
    }


def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    total = df.isna().sum()
    pct = (total / max(len(df), 1)) * 100
    out = pd.DataFrame({"missing_count": total, "missing_pct": pct}).sort_values(
        "missing_count", ascending=False
    )
    out = out[out["missing_count"] > 0]
    return out


def split_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    datetime = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
    # Everything else as categorical/text
    other = [c for c in df.columns if c not in numeric and c not in datetime]
    return numeric, datetime, other


def try_parse_dates(df: pd.DataFrame, cols: List[str], max_cols: int = 5) -> pd.DataFrame:
    # Try convert a few user-selected columns to datetime; keep safe
    new_df = df.copy()
    for c in cols[:max_cols]:
        try:
            new_df[c] = pd.to_datetime(new_df[c], errors="coerce", infer_datetime_format=True)
        except Exception:
            pass
    return new_df


def top_values_table(series: pd.Series, top_k: int = 10) -> pd.DataFrame:
    vc = series.value_counts(dropna=False).head(top_k)
    pct = (vc / max(len(series), 1)) * 100
    return pd.DataFrame({"value": vc.index.astype(str), "count": vc.values, "pct": pct.values})


def correlation_pairs(corr: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    # Get top absolute correlations excluding diagonal and duplicates
    if corr.empty:
        return pd.DataFrame(columns=["col_a", "col_b", "corr"])

    c = corr.copy()
    np.fill_diagonal(c.values, np.nan)
    stacked = c.stack(dropna=True).reset_index()
    stacked.columns = ["col_a", "col_b", "corr"]
    stacked["abs_corr"] = stacked["corr"].abs()

    # Remove duplicate pairs (A,B) and (B,A)
    stacked["pair"] = stacked.apply(lambda r: tuple(sorted([r["col_a"], r["col_b"]])), axis=1)
    stacked = stacked.drop_duplicates(subset=["pair"]).drop(columns=["pair"])

    stacked = stacked.sort_values("abs_corr", ascending=False).head(top_k).drop(columns=["abs_corr"])
    return stacked


def plot_hist(df: pd.DataFrame, col: str, bins: int = 30):
    x = df[col].dropna()
    fig, ax = plt.subplots()
    ax.hist(x, bins=bins)
    ax.set_title(f"Histogram: {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    st.pyplot(fig)


def plot_bar_top_values(df: pd.DataFrame, col: str, top_k: int = 10):
    s = df[col].astype("string")
    vc = s.value_counts(dropna=False).head(top_k)
    fig, ax = plt.subplots()
    ax.bar(vc.index.astype(str), vc.values)
    ax.set_title(f"Top {top_k} values: {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)


def plot_corr_heatmap(corr: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, aspect="auto")
    ax.set_title("Correlation heatmap")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.index)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig)


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="CSV Insight Generator", layout="wide")
st.title("CSV Insight Generator")
st.caption("Upload a CSV and get instant data profiling: missing values, stats, correlations, and plots.")

with st.sidebar:
    st.header("Load settings")
    uploaded = st.file_uploader("Upload CSV", type=["csv", "tsv", "txt"])
    sep = st.selectbox("Separator", options=[",", "\t", ";", "|"], index=0)
    encoding = st.selectbox("Encoding", options=["utf-8", "latin-1", "utf-16"], index=0)
    na_text = st.text_area(
        "Treat these as NA (comma-separated)",
        value=",".join(DEFAULT_NA_VALUES),
        help="Add any custom missing-value tokens your dataset uses.",
    )
    sample_rows = st.slider("Rows to preview", min_value=5, max_value=200, value=25)

    st.divider()
    st.header("Analysis settings")
    top_k = st.slider("Top-K categorical values", 5, 30, 10)
    corr_top_k = st.slider("Top correlated pairs", 5, 30, 10)
    bins = st.slider("Histogram bins", 10, 80, 30)

if not uploaded:
    st.info("Upload a CSV to start.")
    st.stop()

# If the user didn’t change sep and we can infer a better default from filename, use it
if sep == ",":
    sep_guess = infer_separator(uploaded.name)
    # only override if guess differs
    if sep_guess != sep:
        sep = sep_guess

na_values = tuple([v.strip() for v in na_text.split(",") if v.strip() != ""])

opts = LoadOptions(separator=sep, encoding=encoding, na_values=na_values, sample_rows=sample_rows)

file_bytes = uploaded.getvalue()
try:
    df = safe_read_csv(file_bytes, opts)
except Exception as e:
    st.error("Failed to read the file. Try changing separator/encoding.")
    st.code(str(e))
    st.stop()

# Optional: let user attempt parsing date columns
with st.expander("Optional: parse date columns (if you have timestamps)"):
    candidate_cols = df.columns.tolist()
    selected_date_cols = st.multiselect("Columns to parse as datetime", candidate_cols, default=[])
    if selected_date_cols:
        df = try_parse_dates(df, selected_date_cols)

overview = get_basic_overview(df)
numeric_cols, datetime_cols, other_cols = split_columns(df)

# -----------------------------
# Overview
# -----------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", overview["rows"])
c2.metric("Columns", overview["cols"])
c3.metric("Duplicates", overview["duplicates"])
c4.metric("Memory", overview["memory"])

st.subheader("Preview")
st.dataframe(df.head(opts.sample_rows), use_container_width=True)

# Schema
st.subheader("Schema")
schema_df = pd.DataFrame(
    {
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "non_null": [int(df[c].notna().sum()) for c in df.columns],
        "nulls": [int(df[c].isna().sum()) for c in df.columns],
        "null_pct": [(df[c].isna().mean() * 100) for c in df.columns],
        "unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
    }
).sort_values("nulls", ascending=False)
st.dataframe(schema_df, use_container_width=True)

# -----------------------------
# Missing values
# -----------------------------
st.subheader("Missing values")
miss = missing_table(df)
if miss.empty:
    st.success("No missing values detected (based on your NA settings).")
else:
    st.dataframe(miss, use_container_width=True)

# -----------------------------
# Numeric stats + plots
# -----------------------------
st.subheader("Numeric columns")
if not numeric_cols:
    st.info("No numeric columns detected.")
else:
    num_stats = df[numeric_cols].describe().T
    # add extra columns
    num_stats["missing"] = df[numeric_cols].isna().sum()
    num_stats["missing_pct"] = df[numeric_cols].isna().mean() * 100
    st.dataframe(num_stats, use_container_width=True)

    with st.expander("Numeric plots"):
        col_to_plot = st.selectbox("Choose a numeric column (histogram)", options=numeric_cols)
        plot_hist(df, col_to_plot, bins=bins)

# -----------------------------
# Categorical/text insights + plots
# -----------------------------
st.subheader("Categorical / text columns")
if not other_cols:
    st.info("No categorical/text columns detected.")
else:
    cat_col = st.selectbox("Choose a categorical column", options=other_cols)
    st.write("Top values")
    st.dataframe(top_values_table(df[cat_col], top_k=top_k), use_container_width=True)

    with st.expander("Categorical plot"):
        plot_bar_top_values(df, cat_col, top_k=top_k)

# -----------------------------
# Correlations
# -----------------------------
st.subheader("Correlations")
if len(numeric_cols) < 2:
    st.info("Need at least 2 numeric columns for correlations.")
else:
    corr = df[numeric_cols].corr(numeric_only=True)
    st.write("Top correlated pairs (absolute value)")
    st.dataframe(correlation_pairs(corr, top_k=corr_top_k), use_container_width=True)

    with st.expander("Correlation heatmap"):
        plot_corr_heatmap(corr)

# -----------------------------
# Download cleaned CSV (optional)
# -----------------------------
st.subheader("Download")
st.caption("Optionally download the dataset as-is (after any date parsing).")
csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV (utf-8)", data=csv_bytes, file_name="cleaned.csv", mime="text/csv")