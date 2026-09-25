"""
Benchmark Arena Page: Model Evaluation and Centralized vs Federated Comparisons.
Replicates Stage 5 (Evaluation and Performance Benchmarking).
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Ensure local packages are importable
APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from components.sidebar import render_sidebar
from src.config import RESULTS_DIR, DEFAULT_BENCHMARK_RESULTS
from src.artifacts import result_path

# Page Setup
st.set_page_config(page_title="Benchmark Arena | SECIoHT-FL", layout="wide")
render_sidebar()

st.title("Model Arena and Benchmark Evaluation")
st.markdown(
    "Compare Centralized baselines, Federated Learning (FedAvg), and Differential Privacy (DP-SGD) "
    "against the MDPI Electronics 2025 published paper results."
)

st.markdown("---")

# Load Benchmark Table
stage5_file = result_path("stage5")
stage4_file = result_path("stage4")
if stage5_file.exists():
    results_file = stage5_file
    df_results = pd.read_csv(results_file)
    results_source = "Notebook Stage 5 evaluation artifact"
elif stage4_file.exists():
    results_file = stage4_file
    df_results = pd.read_csv(results_file)
    results_source = "Bundled Stage 4 comparison artifact"
else:
    df_results = pd.DataFrame(DEFAULT_BENCHMARK_RESULTS)
    results_source = "Fallback reference table; no notebook result artifact found"

st.caption(f"Data source: {results_source}.")

# Stage 5 exports lowercase metric names; normalize them for this page.
df_results = df_results.rename(columns={
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-Score",
    "f1_score": "F1-Score",
    "epsilon": "Privacy (Epsilon)",
})

# Format numerical columns
format_cols = ["Accuracy", "Precision", "Recall", "F1-Score"]
for c in format_cols:
    if c in df_results.columns:
        df_results[c] = df_results[c].astype(float)

st.subheader("Performance Comparison Matrix")
st.dataframe(
    df_results.style.format({
        "Accuracy": "{:.1%}",
        "Precision": "{:.3f}",
        "Recall": "{:.3f}",
        "F1-Score": "{:.3f}"
    }).highlight_max(subset=["Accuracy", "F1-Score"], color="#dcfce7"),
    use_container_width=True
)

st.markdown("---")

# Metric Visualizations
col_c1, col_c2 = st.columns(2)

with col_c1:
    fig_bar = px.bar(
        df_results,
        x="Setting",
        y=["Accuracy", "F1-Score"],
        barmode="group",
        title="Accuracy vs F1-Score across Architectures and Privacy Settings",
        labels={"value": "Score (0 - 1.0)", "variable": "Metric"}
    )
    fig_bar.update_layout(xaxis_tickangle=-25)
    st.plotly_chart(fig_bar, use_container_width=True)

with col_c2:
    fig_pr = px.scatter(
        df_results,
        x="Recall",
        y="Precision",
        color="Setting",
        size="Accuracy",
        title="Precision vs Recall Trade-off (Bubble Size = Accuracy)",
        hover_data=["Privacy (Epsilon)"]
    )
    st.plotly_chart(fig_pr, use_container_width=True)

st.markdown("---")

# Confusion Matrix Explorer
st.info(
    "Confusion matrices are shown only when the notebook exports actual per-sample predictions. "
    "The current repository contains aggregate metrics, so no synthetic confusion matrix is displayed."
)
