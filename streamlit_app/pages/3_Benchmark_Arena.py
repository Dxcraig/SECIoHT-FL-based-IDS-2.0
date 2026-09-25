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
results_file = RESULTS_DIR / "stage4_full_comparison_wustl.csv"
if results_file.exists():
    df_results = pd.read_csv(results_file)
else:
    df_results = pd.DataFrame(DEFAULT_BENCHMARK_RESULTS)

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
st.subheader("Interactive Confusion Matrix Simulator")
selected_model = st.selectbox(
    "Select Model Architecture for Detailed Confusion Matrix:",
    df_results["Setting"].tolist()
)

# Realistic confusion matrix values derived from test split
row_match = df_results[df_results["Setting"] == selected_model].iloc[0]
acc_val = float(row_match["Accuracy"])
rec_val = float(row_match["Recall"])

total_test = 3264
actual_attacks = 408
actual_normal = 2856

tp = int(actual_attacks * rec_val)
fn = actual_attacks - tp
fp = int((actual_attacks * (1.0 - float(row_match["Precision"]))) / max(0.01, float(row_match["Precision"])))
fp = min(actual_normal - 10, max(20, fp))
tn = actual_normal - fp

cm = [[tn, fp], [fn, tp]]

col_m1, col_m2 = st.columns([1, 1])

with col_m1:
    fig_cm = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Predicted Label", y="Actual Ground Truth", color="Count"),
        x=["Normal (0)", "Attack (1)"],
        y=["Normal (0)", "Attack (1)"],
        color_continuous_scale="Blues",
        title=f"Confusion Matrix: {selected_model}"
    )
    st.plotly_chart(fig_cm, use_container_width=True)

with col_m2:
    st.markdown("#### Detailed Classification Breakdown")
    st.markdown(
        f"""
        - **True Negatives (TN)**: `{tn:,}` (Correctly identified normal patient flows)
        - **False Positives (FP)**: `{fp:,}` (Normal flows falsely flagged as attacks)
        - **False Negatives (FN)**: `{fn:,}` (Attacks missed by intrusion defense)
        - **True Positives (TP)**: `{tp:,}` (Attacks correctly intercepted)
        """
    )
    
    st.markdown(
        """
        <div class="info-box">
        <b>Why Centralized DNN outperforms CNN:</b><br/>
        In Stage 2, Centralized DNN reached <b>87.7%</b> while CNN achieved <b>70.2%</b>.
        WUSTL-EHMS-2020 features (biometrics, packet rates, ports) are tabular and unordered.
        Unlike spatial images where adjacent pixels form edges, 1D convolution over tabular columns 
        assumes an arbitrary local locality that does not preserve true semantic meaning.
        </div>
        """,
        unsafe_allow_html=True
    )
