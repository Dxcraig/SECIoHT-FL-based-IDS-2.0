"""
Dataset Explorer Page: WUSTL-EHMS-2020 and ECU-IoHT Telemetry Analysis.
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
from src.config import (
    WUSTL_FEATURE_NAMES,
    BIOMETRIC_FEATURES,
    PRIMARY_NETWORK_FEATURES,
    DROPPED_COLUMNS
)
from src.preprocessing import load_or_create_sample_data

# Page Setup
st.set_page_config(page_title="Dataset Explorer | SECIoHT-FL", layout="wide")
render_sidebar()

st.title("Dataset and Telemetry Explorer")
st.markdown("Analyze the biometric and network-flow telemetry used to train SECIoHT-FL models.")

# Dataset Tab Selector
dataset_choice = st.radio(
    "Select Dataset to Inspect:",
    ["WUSTL-EHMS-2020 (Primary Research Focus)", "ECU-IoHT (Parked Dataset Analysis)"],
    horizontal=True
)

st.markdown("---")

if "WUSTL" in dataset_choice:
    st.subheader("WUSTL-EHMS-2020 Dataset Exploration")
    st.markdown(
        """
        The **WUSTL-EHMS-2020** dataset contains real-time biometric telemetry collected from medical sensors 
        fused with network flow statistics across healthcare edge networks.
        """
    )

    # Load Sample Telemetry
    df = load_or_create_sample_data()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Usable Features", "36 Cleaned", "Dropped 8 Leaky/ID Cols")
    with col2:
        st.metric("Normal Traffic Ratio", "87.5%", "~7:1 Imbalance")
    with col3:
        st.metric("Attack Ratio", "12.5%", "Spoofing and Alteration")
    with col4:
        st.metric("Balancing Strategy", "SMOTE (Train-Only)", "0% Leakage into Test")

    st.markdown("### Feature Classification Breakdown")
    tab_bio, tab_net, tab_drops = st.tabs(["Biometric Telemetry (Vitals)", "Network Flow Metrics", "Dropped Columns (Why Dropped)"])

    with tab_bio:
        st.markdown("**Biometric Sensor Features (6 Core Measurements):**")
        st.write(
            "Captures patient physiological responses. Spoofing attacks cause unnatural fluctuations in these values."
        )
        bio_cols = [c for c in BIOMETRIC_FEATURES if c in df.columns]
        st.dataframe(df[bio_cols].describe().round(2), use_container_width=True)

        fig_bio = px.box(
            df,
            y=["Mean_Pulse", "Mean_SpO2", "Mean_Temp", "Mean_Blood_Pressure"],
            color="Label",
            title="Biometric Vitals Distribution: Normal (0) vs Attack (1)",
            labels={"Label": "Traffic Class", "value": "Metric Value", "variable": "Sensor"},
            color_discrete_map={0: "#10b981", 1: "#ef4444"}
        )
        st.plotly_chart(fig_bio, use_container_width=True)

    with tab_net:
        st.markdown("**Network Flow Features (Flow Durations, Ports, Packet Rates):**")
        st.write(
            "Captures edge communication behavior. DoS storms and port scanners exhibit extreme deviations here."
        )
        net_cols = [c for c in PRIMARY_NETWORK_FEATURES if c in df.columns]
        st.dataframe(df[net_cols].describe().round(2), use_container_width=True)

        fig_net = px.histogram(
            df,
            x="Dur",
            color="Label",
            nbins=30,
            title="Flow Duration Histogram (Attack vs Normal)",
            labels={"Label": "Class", "Dur": "Duration (seconds)"},
            color_discrete_map={0: "#10b981", 1: "#ef4444"},
            barmode="overlay"
        )
        st.plotly_chart(fig_net, use_container_width=True)

    with tab_drops:
        st.markdown("**8 Non-Predictive or Leaky Columns Dropped During Stage 1:**")
        st.markdown(
            """
            | Column Name | Reason for Dropping | Risk if Retained |
            |---|---|---|
            | `Attack Category` | **Direct Label Leakage** | Gives away the answer; models memorize instead of learning. |
            | `Packet_num` | Row sequence counter | Arbitrary index without generalizable signal. |
            | `SrcAddr` / `DstAddr` | Host IP addresses | Causes models to memorize specific IP identities rather than malicious behavior. |
            | `SrcMac` / `DstMac` | Hardware MAC addresses | Overfits to test environment hardware interfaces. |
            | `Dir` / `Flgs` | Unusable text flags | High dimensionality text without standardized encoding. |
            """
        )

    st.markdown("---")
    st.subheader("Class Imbalance and The SMOTE Leak-Free Split Fix")
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        fig_pie = px.pie(
            df,
            names="Label",
            title="Real-World Target Distribution (WUSTL: 7:1 Normal to Attack)",
            color="Label",
            color_discrete_map={0: "#10b981", 1: "#ef4444"}
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_c2:
        st.markdown("#### Why Splitting First Matters")
        st.markdown(
            """
            In Stage 1, applying SMOTE to the entire dataset risked creating synthetic samples in the test split 
            that were near-clones of training samples.
            
            **Our Methodology Fix in Stage 2:**
            1. **Stratified Split First**: 80% Train, 20% Test (both on 100% real data).
            2. **SMOTE on Train Only**: Synthesize minority attack samples strictly within the training fold.
            3. **Scaler Fitted on Train Only**: Normalization parameters (mean, std) never peek at test records.
            4. **Guaranteed Honest Metrics**: Test set remains 100% unseen, un-oversampled real patient traffic.
            """
        )

else:
    st.subheader("ECU-IoHT Dataset Anomaly Analysis")
    st.markdown(
        """
        The **ECU-IoHT** dataset was evaluated during Stage 1 and subsequently **parked**.
        Below is the quantitative evidence justifying why it was removed from downstream model training.
        """
    )

    ecu_dist = pd.DataFrame([
        {"Class": "Attack Traffic (Synthetic/Over-represented)", "Count": 87754, "Percentage": 78.9},
        {"Class": "Normal Traffic", "Count": 23453, "Percentage": 21.1}
    ])

    col_e1, col_e2 = st.columns(2)

    with col_e1:
        fig_ecu = px.bar(
            ecu_dist,
            x="Class",
            y="Count",
            color="Class",
            title="Inverted Class Distribution in ECU-IoHT",
            text="Count",
            color_discrete_map={
                "Attack Traffic (Synthetic/Over-represented)": "#ef4444",
                "Normal Traffic": "#10b981"
            }
        )
        st.plotly_chart(fig_ecu, use_container_width=True)

    with col_e2:
        st.markdown(
            """
            <div class="warning-box">
            <b>Critical Finding: Inverted Ground Truth</b><br/>
            In authentic healthcare networks, attacks represent less than 10-15% of total flow traffic.
            In ECU-IoHT, attack rows outnumber normal traffic by almost <b>4 to 1</b>.
            </div>
            
            ### Why We Parked ECU-IoHT:
            - **Artificial Construction**: The inverted ratio strongly suggests synthetic capture conditions.
            - **Deceptive High Accuracy**: A naive model predicting "Attack" for every row achieves 79% accuracy without learning any actual threat patterns.
            - **Scientific Integrity**: To avoid contaminating our Federated Learning and Differential Privacy benchmarks with skewed data, we deliberately parked ECU rather than claiming artificial numbers.
            """,
            unsafe_allow_html=True
        )
