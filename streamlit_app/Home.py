"""
Main Landing Page for SECIoHT-FL IDS Application.
"""
import streamlit as st
from pathlib import Path
import sys

# Ensure local app packages are importable
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from components.sidebar import render_sidebar
from src.preprocessing import load_or_create_sample_data

# Page Setup
st.set_page_config(
    page_title="SECIoHT-FL | Privacy-Preserving IDS",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
css_file = APP_DIR / "styles" / "custom.css"
if css_file.exists():
    with open(css_file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Render Sidebar
render_sidebar()

# Pre-initialize sample data in background
load_or_create_sample_data()

# Hero Section
st.title("SECIoHT-FL Intrusion Detection System")
st.markdown(
    "**A Privacy-Preserving Federated Learning Architecture for Internet of Healthcare Things (IoHT) Devices**"
)

# Top KPIs
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Analyzed Features",
        value="36 Features",
        delta="WUSTL-EHMS-2020",
        delta_color="off"
    )

with col2:
    st.metric(
        label="Federated Clients",
        value="3 Edge Hospitals",
        delta="Disjoint Local Shards",
        delta_color="off"
    )

with col3:
    st.metric(
        label="Privacy Budget (Epsilon)",
        value="ε <= 0.44",
        delta="Opacus DP-SGD (Noise 1.5)",
        delta_color="normal"
    )

with col4:
    st.metric(
        label="Replicated Accuracy",
        value="87.7%",
        delta="Centralized DNN (Pre-sweep)",
        delta_color="normal"
    )

st.markdown("---")

# Architectural Flow & Problem Statement
col_arch, col_why = st.columns([3, 2])

with col_arch:
    st.subheader("Federated Architecture Overview")
    st.markdown(
        """
        In modern connected healthcare, patient biometrics (ECG, SpO2, pulse rate) cannot be centralized without privacy violations.
        SECIoHT-FL decouples intrusion detection from centralized data collection:
        """
    )

    st.markdown(
        """
```mermaid
flowchart LR
    subgraph Hospital1["Hospital A (Edge Node 1)"]
        D1[("Local Patient Vitals<br/>ECG, SpO2, Pulse")] --> M1["Local DNN Model"]
        M1 --> DP1["Local DP-SGD Noise<br/>(Opacus: sigma=1.5, C=1e-4)"]
    end

    subgraph Hospital2["Hospital B (Edge Node 2)"]
        D2[("Local Patient Vitals<br/>ECG, SpO2, Pulse")] --> M2["Local DNN Model"]
        M2 --> DP2["Local DP-SGD Noise<br/>(Opacus: sigma=1.5, C=1e-4)"]
    end

    subgraph Server["Federated Aggregation Server"]
        AGG["FedAvg Weight Aggregation<br/>(Weighted Average)"]
        GLOBAL["Updated Global Defense Model"]
    end

    DP1 -->|Noisy Weights Only| AGG
    DP2 -->|Noisy Weights Only| AGG
    AGG --> GLOBAL
    GLOBAL -.->|Broadcast Weights| M1
    GLOBAL -.->|Broadcast Weights| M2
```
        """
    )

with col_why:
    st.subheader("Methodology Innovations and Fixes")
    st.markdown(
        """
        Unlike the original paper which left key steps undocumented, our replication resolves critical methodological risks:
        """
    )
    
    with st.expander("1. Leak-Free SMOTE Split (Stage 2)", expanded=True):
        st.write(
            "Splitting after SMOTE leaks interpolated synthetic samples between train and test. "
            "We enforce a stratified 80/20 split first, running SMOTE strictly on training data so the test set remains 100% real unseen records."
        )

    with st.expander("2. Persistent DPClient Accounting (Stage 4)", expanded=True):
        st.write(
            "Creating a new Opacus PrivacyEngine every round resets the privacy accountant, under-reporting cumulative epsilon. "
            "Our DPClient is persistent, ensuring epsilon accurately accumulates across all communication rounds."
        )

    with st.expander("3. ECU-IoHT Anomaly Discovery (Stage 1)", expanded=False):
        st.write(
            "The ECU dataset displays an inverted label ratio (87k attacks vs 23k normal), opposite to real-world networks. "
            "We deliberately parked ECU to avoid contaminating benchmark evaluations."
        )

st.markdown("---")

# Feature Highlights / Interactive Walkthrough
st.subheader("Application Modules")

m1, m2, m3 = st.columns(3)

with m1:
    st.markdown("#### 1. Dataset Explorer")
    st.write(
        "Inspect the 36 WUSTL features (biometric vs network flow), explore the 7:1 class imbalance, and view why ECU-IoHT is parked."
    )
    st.page_link("pages/1_Dataset_Explorer.py", label="Explore Data and Features")

with m2:
    st.markdown("#### 2. FL and Privacy Lab")
    st.write(
        "Simulate Federated Learning across edge hospitals. Adjust communication rounds and observe how Opacus DP noise trades accuracy for privacy budget epsilon."
    )
    st.page_link("pages/2_FL_and_Privacy_Lab.py", label="Launch FL Simulator")

with m3:
    st.markdown("#### 4. IDS Predictor")
    st.write(
        "Simulate patient telemetry. Inject biometric spoofing or DoS traffic and test how the neural networks flag intrusions."
    )
    st.page_link("pages/4_IDS_Detector.py", label="Test Intrusion Engine")

st.caption("SECIoHT-FL IDS 2.0")
