"""
Edge Device Intrusion Detection Simulator.
Enables real-time inference on synthetic or custom patient biometric and network telemetry,
with an interactive Test-Set Row Sampler for evaluating unseen test records.
"""
import streamlit as st
import pandas as pd
import numpy as np
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
    ATTACK_PRESETS
)
from src.preprocessing import (
    load_or_create_sample_data,
    create_feature_vector,
    get_default_scaler
)
from src.inference import predict_sample

# Page Setup
st.set_page_config(page_title="IDS Detector | SECIoHT-FL", layout="wide")
render_sidebar()

st.title("Edge IoHT Intrusion Detector")
st.markdown(
    "Simulate an active medical sensor transmitting patient vitals over an IoHT network gateway. "
    "Test how models detect biometric spoofing, packet flooding, and evaluate actual unseen test records."
)

st.markdown("---")

# Load sample dataset
df_samples = load_or_create_sample_data()

# Initialize session state for telemetry parameters if not present
default_values = {
    "pulse": 72.0,
    "spo2": 98.0,
    "temp": 36.8,
    "bp": 120.0,
    "resp": 16.0,
    "ecg": 0.85,
    "sport": 443,
    "dport": 52100,
    "dur": 1.20,
    "pkts": 16,
    "bytes": 1950,
    "srate": 12.5,
    "source_description": "Manual Input"
}

for k, v in default_values.items():
    if f"telemetry_{k}" not in st.session_state:
        st.session_state[f"telemetry_{k}"] = v

# Section 1: Input Control Center (Presets vs Test Set Sampler)
st.subheader("1. Telemetry Source and Scenario Setup")

tab_sampler, tab_presets = st.tabs([
    "Test-Set Row Sampler (Unseen Evaluation Records)",
    "Attack Scenario Presets (Quick Tests)"
])

# Tab 1: Enhanced Test-Set Row Sampler
with tab_sampler:
    st.markdown(
        "Browse and select real records from the unseen test evaluation split. "
        "Load any record directly into the detector to verify if the model's prediction matches the ground truth."
    )

    col_filter, col_select = st.columns([1, 2])

    with col_filter:
        class_filter = st.radio(
            "Filter Test Set by Class:",
            ["All Records", "Attack Traffic Only (Class 1)", "Normal Traffic Only (Class 0)"],
            horizontal=False
        )

    # Filter dataframe
    if "Attack" in class_filter:
        filtered_df = df_samples[df_samples["Label"] == 1]
    elif "Normal" in class_filter:
        filtered_df = df_samples[df_samples["Label"] == 0]
    else:
        filtered_df = df_samples

    sample_options = [
        f"{r['Sample_ID']} | {'Attack (1)' if r['Label'] == 1 else 'Normal (0)'} | "
        f"Pulse: {r['Mean_Pulse']} bpm | SpO2: {r['Mean_SpO2']}% | Dur: {r['Dur']}s | Sport: {int(r['Sport'])}"
        for _, r in filtered_df.iterrows()
    ]

    with col_select:
        selected_option = st.selectbox(
            f"Select Sample Record ({len(filtered_df)} available):",
            sample_options
        )
        selected_id = selected_option.split(" | ")[0]
        chosen_row = df_samples[df_samples["Sample_ID"] == selected_id].iloc[0]

    # Load button and immediate sample metrics
    col_load_btn, col_sample_summary = st.columns([1, 2])

    with col_load_btn:
        if st.button("Load Selected Sample into Detector", type="primary", use_container_width=True):
            st.session_state["telemetry_pulse"] = float(chosen_row["Mean_Pulse"])
            st.session_state["telemetry_spo2"] = float(chosen_row["Mean_SpO2"])
            st.session_state["telemetry_temp"] = float(chosen_row["Mean_Temp"])
            st.session_state["telemetry_bp"] = float(chosen_row["Mean_Blood_Pressure"])
            st.session_state["telemetry_resp"] = float(chosen_row["Respiration_Rate"])
            st.session_state["telemetry_ecg"] = float(chosen_row["Mean_ECG"])
            st.session_state["telemetry_sport"] = int(chosen_row["Sport"])
            st.session_state["telemetry_dport"] = int(chosen_row["Dport"])
            st.session_state["telemetry_dur"] = float(chosen_row["Dur"])
            st.session_state["telemetry_pkts"] = int(chosen_row["TotPkts"])
            st.session_state["telemetry_bytes"] = int(chosen_row["TotBytes"])
            st.session_state["telemetry_srate"] = float(chosen_row["sRate"])
            st.session_state["telemetry_source_description"] = (
                f"Loaded Test Sample: {selected_id} (Ground Truth: {'Attack' if chosen_row['Label'] == 1 else 'Normal'})"
            )
            st.rerun()

    with col_sample_summary:
        true_label_text = "Attack / Malicious Traffic (1)" if chosen_row["Label"] == 1 else "Normal Medical Traffic (0)"
        st.markdown(
            f"**Selected ID**: `{selected_id}` | **Ground Truth**: `{true_label_text}`"
        )
        st.caption(
            f"Sensor Snapshot: Heart Rate {chosen_row['Mean_Pulse']} bpm, SpO2 {chosen_row['Mean_SpO2']}%, "
            f"Temp {chosen_row['Mean_Temp']} C, Duration {chosen_row['Dur']}s, Packets {int(chosen_row['TotPkts'])}"
        )

# Tab 2: Attack Presets
with tab_presets:
    col_p_sel, col_p_btn = st.columns([2, 1])
    with col_p_sel:
        preset_name = st.selectbox("Select Scenario Preset:", list(ATTACK_PRESETS.keys()))
        preset_info = ATTACK_PRESETS[preset_name]
        st.write(preset_info["description"])

    with col_p_btn:
        if st.button("Apply Preset to Sliders", use_container_width=True):
            p_vitals = preset_info["vitals"]
            p_net = preset_info["network"]
            st.session_state["telemetry_pulse"] = float(p_vitals["Mean_Pulse"])
            st.session_state["telemetry_spo2"] = float(p_vitals["Mean_SpO2"])
            st.session_state["telemetry_temp"] = float(p_vitals["Mean_Temp"])
            st.session_state["telemetry_bp"] = float(p_vitals["Mean_Blood_Pressure"])
            st.session_state["telemetry_resp"] = float(p_vitals["Respiration_Rate"])
            st.session_state["telemetry_ecg"] = float(p_vitals["Mean_ECG"])
            st.session_state["telemetry_sport"] = int(p_net["Sport"])
            st.session_state["telemetry_dport"] = int(p_net["Dport"])
            st.session_state["telemetry_dur"] = float(p_net["Dur"])
            st.session_state["telemetry_pkts"] = int(p_net["TotPkts"])
            st.session_state["telemetry_bytes"] = int(p_net["TotBytes"])
            st.session_state["telemetry_srate"] = float(p_net["sRate"])
            st.session_state["telemetry_source_description"] = f"Preset: {preset_name}"
            st.rerun()

st.markdown("---")

# Active Status Info Box
st.info(f"Active Telemetry Context: {st.session_state['telemetry_source_description']}")

# Section 2: Model Selection & Sliders
col_m_select, col_empty = st.columns([1, 1])
with col_m_select:
    active_model = st.selectbox(
        "Active IDS Defense Model for Diagnosis:",
        [
            "Centralized DNN (Replication Baseline)",
            "Centralized CNN (Replication Baseline)",
            "Federated DNN (FedAvg No DP)",
            "Federated DNN + DP (Opacus Noise=1.5)"
        ]
    )

col_vitals, col_net = st.columns(2)

with col_vitals:
    st.subheader("2. Patient Biometric Sensors")

    pulse_val = st.slider(
        "Heart Rate / Pulse (bpm)",
        min_value=40.0, max_value=200.0,
        value=float(st.session_state["telemetry_pulse"]),
        step=1.0,
        key="slider_pulse"
    )

    spo2_val = st.slider(
        "Blood Oxygen Saturation (SpO2 %)",
        min_value=70.0, max_value=100.0,
        value=float(st.session_state["telemetry_spo2"]),
        step=0.5,
        key="slider_spo2"
    )

    temp_val = st.slider(
        "Body Temperature (Celsius)",
        min_value=34.0, max_value=42.5,
        value=float(st.session_state["telemetry_temp"]),
        step=0.1,
        key="slider_temp"
    )

    bp_val = st.slider(
        "Mean Blood Pressure (mmHg)",
        min_value=60.0, max_value=210.0,
        value=float(st.session_state["telemetry_bp"]),
        step=1.0,
        key="slider_bp"
    )

    resp_val = st.slider(
        "Respiration Rate (breaths/min)",
        min_value=6.0, max_value=45.0,
        value=float(st.session_state["telemetry_resp"]),
        step=1.0,
        key="slider_resp"
    )

    ecg_val = st.slider(
        "ECG Mean Signal Amplitude (mV)",
        min_value=-2.0, max_value=3.5,
        value=float(st.session_state["telemetry_ecg"]),
        step=0.05,
        key="slider_ecg"
    )

with col_net:
    st.subheader("3. Network Flow Parameters")

    sport_val = st.number_input(
        "Source Port (Sport)",
        min_value=1, max_value=65535,
        value=int(st.session_state["telemetry_sport"]),
        key="input_sport"
    )

    dport_val = st.number_input(
        "Destination Port (Dport)",
        min_value=1, max_value=65535,
        value=int(st.session_state["telemetry_dport"]),
        key="input_dport"
    )

    dur_val = st.slider(
        "Flow Duration (seconds)",
        min_value=0.01, max_value=60.0,
        value=float(st.session_state["telemetry_dur"]),
        step=0.05,
        key="slider_dur"
    )

    pkts_val = st.slider(
        "Total Flow Packets (TotPkts)",
        min_value=1, max_value=8000,
        value=int(st.session_state["telemetry_pkts"]),
        key="slider_pkts"
    )

    bytes_val = st.slider(
        "Total Bytes (TotBytes)",
        min_value=64, max_value=1000000,
        value=int(st.session_state["telemetry_bytes"]),
        step=100,
        key="slider_bytes"
    )

    srate_val = st.slider(
        "Source Packet Rate (sRate pkts/sec)",
        min_value=0.1, max_value=1000.0,
        value=float(st.session_state["telemetry_srate"]),
        step=1.0,
        key="slider_srate"
    )

st.markdown("---")

# Section 3: Intrusion Detection Diagnosis
user_vitals = {
    "Mean_Pulse": pulse_val,
    "Mean_SpO2": spo2_val,
    "Mean_Temp": temp_val,
    "Mean_Blood_Pressure": bp_val,
    "Respiration_Rate": resp_val,
    "Mean_ECG": ecg_val
}

user_network = {
    "Sport": float(sport_val),
    "Dport": float(dport_val),
    "Dur": float(dur_val),
    "TotPkts": float(pkts_val),
    "TotBytes": float(bytes_val),
    "sRate": float(srate_val)
}

feature_vec = create_feature_vector(user_vitals, user_network)
scaler = get_default_scaler()
result = predict_sample(feature_vec, model_choice=active_model, scaler=scaler)

st.subheader("Intrusion Detection Diagnosis")

col_badge, col_gauge = st.columns([1, 1])

with col_badge:
    if result["is_attack"]:
        st.markdown(
            """
            <div class="badge-alert" style="font-size: 1.25rem; padding: 0.6rem 1.4rem;">
            ALERT: SECURITY INTRUSION DETECTED
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div class="badge-normal" style="font-size: 1.25rem; padding: 0.6rem 1.4rem;">
            NORMAL: SECURE PATIENT TELEMETRY
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(f"**Classification**: `{result['prediction_name']}`")
    st.markdown(f"**Decision Confidence**: `{result['confidence']:.1f}%`")
    st.markdown(f"**Evaluated by**: `{active_model}`")
    st.caption(f"Inference source: {result['inference_source']}")

    st.markdown("#### Detected Diagnostic Factors:")
    for f in result["anomaly_factors"]:
        st.markdown(f"- {f}")

with col_gauge:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=result["attack_probability"] * 100.0,
        title={'text': "Attack Probability (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#ef4444" if result["is_attack"] else "#10b981"},
            'steps': [
                {'range': [0, 50], 'color': "#f0fdf4"},
                {'range': [50, 80], 'color': "#fffbeb"},
                {'range': [80, 100], 'color': "#fef2f2"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 3},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

st.markdown("---")

# Section 4: Batch Evaluation Tool
with st.expander("Batch Evaluation on Unseen Test Records (Verify 25 Rows at Once)"):
    st.markdown(
        "Evaluate a random slice of 25 records from the unseen test split and calculate "
        "the batch accuracy and false alarm rates."
    )

    if st.button("Run Batch Evaluation on 25 Records"):
        batch_slice = df_samples.head(25)
        batch_results = []
        correct_count = 0

        for _, row in batch_slice.iterrows():
            v_dict = {k: row[k] for k in BIOMETRIC_FEATURES if k in row}
            n_dict = {k: row[k] for k in PRIMARY_NETWORK_FEATURES if k in row}
            f_vec = create_feature_vector(v_dict, n_dict)
            pred_out = predict_sample(f_vec, model_choice=active_model, scaler=scaler)

            is_correct = pred_out["prediction_label"] == int(row["Label"])
            if is_correct:
                correct_count += 1

            batch_results.append({
                "Sample ID": row["Sample_ID"],
                "Ground Truth": "Attack (1)" if row["Label"] == 1 else "Normal (0)",
                "Model Prediction": pred_out["prediction_name"],
                "Attack Probability": f"{pred_out['attack_probability']:.1%}",
                "Status": "Correct" if is_correct else "Misclassified"
            })

        st.metric("Batch Test Accuracy", f"{(correct_count / 25):.1%}", f"{correct_count}/25 Correct")
        st.dataframe(pd.DataFrame(batch_results), use_container_width=True)
