"""
Federated Learning and Differential Privacy Simulator Lab.
Replicates Stage 3 (FedAvg) and Stage 4 (Opacus DP-SGD).
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import time

# Ensure local packages are importable
APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from components.sidebar import render_sidebar
from src.federated import simulate_fl_rounds, calculate_dp_epsilon

# Page Setup
st.set_page_config(page_title="FL and Privacy Lab | SECIoHT-FL", layout="wide")
render_sidebar()

st.title("Federated Learning and Differential Privacy Lab")
st.markdown(
    "Simulate decentralized model aggregation across edge hospital clients and analyze the "
    "privacy-utility trade-off using Opacus DP-SGD."
)

st.markdown("---")

col_ctrl, col_info = st.columns([1, 2])

with col_ctrl:
    st.subheader("Simulation Controls")
    
    num_clients = st.slider("Number of Hospital Edge Clients", min_value=2, max_value=5, value=3)
    num_rounds = st.slider("Communication Rounds", min_value=2, max_value=20, value=8)
    local_epochs = st.slider("Local Epochs per Round", min_value=1, max_value=5, value=2)
    
    noise_choice = st.selectbox(
        "Differential Privacy Noise Multiplier (sigma)",
        [
            "1.5 (Paper Recommended - Strong Privacy)",
            "0.5 (Paper Degraded - Higher Variance)",
            "0.0 (Plain FedAvg - No Differential Privacy)"
        ]
    )

    if "1.5" in noise_choice:
        noise_val = 1.5
    elif "0.5" in noise_choice:
        noise_val = 0.5
    else:
        noise_val = 0.0

    st.markdown(
        """
        - **Target Delta**: `1e-4`
        - **Clipping Norm (C)**: `1e-4` (Paper stated)
        - **Algorithm**: DP-SGD (Gaussian Mechanism)
        """
    )

    run_sim = st.button("Run Federated Simulation", type="primary", use_container_width=True)

with col_info:
    st.subheader("Methodology: Local DP and Persistent Accounting")
    st.markdown(
        """
        In Stage 4, each simulated hospital injects calibrated Gaussian noise into its gradients 
        **locally on the device** before sending weights to the server.
        
        <div class="info-box">
        <b>Critical Privacy Accounting Guard:</b><br/>
        If an Opacus PrivacyEngine is re-instantiated fresh every round, its internal Renyi Differential 
        Privacy (RDP) accountant silently resets to zero. This makes reported privacy look artificially strong.
        Our <code>DPClient</code> architecture is <b>persistent</b>: model weights update from global averaging, 
        but the privacy engine continues accumulating total epsilon spend across all communication rounds.
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# Simulation Execution & Results
if run_sim:
    st.subheader(f"Simulation in Progress: {num_clients} Clients across {num_rounds} Rounds")
    progress_bar = st.progress(0)
    status_text = st.empty()

    history_records = []
    
    gen = simulate_fl_rounds(
        num_clients=num_clients,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        noise_multiplier=noise_val
    )

    chart_placeholder = st.empty()

    for idx, step in enumerate(gen):
        time.sleep(0.12)
        history_records.append({
            "Round": step["round"],
            "Global Accuracy (%)": step["global_accuracy"],
            "Global Loss": step["global_loss"],
            "Cumulative Epsilon": step["cumulative_epsilon"] if noise_val > 0 else 0.0
        })

        progress_bar.progress((idx + 1) / num_rounds)
        status_text.text(
            f"Completed Round {step['round']}/{num_rounds} | "
            f"Accuracy: {step['global_accuracy']}% | "
            f"Loss: {step['global_loss']} | "
            f"Cumulative Epsilon: {step['cumulative_epsilon'] if noise_val > 0 else 'N/A'}"
        )

    progress_bar.empty()
    status_text.empty()
    df_history = pd.DataFrame(history_records)

    st.success(f"Simulation completed successfully across {num_rounds} rounds.")

    # KPI Summary Cards
    c1, c2, c3 = st.columns(3)
    final_acc = df_history["Global Accuracy (%)"].iloc[-1]
    final_loss = df_history["Global Loss"].iloc[-1]
    final_eps = df_history["Cumulative Epsilon"].iloc[-1]

    with c1:
        st.metric("Final Converged Accuracy", f"{final_acc:.2f}%", f"+{(final_acc - 52.0):.1f}% from baseline")
    with c2:
        st.metric("Final Model Loss", f"{final_loss:.4f}")
    with c3:
        st.metric(
            "Final Privacy Budget Spent (Epsilon)",
            f"{final_eps:.2f}" if noise_val > 0 else "Unbounded (No DP)",
            "Lower is more private" if noise_val > 0 else None
        )

    # Plots
    col_p1, col_p2 = st.columns(2)

    with col_p1:
        fig_acc = px.line(
            df_history,
            x="Round",
            y="Global Accuracy (%)",
            title="Global Model Accuracy Progression across Rounds",
            markers=True,
            line_shape="spline"
        )
        st.plotly_chart(fig_acc, use_container_width=True)

    with col_p2:
        if noise_val > 0:
            fig_eps = px.line(
                df_history,
                x="Round",
                y="Cumulative Epsilon",
                title="Privacy Budget Spent Accumulation (Epsilon vs Round)",
                markers=True,
                color_discrete_sequence=["#f59e0b"]
            )
            st.plotly_chart(fig_eps, use_container_width=True)
        else:
            fig_loss = px.line(
                df_history,
                x="Round",
                y="Global Loss",
                title="Global Training Loss Progression (No DP)",
                markers=True,
                color_discrete_sequence=["#ef4444"]
            )
            st.plotly_chart(fig_loss, use_container_width=True)

else:
    # Default visual view before running
    st.info("Click 'Run Federated Simulation' above to trigger real-time multi-client FedAvg and Opacus DP-SGD aggregation.")
    
    # Pre-calculated standard comparison curve
    demo_rounds = list(range(1, 11))
    demo_df = pd.DataFrame({
        "Round": demo_rounds,
        "No DP Accuracy (%)": [58.2, 67.4, 74.1, 79.8, 83.2, 85.1, 86.0, 86.8, 87.2, 87.5],
        "DP Noise=1.5 Accuracy (%)": [56.1, 64.2, 70.8, 75.3, 78.4, 80.9, 82.1, 82.8, 83.2, 83.5],
        "DP Noise=0.5 Accuracy (%)": [54.0, 61.2, 66.8, 70.5, 73.8, 75.6, 76.8, 77.4, 77.9, 78.1]
    })
    
    fig_comp = px.line(
        demo_df,
        x="Round",
        y=["No DP Accuracy (%)", "DP Noise=1.5 Accuracy (%)", "DP Noise=0.5 Accuracy (%)"],
        title="Privacy-Utility Trade-off: Accuracy Curves under Varying Opacus Noise",
        labels={"value": "Test Accuracy (%)", "variable": "Privacy Setting"}
    )
    st.plotly_chart(fig_comp, use_container_width=True)
