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
from src.artifacts import result_path
from src.federated import simulate_fl_rounds

# Page Setup
st.set_page_config(page_title="FL and Privacy Lab | SECIoHT-FL", layout="wide")
render_sidebar()

st.title("Federated Learning and Differential Privacy Lab")
st.markdown(
    "Explore decentralized model aggregation across edge hospital clients and analyze the "
    "privacy-utility trade-off using Opacus DP-SGD across 100 communication rounds."
)

st.markdown("---")

# Tab Selector: Recorded Notebook Artifacts vs Live Simulation
tab_recorded, tab_sim = st.tabs([
    "Recorded 100-Round Training History (Notebook Artifacts)",
    "Interactive Parameter Explorer (Live Simulation)"
])

# =========================================================================
# TAB 1: RECORDED NOTEBOOK ARTIFACTS
# =========================================================================
with tab_recorded:
    stage3_log_path = result_path("stage3").with_name("stage3_round_log.csv")
    stage4_log_path = result_path("stage4").with_name("stage4_round_log.csv")

    col_select_log, col_meta = st.columns([1, 2])

    with col_select_log:
        available_logs = {}
        if stage4_log_path.exists():
            available_logs["Stage 4: Federated + DP (Noise=1.5, 100 by 1)"] = stage4_log_path
        if stage3_log_path.exists():
            available_logs["Stage 3: Plain Federated FedAvg (No DP, 100 by 1)"] = stage3_log_path

        if not available_logs:
            st.warning("No round log CSV files found in artifacts/results/.")
            selected_log_name = None
            log_df = None
        else:
            selected_log_name = st.selectbox(
                "Select Training Run to Inspect:",
                list(available_logs.keys())
            )
            log_path = available_logs[selected_log_name]
            log_df = pd.read_csv(log_path)

    with col_meta:
        if "Stage 4" in str(selected_log_name):
            st.markdown(
                """
                <div class="info-box">
                <b>Stage 4 Run Context (100 Communication Rounds x 1 Local Epoch):</b><br/>
                Opacus DP-SGD with Gaussian noise multiplier <code>sigma = 1.5</code>, clipping norm <code>C = 1e-4</code>,
                and persistent privacy accounting across 3 simulated edge hospital nodes.
                </div>
                """,
                unsafe_allow_html=True
            )
        elif "Stage 3" in str(selected_log_name):
            st.markdown(
                """
                <div class="info-box">
                <b>Stage 3 Run Context (100 Communication Rounds x 1 Local Epoch):</b><br/>
                Decentralized Federated Averaging (FedAvg) without differential privacy noise.
                3 edge hospital nodes each train locally and exchange weight updates with the server.
                </div>
                """,
                unsafe_allow_html=True
            )

    if log_df is not None:
        st.markdown("---")
        final_row = log_df.iloc[-1]
        final_acc = float(final_row["test_accuracy"]) * 100.0 if final_row["test_accuracy"] <= 1.0 else float(final_row["test_accuracy"])
        final_loss = float(final_row["loss"]) if "loss" in final_row else 0.0
        final_eps = float(final_row["epsilon"]) if "epsilon" in final_row and pd.notna(final_row["epsilon"]) else None

        # KPI Summary Cards
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Final Test Accuracy", f"{final_acc:.2f}%", "Unseen Test Records")
        with k2:
            st.metric("Final Model Loss", f"{final_loss:.4f}", "Cross-Entropy")
        with k3:
            if final_eps is not None:
                st.metric("Cumulative Privacy Spent (Epsilon)", f"ε = {final_eps:.4f}", f"Target Delta: 1e-4")
            else:
                st.metric("Privacy Guarantee", "Unbounded (No DP)", "Plain FedAvg")
        with k4:
            st.metric("Total Rounds Replayed", f"{len(log_df)} Rounds", "100 by 1 Pipeline")

        st.markdown("---")

        # Interactive Visual Progression Plots
        col_plot1, col_plot2 = st.columns(2)

        with col_plot1:
            # Accuracy progression
            fig_acc = px.line(
                log_df,
                x="round",
                y="test_accuracy",
                title=f"Global Model Accuracy Progression across {len(log_df)} Rounds",
                markers=True,
                labels={"round": "Communication Round", "test_accuracy": "Test Accuracy (0 - 1.0)"},
                line_shape="spline"
            )
            fig_acc.update_traces(line_color="#10b981" if "Stage 3" in str(selected_log_name) else "#3b82f6")
            fig_acc.update_layout(height=380)
            st.plotly_chart(fig_acc, use_container_width=True)

        with col_plot2:
            if "epsilon" in log_df.columns and final_eps is not None:
                # Epsilon accumulation curve
                fig_eps = px.line(
                    log_df,
                    x="round",
                    y="epsilon",
                    title="Cumulative Privacy Budget Spent (Epsilon vs Round)",
                    markers=True,
                    labels={"round": "Communication Round", "epsilon": "Cumulative Epsilon (ε)"},
                    line_shape="spline"
                )
                fig_eps.update_traces(line_color="#f59e0b")
                fig_eps.update_layout(height=380)
                st.plotly_chart(fig_eps, use_container_width=True)
            else:
                # Loss progression
                fig_loss = px.line(
                    log_df,
                    x="round",
                    y="loss",
                    title="Global Training Loss Progression (No DP)",
                    markers=True,
                    labels={"round": "Communication Round", "loss": "Cross-Entropy Loss"},
                    line_shape="spline"
                )
                fig_loss.update_traces(line_color="#ef4444")
                fig_loss.update_layout(height=380)
                st.plotly_chart(fig_loss, use_container_width=True)

        st.markdown("---")

        # Round Inspector Tool
        st.subheader("Step-by-Step Round Inspector")
        st.markdown("Scrub the slider below to inspect model telemetry at any specific communication round:")

        selected_round = st.slider(
            "Select Communication Round to Inspect:",
            min_value=1,
            max_value=int(log_df["round"].max()),
            value=int(log_df["round"].max()),
            key="round_scrubber"
        )

        scrub_row = log_df[log_df["round"] == selected_round].iloc[0]
        scrub_acc = float(scrub_row["test_accuracy"]) * 100.0 if scrub_row["test_accuracy"] <= 1.0 else float(scrub_row["test_accuracy"])
        scrub_loss = float(scrub_row["loss"]) if "loss" in scrub_row else 0.0
        scrub_eps = float(scrub_row["epsilon"]) if "epsilon" in scrub_row and pd.notna(scrub_row["epsilon"]) else None

        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.markdown(f"**Round**: `{selected_round} / {len(log_df)}`")
            st.markdown(f"**Test Accuracy**: `{scrub_acc:.2f}%`")
        with sc2:
            st.markdown(f"**Loss**: `{scrub_loss:.4f}`")
            st.markdown(f"**Client Optimization Steps**: `{selected_round * 119:,} steps`")
        with sc3:
            if scrub_eps is not None:
                st.markdown(f"**Privacy Spent**: `ε = {scrub_eps:.4f}`")
                st.caption(f"Opacus RDP accountant accumulated through round {selected_round}")
            else:
                st.markdown("**Privacy Guarantee**: `Unbounded (Plain FedAvg)`")

# =========================================================================
# TAB 2: LIVE SIMULATION & PARAMETER EXPLORER
# =========================================================================
with tab_sim:
    col_ctrl, col_info = st.columns([1, 2])

    with col_ctrl:
        st.subheader("Simulation Controls")
        
        sim_clients = st.slider("Simulated Hospital Edge Nodes", min_value=2, max_value=5, value=3)
        sim_rounds = st.slider("Communication Rounds", min_value=5, max_value=100, value=20)
        sim_local_epochs = st.slider("Local Epochs per Round", min_value=1, max_value=20, value=1)
        
        sim_noise_choice = st.selectbox(
            "Differential Privacy Noise Multiplier (sigma)",
            [
                "1.5 (Paper Recommended - Strong Privacy)",
                "0.5 (Paper Degraded - Higher Variance)",
                "0.0 (Plain FedAvg - No Differential Privacy)"
            ]
        )

        if "1.5" in sim_noise_choice:
            sim_noise_val = 1.5
        elif "0.5" in sim_noise_choice:
            sim_noise_val = 0.5
        else:
            sim_noise_val = 0.0

        st.markdown(
            """
            - **Target Delta**: `1e-4`
            - **Clipping Norm (C)**: `1e-4`
            - **Algorithm**: FedAvg + Opacus DP-SGD
            """
        )

        start_sim = st.button("Run Live Simulation", type="primary", use_container_width=True)

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

    if start_sim:
        st.subheader(f"Simulation in Progress: {sim_clients} Clients across {sim_rounds} Rounds")
        progress_bar = st.progress(0)
        status_text = st.empty()
        sim_records = []

        gen = simulate_fl_rounds(
            num_clients=sim_clients,
            num_rounds=sim_rounds,
            local_epochs=sim_local_epochs,
            noise_multiplier=sim_noise_val
        )

        step_delay = max(0.01, min(0.06, 2.0 / sim_rounds))
        for idx, step in enumerate(gen):
            time.sleep(step_delay)
            sim_records.append({
                "round": step["round"],
                "global_accuracy": step["global_accuracy"],
                "global_loss": step["global_loss"],
                "cumulative_epsilon": step["cumulative_epsilon"] if sim_noise_val > 0 else 0.0
            })

            progress_bar.progress((idx + 1) / sim_rounds)
            status_text.text(
                f"Round {step['round']}/{sim_rounds} | "
                f"Accuracy: {step['global_accuracy']}% | "
                f"Loss: {step['global_loss']} | "
                f"Epsilon: {step['cumulative_epsilon'] if sim_noise_val > 0 else 'N/A'}"
            )

        progress_bar.empty()
        status_text.empty()
        df_sim = pd.DataFrame(sim_records)
        st.success(f"Simulation completed successfully across {sim_rounds} rounds.")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Final Converged Accuracy", f"{df_sim['global_accuracy'].iloc[-1]:.2f}%")
        with c2:
            st.metric("Final Model Loss", f"{df_sim['global_loss'].iloc[-1]:.4f}")
        with c3:
            st.metric(
                "Final Epsilon Spent",
                f"ε = {df_sim['cumulative_epsilon'].iloc[-1]:.2f}" if sim_noise_val > 0 else "Unbounded (No DP)"
            )

        col_sp1, col_sp2 = st.columns(2)
        with col_sp1:
            fig_sim_acc = px.line(
                df_sim, x="round", y="global_accuracy",
                title="Simulated Accuracy Progression",
                markers=True
            )
            st.plotly_chart(fig_sim_acc, use_container_width=True)
        with col_sp2:
            if sim_noise_val > 0:
                fig_sim_eps = px.line(
                    df_sim, x="round", y="cumulative_epsilon",
                    title="Simulated Epsilon Accumulation",
                    markers=True, color_discrete_sequence=["#f59e0b"]
                )
                st.plotly_chart(fig_sim_eps, use_container_width=True)
            else:
                fig_sim_loss = px.line(
                    df_sim, x="round", y="global_loss",
                    title="Simulated Loss Progression",
                    markers=True, color_discrete_sequence=["#ef4444"]
                )
                st.plotly_chart(fig_sim_loss, use_container_width=True)
