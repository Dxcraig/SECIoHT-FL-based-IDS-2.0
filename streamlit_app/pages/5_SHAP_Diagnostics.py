"""
Explainable AI (XAI) and SHAP Feature Attribution Page.
Replicates Stage 6 (Model Explainability).
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
from src.config import ATTACK_PRESETS
from src.preprocessing import create_feature_vector, load_or_create_sample_data
from src.explainability import compute_feature_contributions, get_global_feature_importance

# Page Setup
st.set_page_config(page_title="SHAP Diagnostics | SECIoHT-FL", layout="wide")
render_sidebar()

st.title("Explainable AI (SHAP) Diagnostics")
st.markdown(
    "Deconstruct neural network decisions using feature attribution. "
    "Identify which biometric deviations or network markers trigger intrusion alerts."
)

st.markdown("---")

tab_global, tab_local = st.tabs(["Global Feature Importance (Stage 6)", "Local Prediction Waterfall"])

with tab_global:
    st.subheader("Global Feature Importance Ranking")
    st.markdown(
        """
        Global SHAP importance identifies which of the 36 features exert the greatest overall influence 
        across the entire test evaluation set.
        """
    )

    df_global = get_global_feature_importance()

    col_g1, col_g2 = st.columns([3, 2])

    with col_g1:
        fig_global = px.bar(
            df_global,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Category",
            title="Top 10 Most Influential Features (MDPI Electronics 2025 Alignment)",
            color_discrete_map={
                "Network Flow": "#3b82f6",
                "Biometric Vital": "#10b981"
            }
        )
        fig_global.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_global, use_container_width=True)

    with col_g2:
        st.markdown("#### Alignment with Published Paper Findings")
        st.markdown(
            """
            In Mosaiyebzadeh et al. (2025), the authors discovered that:
            1. **Flow Duration (`Dur`)**: The single highest-ranked predictor. Attackers cannot easily mask abnormal flow persistence.
            2. **Biometric Vitals (`Mean_Pulse`, `Mean_SpO2`)**: Physiological features carry high importance weights, confirming that fusing medical sensors with flow telemetry significantly boosts detection capability.
            3. **Source Port (`Sport`)**: Key indicator for identifying rogue edge listeners (ports 8080, 3128, 4444).
            """
        )

with tab_local:
    st.subheader("Local SHAP Attribution for Specific Scenarios")
    st.markdown(
        "Select a scenario to inspect the exact feature-by-feature forces pushing the model toward an Attack or Normal verdict."
    )

    selected_scenario = st.selectbox(
        "Choose Telemetry Scenario to Explain:",
        list(ATTACK_PRESETS.keys())
    )

    p_data = ATTACK_PRESETS[selected_scenario]
    feature_vec = create_feature_vector(p_data["vitals"], p_data["network"])
    is_attack = p_data["label"] == 1

    df_contrib = compute_feature_contributions(feature_vec, is_attack)

    st.markdown(f"**Diagnostic Summary for: {selected_scenario}**")
    
    top_10_contrib = df_contrib.head(10)

    fig_waterfall = go.Figure(go.Bar(
        x=top_10_contrib["SHAP_Value"],
        y=top_10_contrib["Feature"],
        orientation="h",
        marker=dict(
            color=["#ef4444" if val > 0 else "#10b981" for val in top_10_contrib["SHAP_Value"]]
        )
    ))
    fig_waterfall.update_layout(
        title="Top 10 Feature Contributions (Red = Pushes Toward Attack, Green = Pushes Toward Normal)",
        xaxis_title="Attribution Impact Score",
        yaxis=dict(autorange="reversed"),
        height=400
    )
    st.plotly_chart(fig_waterfall, use_container_width=True)

    st.markdown("#### Tabular Feature Attribution Values")
    st.dataframe(df_contrib.head(12), use_container_width=True)
