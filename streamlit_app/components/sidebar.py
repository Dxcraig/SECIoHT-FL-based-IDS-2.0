"""
Reusable sidebar navigation and status component for SECIoHT-FL Streamlit app.
"""
import streamlit as st


def render_sidebar():
    """Renders professional sidebar widget with system status and research context."""
    with st.sidebar:
        st.markdown("## SECIoHT-FL IDS")
        st.caption("Privacy-Preserving Federated IDS for IoHT Devices")
        st.markdown("---")

        st.markdown("### System Telemetry")
        st.markdown(
            """
            - **Federated Nodes**: 3 Simulated Edge Hospitals
            - **Privacy Budget**: ε <= 0.44 (Opacus DP-SGD)
            - **Dataset**: WUSTL-EHMS-2020 (36 features)
            - **Active Pipeline**: Leak-Free SMOTE (Train-Only)
            """
        )

        st.markdown("---")
        st.markdown("### Research Reference")
        st.markdown(
            """
            *Electronics 2025, 14, 67*  
            **Authors**: Mosaiyebzadeh et al.
            """
        )

        st.markdown("---")
        st.markdown("### Navigation")
        st.markdown(
            """
            - **Home**: Project Overview
            - **1. Data Explorer**: WUSTL vs ECU Analysis
            - **2. FL & Privacy Lab**: FedAvg + DP Simulator
            - **3. Benchmark Arena**: Centralized vs FL Evaluation
            - **4. IDS Predictor**: Edge Testbed
            - **5. SHAP Diagnostics**: Explainable AI
            """
        )
