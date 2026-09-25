"""Artifact discovery for the Streamlit demo.

The notebook is the source of truth. Streamlit only loads artifacts produced by
that notebook and never retrains models or invents evaluation results.
"""
from pathlib import Path
from typing import Dict

from .config import ARTIFACTS_DIR, MODELS_DIR, RESULTS_DIR

MODEL_FILES: Dict[str, str] = {
    "Centralized DNN": "dnn_centralized_wustl.pt",
    "Federated DNN": "dnn_federated_wustl.pt",
    "Federated DNN + DP (noise=1.5)": "dnn_federated_dp_noise1_5_wustl.pt",
    "Federated DNN + DP (noise=0.5)": "dnn_federated_dp_noise0_5_wustl.pt",
}

RESULT_FILES: Dict[str, str] = {
    "stage3": "stage3_federated_vs_centralized_wustl.csv",
    "stage4": "stage4_full_comparison_wustl.csv",
    "stage5": "stage5_benchmark_evaluation.csv",
    "stage6_importance": "stage6_feature_importance.csv",
}


def model_path(label: str) -> Path:
    return MODELS_DIR / MODEL_FILES[label]


def result_path(name: str) -> Path:
    return RESULTS_DIR / RESULT_FILES[name]


def artifact_status() -> Dict[str, bool]:
    """Return availability without treating missing artifacts as results."""
    return {
        **{label: model_path(label).exists() for label in MODEL_FILES},
        **{name: result_path(name).exists() for name in RESULT_FILES},
        "scaler": (ARTIFACTS_DIR / "scaler.joblib").exists(),
        "sample_data": (ARTIFACTS_DIR / "samples" / "wustl_sample.csv").exists(),
    }
