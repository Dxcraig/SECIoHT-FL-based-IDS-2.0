"""
Inference engine and attack diagnostic evaluator for SECIoHT-FL.
Handles live predictions, model loading, and preset scenario evaluation.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from pathlib import Path
from .config import (
    MODELS_DIR,
    WUSTL_FEATURE_NAMES,
    ATTACK_PRESETS,
    CLASS_NAMES
)
from .artifacts import MODEL_FILES, model_path
from .models import DNN, CNN, softmax, TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch


def load_model_weights(model_type: str = "dnn", filename: str = "best_dnn.pt"):
    """Loads PyTorch model if checkpoint exists, otherwise returns unweighted instance."""
    if not TORCH_AVAILABLE:
        return None

    model = DNN(36) if model_type == "dnn" else CNN(36)
    model_path = MODELS_DIR / filename
    if model_path.exists():
        try:
            state_dict = torch.load(model_path, map_location=torch.device("cpu"))
            model.load_state_dict(state_dict)
            model.eval()
        except Exception:
            pass
    return model


def _checkpoint_for_choice(model_choice: str):
    """Resolve user-selected model labels to the notebook-produced checkpoints."""
    if "CNN" in model_choice:
        return None
    if "0.5" in model_choice:
        return "Federated DNN + DP (noise=0.5)"
    if "1.5" in model_choice:
        return "Federated DNN + DP (noise=1.5)"
    if "Federated" in model_choice:
        return "Federated DNN"
    if "Centralized" in model_choice:
        return "Centralized DNN"
    return None



def predict_sample(
    feature_vector: np.ndarray,
    model_choice: str = "Centralized DNN (Replication Baseline)",
    scaler: Any = None
) -> Dict[str, Any]:
    """
    Executes inference on a 36-feature telemetry vector.
    Returns predicted class, probabilities, anomaly indicators, and explanation hints.
    """
    # 1. Scale feature vector if scaler provided
    scaled_vector = feature_vector.copy()
    if scaler is not None:
        try:
            scaled_vector = scaler.transform(feature_vector)
        except Exception:
            pass

    # 2. Extract key domain signals for heuristic verification
    # (Matches top features identified by Mosaiyebzadeh et al.: Flow Duration, Sport, Pulse, Temp)
    sport_idx = WUSTL_FEATURE_NAMES.index("Sport")
    dport_idx = WUSTL_FEATURE_NAMES.index("Dport")
    dur_idx = WUSTL_FEATURE_NAMES.index("Dur")
    pkts_idx = WUSTL_FEATURE_NAMES.index("TotPkts")
    pulse_idx = WUSTL_FEATURE_NAMES.index("Mean_Pulse")
    spo2_idx = WUSTL_FEATURE_NAMES.index("Mean_SpO2")
    temp_idx = WUSTL_FEATURE_NAMES.index("Mean_Temp")
    bp_idx = WUSTL_FEATURE_NAMES.index("Mean_Blood_Pressure")

    raw_sport = feature_vector[0, sport_idx]
    raw_dur = feature_vector[0, dur_idx]
    raw_pkts = feature_vector[0, pkts_idx]
    raw_pulse = feature_vector[0, pulse_idx]
    raw_spo2 = feature_vector[0, spo2_idx]
    raw_temp = feature_vector[0, temp_idx]
    raw_bp = feature_vector[0, bp_idx]

    anomaly_score = 0.0
    anomaly_reasons = []

    # Check biometric anomalies
    if raw_pulse > 140.0 or raw_pulse < 48.0:
        anomaly_score += 0.35
        anomaly_reasons.append(f"Abnormal Pulse Rate ({raw_pulse:.1f} bpm)")
    if raw_spo2 < 90.0:
        anomaly_score += 0.35
        anomaly_reasons.append(f"Hypoxia / SpO2 Depletion ({raw_spo2:.1f}%)")
    if raw_temp > 39.5 or raw_temp < 35.0:
        anomaly_score += 0.25
        anomaly_reasons.append(f"Extreme Temperature deviation ({raw_temp:.1f} °C)")
    if raw_bp > 165.0:
        anomaly_score += 0.20
        anomaly_reasons.append(f"Severe Hypertension reading ({raw_bp:.1f} mmHg)")

    # Check network anomalies
    if raw_pkts > 500.0 or (raw_dur > 0 and (raw_pkts / raw_dur) > 200.0):
        anomaly_score += 0.40
        anomaly_reasons.append(f"Packet Flood / DoS Storm ({raw_pkts:.0f} pkts in {raw_dur:.2f}s)")
    if raw_sport in [8080, 3128, 4444, 1337]:
        anomaly_score += 0.20
        anomaly_reasons.append(f"Suspicious Edge Service Port ({int(raw_sport)})")

    # 3. Model inference: prefer the exact checkpoint produced by the notebook.
    p_attack = 0.0
    inference_source = "No trained model artifact available"
    checkpoint_label = _checkpoint_for_choice(model_choice)
    checkpoint = model_path(checkpoint_label) if checkpoint_label else None

    if checkpoint and checkpoint.exists() and TORCH_AVAILABLE:
        try:
            model = DNN(36)
            state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            with torch.no_grad():
                tensor_in = torch.tensor(scaled_vector, dtype=torch.float32)
                probs = torch.softmax(model(tensor_in), dim=1).numpy()[0]
            p_attack = float(probs[1])
            inference_source = f"Notebook checkpoint: {checkpoint.name}"
        except Exception:
            inference_source = f"Checkpoint failed to load: {checkpoint.name}"

    if inference_source == "No trained model artifact available" and "CNN" not in model_choice:
        mlp_path = MODELS_DIR / "baseline_mlp.joblib"
        if mlp_path.exists():
            try:
                import joblib
                mlp_model = joblib.load(mlp_path)
                p_attack = float(mlp_model.predict_proba(scaled_vector)[0][1])
                inference_source = "Bundled baseline MLP fallback (not the notebook checkpoint)"
            except Exception:
                inference_source = "Bundled baseline model failed to load"

    if inference_source == "No trained model artifact available" and "CNN" not in model_choice and TORCH_AVAILABLE and (MODELS_DIR / "best_dnn.pt").exists():
        try:
            model = load_model_weights("dnn", "best_dnn.pt")
            with torch.no_grad():
                tensor_in = torch.tensor(scaled_vector, dtype=torch.float32)
                logits = model(tensor_in).numpy()
                probs = softmax(logits)[0]
                p_attack = float(probs[1])
            inference_source = "Legacy best_dnn.pt fallback"
        except Exception:
            inference_source = "Legacy model failed to load"

    if inference_source.endswith("failed to load"):
        p_attack = min(0.99, max(0.01, anomaly_score))
    elif inference_source == "No trained model artifact available":
        p_attack = min(0.99, max(0.01, anomaly_score))

    p_normal = 1.0 - p_attack
    pred_label = 1 if p_attack >= 0.50 else 0

    return {
        "prediction_label": pred_label,
        "prediction_name": CLASS_NAMES[pred_label],
        "attack_probability": p_attack,
        "normal_probability": p_normal,
        "confidence": max(p_attack, p_normal) * 100.0,
        "is_attack": pred_label == 1,
        "inference_source": inference_source,
        "anomaly_factors": anomaly_reasons if anomaly_reasons else ["Telemetry within normal medical & network bounds"]
    }
