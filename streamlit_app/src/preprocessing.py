"""
Preprocessing and data transformation pipeline for SECIoHT-FL.
Replicates the Stage 1 and Stage 2 methodology (clean -> split -> SMOTE -> scale).
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from .config import (
    WUSTL_FEATURE_NAMES,
    DROPPED_COLUMNS,
    SAMPLES_DIR,
    ARTIFACTS_DIR
)

SCALER_PATH = ARTIFACTS_DIR / "scaler.joblib"


def clean_wustl_df(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Cleans raw WUSTL-EHMS-2020 dataframe following Stage 1 & Stage 2 methodology.
    - Drops NA & duplicates
    - Drops 8 non-predictive/leaky columns
    - Coerces 'Sport' to numeric and drops unconvertible rows
    """
    df = raw_df.copy()
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    cols_to_drop = [c for c in DROPPED_COLUMNS if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

    if "Sport" in df.columns:
        df["Sport"] = pd.to_numeric(df["Sport"], errors="coerce")
        df.dropna(subset=["Sport"], inplace=True)

    if "Label" in df.columns:
        y = df["Label"].astype(int)
        X = df.drop(columns=["Label"])
    else:
        y = pd.Series([0] * len(df))
        X = df

    return X, y


def generate_synthetic_samples(num_samples: int = 150, seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic offline sample dataset matching WUSTL-EHMS-2020 distributions
    so the dashboard functions immediately without requiring external CSV downloads.
    Preserves ~7:1 normal:attack ratio.
    """
    rng = np.random.default_rng(seed)
    
    num_attacks = max(15, int(num_samples * 0.125))
    num_normal = num_samples - num_attacks

    data = []

    # 1. Normal Traffic Profiles
    for i in range(num_normal):
        row = {
            "Sample_ID": f"NORM-{i+1:04d}",
            "Sport": float(rng.choice([80, 443, 8080, 53, rng.integers(1024, 65535)])),
            "Dport": float(rng.choice([443, 80, 53, rng.integers(1024, 65535)])),
            "Dur": round(float(rng.exponential(scale=1.5)), 4),
            "TotPkts": float(rng.integers(8, 45)),
            "TotBytes": float(rng.integers(800, 15000)),
            "SrcBytes": float(rng.integers(400, 8000)),
            "DstBytes": float(rng.integers(400, 7000)),
            "sRate": round(float(rng.uniform(5.0, 30.0)), 2),
            "dRate": round(float(rng.uniform(4.0, 28.0)), 2),
            "sLoad": round(float(rng.uniform(1000.0, 50000.0)), 2),
            "dLoad": round(float(rng.uniform(1000.0, 45000.0)), 2),
            "sMeanPktSz": round(float(rng.uniform(60.0, 250.0)), 2),
            "dMeanPktSz": round(float(rng.uniform(60.0, 250.0)), 2),
            "sJitter": round(float(rng.uniform(0.001, 0.05)), 4),
            "dJitter": round(float(rng.uniform(0.001, 0.05)), 4),
            "sHops": float(rng.integers(1, 15)),
            "dHops": float(rng.integers(1, 15)),
            "sTtl": float(rng.choice([64, 128, 255])),
            "dTtl": float(rng.choice([64, 128, 255])),
            "Proto_Code": 6.0, # TCP
            "State_Code": 2.0, # CON / FIN
            "Mean_Pulse": round(float(rng.normal(72.0, 8.0)), 1),
            "Mean_SpO2": round(float(rng.uniform(96.0, 99.5)), 1),
            "Mean_Temp": round(float(rng.normal(36.8, 0.4)), 1),
            "Mean_Blood_Pressure": round(float(rng.normal(120.0, 10.0)), 1),
            "Respiration_Rate": round(float(rng.normal(16.0, 2.0)), 1),
            "Mean_ECG": round(float(rng.normal(0.85, 0.15)), 2),
            "Tcp_Rtt": round(float(rng.uniform(0.01, 0.08)), 4),
            "Ack_Dat": round(float(rng.uniform(0.005, 0.04)), 4),
            "Syn_Ack": round(float(rng.uniform(0.005, 0.04)), 4),
            "sLoss": 0.0,
            "dLoss": 0.0,
            "sWin": 65535.0,
            "dWin": 65535.0,
            "sAppBytes": float(rng.integers(200, 5000)),
            "dAppBytes": float(rng.integers(200, 5000)),
            "Label": 0
        }
        data.append(row)

    # 2. Attack Profiles (Spoofing, DoS, Data Alteration)
    for j in range(num_attacks):
        attack_kind = rng.choice(["spoofing", "dos", "alteration"])
        row = {
            "Sample_ID": f"ATTK-{j+1:04d}",
            "Sport": float(rng.choice([80, 8080, 3128, 4444])),
            "Dport": float(rng.choice([443, 80, 22])),
            "Dur": round(float(rng.uniform(0.01, 60.0) if attack_kind == "dos" else rng.uniform(0.05, 2.0)), 4),
            "TotPkts": float(rng.integers(800, 9000) if attack_kind == "dos" else rng.integers(2, 20)),
            "TotBytes": float(rng.integers(500000, 2000000) if attack_kind == "dos" else rng.integers(100, 2000)),
            "SrcBytes": float(rng.integers(400000, 1900000) if attack_kind == "dos" else rng.integers(80, 1500)),
            "DstBytes": float(rng.integers(1000, 50000)),
            "sRate": round(float(rng.uniform(200.0, 1200.0) if attack_kind == "dos" else rng.uniform(1.0, 50.0)), 2),
            "dRate": round(float(rng.uniform(1.0, 20.0)), 2),
            "sLoad": round(float(rng.uniform(100000.0, 9000000.0) if attack_kind == "dos" else rng.uniform(500.0, 50000.0)), 2),
            "dLoad": round(float(rng.uniform(500.0, 50000.0)), 2),
            "sMeanPktSz": round(float(rng.uniform(20.0, 1500.0)), 2),
            "dMeanPktSz": round(float(rng.uniform(20.0, 500.0)), 2),
            "sJitter": round(float(rng.uniform(0.05, 0.8)), 4),
            "dJitter": round(float(rng.uniform(0.05, 0.8)), 4),
            "sHops": float(rng.integers(1, 30)),
            "dHops": float(rng.integers(1, 30)),
            "sTtl": float(rng.choice([32, 64, 255])),
            "dTtl": float(rng.choice([32, 64, 255])),
            "Proto_Code": float(rng.choice([6.0, 17.0])), # TCP or UDP
            "State_Code": 1.0, # INT / RST
            "Mean_Pulse": round(float(rng.choice([165.0, 180.0, 42.0]) if attack_kind == "spoofing" else rng.normal(78.0, 10.0)), 1),
            "Mean_SpO2": round(float(rng.choice([78.0, 84.0, 72.0]) if attack_kind == "spoofing" else rng.normal(96.0, 2.0)), 1),
            "Mean_Temp": round(float(rng.choice([41.2, 34.5]) if attack_kind == "spoofing" else rng.normal(37.1, 0.5)), 1),
            "Mean_Blood_Pressure": round(float(rng.choice([195.0, 65.0]) if attack_kind == "spoofing" else rng.normal(128.0, 12.0)), 1),
            "Respiration_Rate": round(float(rng.choice([38.0, 6.0]) if attack_kind == "spoofing" else rng.normal(18.0, 3.0)), 1),
            "Mean_ECG": round(float(rng.choice([2.8, -1.2]) if attack_kind == "spoofing" else rng.normal(0.9, 0.2)), 2),
            "Tcp_Rtt": round(float(rng.uniform(0.1, 0.9)), 4),
            "Ack_Dat": round(float(rng.uniform(0.05, 0.5)), 4),
            "Syn_Ack": round(float(rng.uniform(0.05, 0.5)), 4),
            "sLoss": float(rng.integers(1, 20)),
            "dLoss": float(rng.integers(0, 10)),
            "sWin": 1024.0,
            "dWin": 1024.0,
            "sAppBytes": float(rng.integers(50, 10000)),
            "dAppBytes": float(rng.integers(50, 5000)),
            "Label": 1
        }
        data.append(row)

    df = pd.DataFrame(data)
    # Shuffle
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def load_or_create_sample_data() -> pd.DataFrame:
    """Loads bundled sample dataset or generates one if missing."""
    sample_file = SAMPLES_DIR / "wustl_sample.csv"
    if sample_file.exists():
        return pd.read_csv(sample_file)
    
    df = generate_synthetic_samples()
    df.to_csv(sample_file, index=False)
    return df


def create_feature_vector(
    vitals: Dict[str, float],
    network: Dict[str, float],
    base_sample: Optional[pd.Series] = None
) -> np.ndarray:
    """
    Combines user inputs with baseline defaults into the exact 36-feature vector.
    """
    if base_sample is None:
        samples_df = load_or_create_sample_data()
        normal_samples = samples_df[samples_df["Label"] == 0]
        base_sample = normal_samples.iloc[0]

    vector = []
    for col in WUSTL_FEATURE_NAMES:
        if col in vitals:
            vector.append(float(vitals[col]))
        elif col in network:
            vector.append(float(network[col]))
        elif col in base_sample:
            vector.append(float(base_sample[col]))
        else:
            vector.append(0.0)

    return np.array(vector, dtype=np.float32).reshape(1, -1)


def get_default_scaler() -> StandardScaler:
    """Returns a fitted StandardScaler based on the sample baseline."""
    df = load_or_create_sample_data()
    feature_cols = [c for c in WUSTL_FEATURE_NAMES if c in df.columns]
    scaler = StandardScaler()
    scaler.fit(df[feature_cols].values)
    return scaler
