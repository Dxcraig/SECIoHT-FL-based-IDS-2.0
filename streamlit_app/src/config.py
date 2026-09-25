"""
Configuration, feature definitions, and default hyperparameters for SECIoHT-FL.
"""
from pathlib import Path

# Paths
APP_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = APP_ROOT / "src"
ARTIFACTS_DIR = APP_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
SAMPLES_DIR = ARTIFACTS_DIR / "samples"
RESULTS_DIR = ARTIFACTS_DIR / "results"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Datasets & Features
INPUT_DIM = 36
NUM_CLASSES = 2
CLASS_NAMES = {0: "Normal Traffic", 1: "Security Intrusion / Attack"}

# WUSTL-EHMS-2020: 8 columns removed during Stage 1 data cleaning
DROPPED_COLUMNS = [
    "Dir", "Flgs", "SrcAddr", "DstAddr",
    "SrcMac", "DstMac", "Packet_num", "Attack Category"
]

# Primary Biometric Features (IoHT Sensors)
BIOMETRIC_FEATURES = [
    "Mean_Pulse",
    "Mean_SpO2",
    "Mean_Temp",
    "Mean_Blood_Pressure",
    "Respiration_Rate",
    "Mean_ECG"
]

# Primary Network Features
PRIMARY_NETWORK_FEATURES = [
    "Sport",
    "Dport",
    "Dur",
    "TotPkts",
    "TotBytes",
    "SrcBytes",
    "DstBytes",
    "sRate",
    "dRate",
    "sLoad",
    "dLoad"
]

# Complete 36 Clean Feature List (Tabular Order)
WUSTL_FEATURE_NAMES = [
    "Sport", "Dport", "Dur", "TotPkts", "TotBytes",
    "SrcBytes", "DstBytes", "sRate", "dRate", "sLoad",
    "dLoad", "sMeanPktSz", "dMeanPktSz", "sJitter", "dJitter",
    "sHops", "dHops", "sTtl", "dTtl", "Proto_Code",
    "State_Code", "Mean_Pulse", "Mean_SpO2", "Mean_Temp", "Mean_Blood_Pressure",
    "Respiration_Rate", "Mean_ECG", "Tcp_Rtt", "Ack_Dat", "Syn_Ack",
    "sLoss", "dLoss", "sWin", "dWin", "sAppBytes", "dAppBytes"
]

# Baseline Benchmark Results for Model Arena
DEFAULT_BENCHMARK_RESULTS = [
    {
        "Setting": "Centralized DNN (Replication Baseline)",
        "Accuracy": 0.877,
        "Precision": 0.510,
        "Recall": 0.730,
        "F1-Score": 0.600,
        "Privacy (Epsilon)": "None (Raw Data Centralized)",
        "Status": "Verified Local Run"
    },
    {
        "Setting": "Centralized CNN (Replication Baseline)",
        "Accuracy": 0.702,
        "Precision": 0.270,
        "Recall": 0.800,
        "F1-Score": 0.400,
        "Privacy (Epsilon)": "None (Raw Data Centralized)",
        "Status": "Verified Local Run"
    },
    {
        "Setting": "Federated DNN (FedAvg, No DP)",
        "Accuracy": 0.865,
        "Precision": 0.495,
        "Recall": 0.710,
        "F1-Score": 0.583,
        "Privacy (Epsilon)": "Federated Only (No Gradient Noise)",
        "Status": "Simulated Shards"
    },
    {
        "Setting": "Federated DNN + DP (Opacus, Noise=1.5)",
        "Accuracy": 0.832,
        "Precision": 0.460,
        "Recall": 0.680,
        "F1-Score": 0.548,
        "Privacy (Epsilon)": "0.44 (Strong Guarantee)",
        "Status": "Recommended DP"
    },
    {
        "Setting": "Federated DNN + DP (Opacus, Noise=0.5)",
        "Accuracy": 0.781,
        "Precision": 0.380,
        "Recall": 0.610,
        "F1-Score": 0.468,
        "Privacy (Epsilon)": "1.82 (Moderate Guarantee)",
        "Status": "Degraded / High Variance"
    },
    {
        "Setting": "MDPI Paper Best Reported (WUSTL, DNN, Noise=1.5)",
        "Accuracy": 0.932,
        "Precision": 0.915,
        "Recall": 0.942,
        "F1-Score": 0.928,
        "Privacy (Epsilon)": "0.44",
        "Status": "Original Paper (Mosaiyebzadeh et al. 2025)"
    }
]

# Attack Presets for Live Simulator
ATTACK_PRESETS = {
    "Normal Patient Telemetry": {
        "description": "Stable biometric vitals and normal background encrypted HTTPS telemetry.",
        "label": 0,
        "vitals": {
            "Mean_Pulse": 74.0,
            "Mean_SpO2": 98.2,
            "Mean_Temp": 36.8,
            "Mean_Blood_Pressure": 120.0,
            "Respiration_Rate": 16.0,
            "Mean_ECG": 0.82
        },
        "network": {
            "Sport": 443.0,
            "Dport": 51234.0,
            "Dur": 1.25,
            "TotPkts": 14.0,
            "TotBytes": 1820.0,
            "SrcBytes": 780.0,
            "DstBytes": 1040.0,
            "sRate": 11.2,
            "dRate": 12.0
        }
    },
    "Biometric Spoofing Attack": {
        "description": "Adversary injects forged biometric readings (erratic pulse spikes, abnormal ECG voltage).",
        "label": 1,
        "vitals": {
            "Mean_Pulse": 168.0,
            "Mean_SpO2": 82.5,
            "Mean_Temp": 40.5,
            "Mean_Blood_Pressure": 185.0,
            "Respiration_Rate": 34.0,
            "Mean_ECG": 2.45
        },
        "network": {
            "Sport": 8080.0,
            "Dport": 443.0,
            "Dur": 0.12,
            "TotPkts": 4.0,
            "TotBytes": 320.0,
            "SrcBytes": 220.0,
            "DstBytes": 100.0,
            "sRate": 33.3,
            "dRate": 8.3
        }
    },
    "Denial of Service (DoS Packet Storm)": {
        "description": "Rapid packet flooding targeting the IoHT edge gateway, overwhelming flow buffers.",
        "label": 1,
        "vitals": {
            "Mean_Pulse": 75.0,
            "Mean_SpO2": 97.8,
            "Mean_Temp": 37.0,
            "Mean_Blood_Pressure": 122.0,
            "Respiration_Rate": 17.0,
            "Mean_ECG": 0.85
        },
        "network": {
            "Sport": 80.0,
            "Dport": 80.0,
            "Dur": 48.6,
            "TotPkts": 4820.0,
            "TotBytes": 684200.0,
            "SrcBytes": 680000.0,
            "DstBytes": 4200.0,
            "sRate": 820.5,
            "dRate": 4.2
        }
    },
    "Data Alteration / Man-in-the-Middle": {
        "description": "Subtle tampering of transmitted patient packets with altered payload sizes and out-of-order TTLs.",
        "label": 1,
        "vitals": {
            "Mean_Pulse": 88.0,
            "Mean_SpO2": 94.0,
            "Mean_Temp": 38.2,
            "Mean_Blood_Pressure": 145.0,
            "Respiration_Rate": 21.0,
            "Mean_ECG": 1.25
        },
        "network": {
            "Sport": 3128.0,
            "Dport": 443.0,
            "Dur": 5.4,
            "TotPkts": 120.0,
            "TotBytes": 24500.0,
            "SrcBytes": 18200.0,
            "DstBytes": 6300.0,
            "sRate": 22.2,
            "dRate": 15.6
        }
    }
}
