# SECIoHT-FL Streamlit Application

This directory contains the self-contained interactive dashboard for the Privacy-Preserving Federated Learning-Based Intrusion Detection System (SECIoHT-FL).

---

## 1. Directory Structure

```
streamlit_app/
├── Home.py                       # Application landing page and architecture hub
├── requirements.txt              # Application Python dependencies
├── README.md                     # Documentation and launch instructions
├── pages/                        # Multi-page views
│   ├── 1_Dataset_Explorer.py     # WUSTL vs ECU telemetry and SMOTE analysis
│   ├── 2_FL_and_Privacy_Lab.py   # Multi-client FedAvg and Opacus DP simulator
│   ├── 3_Benchmark_Arena.py      # Centralized vs FL vs DP evaluation tables
│   ├── 4_Live_IDS_Detector.py    # Real-time IoHT patient telemetry and attack testbed
│   └── 5_SHAP_Diagnostics.py     # Explainable AI feature attribution
├── src/                          # Modular application backend
│   ├── __init__.py
│   ├── config.py                 # Feature definitions, presets, hyperparameters
│   ├── preprocessing.py          # Data cleaner, scaler loader, sample generator
│   ├── models.py                 # PyTorch DNN and 1D-CNN architectures
│   ├── federated.py              # FedAvg aggregation and DP simulation
│   ├── inference.py              # Model loader, predictor, and anomaly evaluator
│   └── explainability.py         # SHAP explanation and attribution helpers
├── artifacts/                    # Models, sample data, cached results
│   ├── models/                   # PyTorch checkpoint storage (.pt)
│   ├── samples/                  # Bundled offline test samples (zero crash)
│   └── results/                  # Precomputed benchmark comparison tables
├── components/                   # Reusable UI widgets
│   ├── __init__.py
│   └── sidebar.py                # Navigation and system status widget
└── styles/
    └── custom.css                # Professional styling
```

---

## 2. Quickstart Instructions

### Step 1: Activate Virtual Environment
Ensure your Python environment is activated:
```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 2: Install App Dependencies
If not already installed, run:
```powershell
pip install -r streamlit_app/requirements.txt
```

### Step 3: Run the Application
From the workspace root, execute:
```powershell
streamlit run streamlit_app/Home.py
```
Open `http://localhost:8501` in your browser.

---

## 3. Key Application Features

1. **Zero-Setup Offline Resilience**:
   The app bundles a realistic sample dataset in `artifacts/samples/wustl_sample.csv`. The entire dashboard functions immediately without downloading the multi-gigabyte raw datasets.

2. **Live Edge IoHT Testbed**:
   Interactive sliders for patient vitals (Heart rate, SpO2, Temperature, Blood pressure) and network flow telemetry (Ports, Flow duration, Packets) with real-time classification alerts.

3. **Federated Learning and Opacus DP Sandbox**:
   Simulate decentralized model training across 3 simulated hospitals with configurable communication rounds, local epochs, and differential privacy noise multipliers (0.5 vs 1.5).

4. **Explainable AI (SHAP)**:
   Waterfall plots and global feature rankings identify which physiological anomalies or packet characteristics triggered the intrusion alert.
