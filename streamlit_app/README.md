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

Run these commands from the repository root in PowerShell:

### First-time setup

Use Python 3.12 or 3.13 because the pinned PyTorch dependency may not be
available for newer Python versions:

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r streamlit_app/requirements.txt
```

If Python 3.12 is not installed, use `py -3.13` in the first command instead.

### Start the app

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run streamlit_app/Home.py
```

Open `http://localhost:8501` in your browser.

### Start on another port

Use this when port 8501 is already occupied:

```powershell
python -m streamlit run streamlit_app/Home.py --server.port 8502
```

Then open `http://localhost:8502`.

### Stop the app

Press `Ctrl+C` in the terminal running Streamlit.

---

## 3. Key Application Features

1. **Zero-Setup Offline Resilience**:
   The app bundles a realistic sample dataset in `artifacts/samples/wustl_sample.csv`. The entire dashboard functions immediately without downloading the multi-gigabyte raw datasets.

2. **Live Edge IoHT Testbed**:
   Interactive sliders for patient vitals (Heart rate, SpO2, Temperature, Blood pressure) and network flow telemetry (Ports, Flow duration, Packets) with real-time classification alerts.

3. **Federated Learning and Privacy Results**:
   Review recorded FedAvg and Opacus results from the notebook. The app does not retrain or fabricate round-by-round curves; missing logs are marked as pending.

4. **Explainable AI (SHAP)**:
   Waterfall plots and global feature rankings identify which physiological anomalies or packet characteristics triggered the intrusion alert.

### Artifact contract

Copy notebook outputs into `artifacts/models/` and `artifacts/results/`. The app
prefers these notebook-produced files:

```text
models/dnn_centralized_wustl.pt
models/dnn_federated_wustl.pt
models/dnn_federated_dp_noise1_5_wustl.pt
models/dnn_federated_dp_noise0_5_wustl.pt
results/stage3_federated_vs_centralized_wustl.csv
results/stage4_full_comparison_wustl.csv
results/stage5_benchmark_evaluation.csv
results/stage6_feature_importance.csv
```

Until those files are produced by Colab, bundled values are labeled as
fallbacks rather than notebook results.
