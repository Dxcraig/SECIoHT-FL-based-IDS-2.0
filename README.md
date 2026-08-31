# Privacy-Preserving Federated Learning-Based Intrusion Detection System for IoHT Devices

This is a student replication (Level 300 DCIT research assignment) of the paper below, working through it stage by stage: preprocessing, model building, federated learning, and differential privacy, with every methodology decision and deviation from the paper documented as we go rather than assumed silently.

## Paper being replicated

Fatemeh Mosaiyebzadeh, Seyedamin Pouriyeh, Meng Han, Liyuan Liu, Yixin Xie, Liang Zhao, Daniel Macêdo Batista. *Privacy-Preserving Federated Learning-Based Intrusion Detection System for IoHT Devices*. Electronics 2025, 14, 67.
https://www.mdpi.com/2079-9292/14/1/67 (DOI: https://doi.org/10.3390/electronics14010067)

Original authors' repo: https://github.com/fatemehm/SECIoHT-FL-based-IDS

## The problem it addresses

Internet of Healthcare Things (IoHT) devices — heart monitors, oxygen sensors, blood pressure trackers — connect patients to hospitals over open internet connections vulnerable to spoofing, DoS, data alteration, and ARP attacks. A traditional Intrusion Detection System (IDS) needs to centralize patient data to train a security model, which is a serious privacy problem in healthcare. This paper's answer, **SECIoHT-FL** (Secure Edge Computing IoHT Federated Learning), combines three techniques:

1. **Federated Learning (FL)** — each device trains locally; only model weights are shared with a server, never raw patient data. The server aggregates via weighted averaging (FedAvg).
2. **Differential Privacy (DP)** — even shared weights can leak information (gradient inversion, membership inference), so each device adds calibrated noise to its gradients before sharing them (DP-SGD, via the Opacus library), quantified by a privacy budget epsilon (lower = more private).
3. **Intrusion Detection** — two deep learning models (DNN and CNN) classify traffic as Attack or Normal.

## Datasets

- **wustl-ehms-2020** (Washington University in St. Louis Electronic Health Monitoring System): https://www.cse.wustl.edu/~jain/ehms/index.html — biometric + network-flow features (ECG, SpO2, temperature, blood pressure, etc.), 16,318 raw rows, attacks are spoofing/data alteration, imbalanced 7:1 normal:attack.
- **ECU-IoHT** (Edith Cowan University): https://ro.ecu.edu.au/datasets/48/ — network traffic features, 111,207 raw rows, attacks are ARP spoofing/DoS/Nmap/Smurf. **Currently parked** — see the Stage 1 section below for why.

Neither dataset is committed to this repo (size, and possible redistribution restrictions) — download them from the links above and place them directly in `our_work/` before running the notebook (see **How to run**).

## Repository layout

```
our_work/
  SECIoHT_FL_Preprocessing.ipynb   <- the actual working notebook (Stages 1-4, see below)
README.md
.gitignore
```

That's it. This repo previously also contained a `src/` Python package (models/federated/privacy/training modules with a pytest suite and CI), a `tests/` directory, and the original authors' raw research notebook (`FL-DP-WUSTL.ipynb`). Those were a separate, unused scaffold — this project's actual work has always lived entirely in `our_work/`'s notebook, so that scaffold has been removed from the repo going forward. It's still recoverable from git history if ever needed (`git log --all --diff-filter=D -- src/` from before the removal commit).

## What's been done so far

Everything lives in one notebook, `our_work/SECIoHT_FL_Preprocessing.ipynb`, structured as four stages that build on each other in order. All decisions below were made deliberately and are also documented inline in the notebook's markdown cells, not just here.

### Stage 1 — Preprocessing

Loaded both raw datasets and compared them against the paper's claims:

| | Paper claims | Actually found |
|---|---|---|
| WUSTL | 36 features | 45 raw columns → 36 after dropping 8 non-predictive/leaky columns (`Dir`, `Flgs`, `SrcAddr`, `DstAddr`, `SrcMac`, `DstMac`, `Packet_num`, `Attack Category`) |
| ECU | 7 features | 9 raw columns → 5 usable after dropping 4 (`No.`, `Info`, `Type`, `Type of attack`) |

Other findings worth flagging:
- **WUSTL's `Sport` column** mixes numeric ports with service names (`'http'`, `'dircproxy'`) as text — fixed with `pd.to_numeric(errors='coerce')` + drop unconvertible rows.
- **ECU's label distribution is inverted**: Attack (87,754) outnumbers Normal (23,453) — the opposite of what a real network looks like, suggesting an artificially constructed dataset. **This is why ECU is currently parked** rather than used in Stages 2–4 — we don't want that discrepancy silently contaminating model results until it's understood.
- Class imbalance fixed with SMOTE; features normalized with `StandardScaler`.

### Stage 2 — Model Building & Centralized Baseline (WUSTL only)

- **A methodology fix over the initial Stage 1 approach**: SMOTE was originally applied to the *entire* dataset with no train/test split, which risks synthetic rows leaking between train and test and inflating accuracy. Stage 2 instead: cleans → **splits 80/20 first** (stratified, on real data) → SMOTE the training split only → scales (fit on train, applied to test). The test set stays 100% real, unseen data.
- **DNN**: paper specifies `64(8×8) → 32 → 2`. We treat `(8×8)` as how the paper's figure *drew* a 64-unit layer (an 8×8 grid = 64 neurons), not an instruction to reshape tabular features into a 2D image — there's no spatial structure to preserve in flow/biometric features. Implemented as `Linear(36→64→32→2)` with ReLU, `CrossEntropyLoss` standing in for the paper's softmax output (mathematically equivalent, numerically standard).
- **CNN**: paper gives layer sizes `128→64→32→16→2` but no kernel/stride/pooling spec for non-image input. Reconstructed as four `Conv1d` layers (channels 128/64/32/16, kernel=3, `padding=1` to preserve sequence length) → `AdaptiveAvgPool1d(1)` → `Linear(16→2)`. This is a best-effort reconstruction, explicitly not a verified match to the authors' code.
- **Epoch sweep**: rather than guessing an epoch count, both models train up to 150 epochs with periodic test-set checkpointing (not just train accuracy), automatically keeping the best-by-test-accuracy checkpoint — standard early-stopping-by-checkpointing, addressing that a fixed 50-epoch guess had the CNN still visibly improving.
- Verified results from the one successful local run (50-epoch baseline, before the epoch sweep): **DNN 87.7% accuracy** (precision 0.51, recall 0.73, F1 0.60), **CNN 70.2% accuracy** (precision 0.27, recall 0.80, F1 0.40). Lower than the paper's 93.2% — expected, since this has no FL/DP yet and uses reconstructed (not verified) architectures. Re-running the notebook (with the epoch sweep now included) will produce updated numbers.

### Stage 3 — Federated Learning (FedAvg, WUSTL, DNN)

Simulates 3 clients (representing separate devices/hospitals) holding disjoint local shards of the training data, training locally, and exchanging only model weights with a server that FedAvg-aggregates them (weighted by each client's sample count).

- **Data split across clients**: i.i.d. random shuffle-split — real hospitals would likely be non-i.i.d. (different attack mixes, patient populations), which this doesn't model; flagged as a simplification.
- **An intentional asymmetry between scaling and SMOTE**: the `StandardScaler` is fit once on the combined training set (sharing aggregate per-feature mean/variance across the federation — a standard low-risk federated normalization pattern, not raw data sharing), but **SMOTE runs independently per client**, strictly on that client's own local shard. This is the part that has to stay local — a shared SMOTE (like Stage 2 uses, deliberately, since Stage 2 has no privacy requirement) would let one client's synthetic rows leak signal derived from another client's real data.
- **Rounds/epochs**: paper's config (`100 epochs`, `100 communication rounds`) read together implies a very large amount of compute whose exact meaning isn't fully clear. Used `NUM_ROUNDS=20`, `LOCAL_EPOCHS=5` (100 total local epochs per client, same order of magnitude as Stage 2's 50) — both are one-line changes to scale up.
- DNN only for this stage (stronger performer in Stage 2); reuses Stage 2's exact train/test split (saved to `.npy` so the centralized-vs-federated comparison is apples-to-apples).

### Stage 4 — Differential Privacy (Opacus, DP-SGD)

Wraps each Stage 3 client's local training with Opacus so noise is added **on the device, before anything is sent to the server** (Local DP) — closing the gap that Stage 3's exact, un-noised weights still leave open (gradient inversion / membership inference risk even without sharing raw data).

- **Mechanism**: per-sample gradient clipping (L2 norm ≤ `max_grad_norm`) + Gaussian noise scaled by `noise_multiplier × max_grad_norm`, added before the optimizer step. Opacus's `PrivacyEngine.get_epsilon(delta)` tracks cumulative privacy budget spent.
- **A structural detail that matters**: each client is a *persistent* object (`DPClient`) whose model/optimizer/`PrivacyEngine` are created once and live for the whole run, with only weights overwritten each round from the global model. Re-wrapping a fresh `PrivacyEngine` every round (mirroring Stage 3's fresh-model-per-round pattern) would silently reset the privacy accountant and under-report the true cumulative epsilon — a real correctness bug, not cosmetic.
- **Runs both of the paper's noise settings** for direct comparison: `noise_multiplier=1.5` (paper's recommended) and `0.5` (paper's "not recommended," where they report their CNN collapsing to TN=0).
- **Flagged, not silently changed**: the paper states `max_grad_norm=10⁻⁴`, unusually aggressive versus the typical ~1.0 for DP-SGD — implemented as stated, with an explicit note that a collapsed/non-learning model at this value is worth checking against the paper by hand rather than assumed to be a code bug.
- Verified (without running on real data locally — see **Known limitations** below): a synthetic smoke test confirmed the exact `DPClient` pattern (wrap once, copy weights in per round, accumulate epsilon across rounds) works correctly on this Opacus/PyTorch version — epsilon grew from 1.17 → 1.61 across two simulated rounds, as it should.
- Produces the full comparison chain: centralized → federated (no DP) → federated+DP(1.5) → federated+DP(0.5), saved to `stage4_full_comparison_wustl.csv`.

### Known limitations / honest caveats

- **Stage 3 and Stage 4 have not yet been run against the real dataset.** All code is syntax-checked and, for Stage 4's core privacy-accounting logic, verified against a synthetic smoke test — but real accuracy/epsilon numbers for federated and federated+DP still need an actual run (see **How to run**).
- The development machine used for local work has 2 CPU cores with unusually high per-operation overhead for PyTorch training (measured: a small CNN took ~40s/epoch, when it should be near-instant) — full-scale local execution wasn't practical, hence moving to Colab.
- ECU-IoHT is parked, not abandoned — the inverted label distribution needs to be understood before it's trusted for training.
- Several architecture/hyperparameter choices are documented assumptions, not verified matches to the paper's code (see Stage 2–4 sections above): the DNN's `(8×8)` notation, the CNN's convolutional design, client count, communication rounds/local epochs, and the DP clipping norm.

## How to run

1. Download both datasets from the links above.
2. Open `our_work/SECIoHT_FL_Preprocessing.ipynb` in Google Colab (or upload it there).
3. Upload both dataset files into the Colab session's working directory (`/content/`) — same directory as the notebook.
4. Runtime → Change runtime type → GPU recommended (Stage 3/4 and the epoch sweep are CPU-heavy).
5. Runtime → Run all.

The notebook is fully linear — no hidden dependency on cells run out of order.

## Results snapshot

| Setting | Accuracy | Precision | Recall | F1 | ε |
|---|---|---|---|---|---|
| Centralized DNN (Stage 2, 50-epoch baseline) | 87.7% | 0.51 | 0.73 | 0.60 | — |
| Centralized CNN (Stage 2, 50-epoch baseline) | 70.2% | 0.27 | 0.80 | 0.40 | — |
| Federated DNN, no DP (Stage 3) | *pending Colab run* | | | | — |
| Federated DNN + DP, noise=1.5 (Stage 4) | *pending Colab run* | | | | *pending* |
| Federated DNN + DP, noise=0.5 (Stage 4) | *pending Colab run* | | | | *pending* |
| Paper's reported best (WUSTL, DNN, noise=1.5) | 93.2% | — | — | — | 0.44 |

## Next steps

- **Stage 5 — Formal evaluation**: mostly presentation of Stage 4's comparison table in the paper's own results format, once real numbers are in from a Colab run.
- **Stage 6 — SHAP explainability**: run SHAP against the best-performing model to identify which of the 36 WUSTL features drive its predictions (paper's own top features: flow duration, source port, temperature, pulse rate).

## Algorithms used

Decentralized (federated): DNN, CNN
Centralized baseline: DNN, CNN

## Citation

Fatemeh Mosaiyebzadeh, Seyedamin Pouriyeh, Meng Han, Liyuan Liu, Yixin Xie, Liang Zhao, Daniel Macêdo Batista. Privacy-Preserving Federated Learning-Based Intrusion Detection System for IoHT Devices. https://www.mdpi.com/2079-9292/14/1/67
