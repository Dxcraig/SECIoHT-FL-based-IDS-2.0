# Methodology Note 2: Model Building & Centralized Baseline (Stage 2) — Why We Deviated From the Paper

**Covers:** train/test methodology, DNN architecture, CNN architecture, training length
**Stage owner:** Edem Doe Lawson (lead) — reviewed by Sonpon Ye-shua Chief

---

## 1. What the paper actually specifies

DNN: `64(8×8) → 32 → 2` nodes. CNN: `128 → 64 → 32 → 16 → 2` nodes. Both use ReLU activation, softmax output, SGD optimizer at learning rate 0.01, binary cross-entropy loss, batch size 64, 100 epochs. That's the entire architecture and training specification available to us — no kernel sizes, no stride, no pooling strategy for the CNN, and no clarification of what "64(8×8)" means for a network trained on flat tabular feature vectors rather than images.

## 2. Where we deviated, and why

### 2.1 We changed *when* SMOTE happens relative to the train/test split — the most consequential deviation in this stage

Our own Stage 1 preprocessing applied SMOTE to the entire dataset before any split existed. This is a real methodological hazard, not a stylistic quibble: SMOTE generates synthetic minority-class rows by interpolating between a real row and its nearest real neighbors. If you split into train/test *after* SMOTE has already run, a synthetic row in the test set can be a near-duplicate of a real row sitting in the training set (or vice versa) — meaning the model may have effectively already seen a close variant of what it's being "tested" on. The resulting accuracy number is inflated by an amount that's hard to quantify after the fact.

We restructured the pipeline so the order is: clean → **stratified 80/20 split on real data** → SMOTE the training split only → `StandardScaler` fit on training data only, applied to both splits. The test set is never touched by SMOTE and never contributes to the scaler's fitted statistics. This is a deviation from our own earlier Stage 1 work, not from the paper directly (the paper doesn't describe its split/SMOTE ordering at all) — but it's the kind of gap where "the paper doesn't say" is not license to pick whichever order is easiest; it's a reason to default to the version that can't accidentally lie to you about accuracy.

### 2.2 We made an explicit, documented interpretive choice about `64(8×8)`

Read literally, "(8×8)" could suggest reshaping a feature vector into an 8×8 grid before the 64-node layer. We rejected that: WUSTL's 36 features (biometric readings, flow statistics) have no 2D spatial relationship to each other the way pixels in an image do, so imposing a grid structure would be arbitrary rather than meaningful. We instead read "(8×8)" as describing how the paper's own architecture *diagram* likely rendered a 64-unit layer visually (8 rows × 8 columns of drawn neurons = 64), and implemented the DNN as a standard `Linear(N_features → 64) → ReLU → Linear(64 → 32) → ReLU → Linear(32 → 2)` stack. This is an assumption, stated as one, not a claim of certainty.

### 2.3 We reconstructed the CNN's convolutional design, because the paper doesn't specify one

A CNN needs kernel size, stride, and a pooling strategy to be a concrete architecture rather than just a list of layer widths — none of which the paper provides for a non-image input. Our reconstruction: treat each sample as a length-N 1D signal with 1 input channel, run it through four `Conv1d` layers using the paper's stated channel progression (128 → 64 → 32 → 16), kernel size 3, `padding=1` (preserves sequence length regardless of how few input features a dataset has), then `AdaptiveAvgPool1d(1)` to collapse to one value per channel, then a final `Linear(16 → 2)`. This is a standard "1D-CNN-for-tabular-data" pattern from the broader IDS literature, not something specific to this paper — we chose it because it's a defensible, commonly-used choice for this exact problem shape, and we say so rather than presenting it as a recovered original design.

### 2.4 We replaced a fixed epoch guess with a measured stopping point

The paper's "100 epochs" figure is stated in the context of federated communication rounds, and it's genuinely unclear whether that means 100 epochs *total* or 100 epochs *per round* repeated across 100 rounds (a very different amount of compute). Rather than guess a number for the centralized baseline and hope it was reasonable, we started at 50 epochs as a first pass, observed that the CNN in particular was still visibly improving in accuracy at epoch 50 (not converged), and added an epoch sweep: train up to 150 epochs, checkpoint on *test* accuracy (not train accuracy) periodically, and automatically keep whichever checkpoint had the best test performance. This directly answers "how long should we actually train" with evidence — the point where test accuracy stops improving while train accuracy keeps climbing — rather than a number picked in advance.

### 2.5 Feature count matching the paper is a coincidence we can state, not a fact we can prove

WUSTL's cleaned feature count came out to 36, matching the paper's stated 36. We're explicit in the notebook and in doc 1 that matching in *count* is not the same as matching in *identity* — we have no confirmed column-for-column list from the paper to compare against.

## 3. Task log

| Task | Lead | Reviewer | Notes |
|---|---|---|---|
| Diagnose the SMOTE-before-split leakage risk in the Stage 1 pipeline | Edem Doe Lawson | Sonpon Ye-shua Chief | Root cause: no train/test split existed prior to SMOTE |
| Redesign and implement the leak-free split → SMOTE(train only) → scale pipeline | Edem Doe Lawson | Sonpon Ye-shua Chief | |
| Implement the DNN architecture and document the `(8×8)` interpretation | Sonpon Ye-shua Chief | Edem Doe Lawson | |
| Reconstruct and implement the CNN (kernel/stride/pooling design) | Edem Doe Lawson | Sonpon Ye-shua Chief | No paper spec to verify against; documented as best-effort |
| Build the epoch-sweep / best-checkpoint training utility | Sonpon Ye-shua Chief | Edem Doe Lawson | Addresses the ambiguous "100 epochs" figure |
| Run centralized baseline, verify metrics (accuracy/precision/recall/F1/confusion matrix) | Edem Doe Lawson | Sonpon Ye-shua Chief | 50-epoch baseline: DNN 87.7%, CNN 70.2% (pre-sweep numbers) |
