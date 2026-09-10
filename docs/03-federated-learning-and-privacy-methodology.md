# Methodology Note 3: Federated Learning & Differential Privacy (Stages 3–4) — Why We Deviated From the Paper

**Covers:** client simulation design, FedAvg, Opacus DP-SGD integration, privacy accounting
**Stage owners:** Stage 3 — Sonpon Ye-shua Chief (lead), Edem Doe Lawson (reviewer). Stage 4 — Edem Doe Lawson (lead), Sonpon Ye-shua Chief (reviewer).

---

## 1. What the paper actually specifies

Each device trains locally and shares only model weights; the server aggregates via weighted averaging over 100 communication rounds. Differential privacy is applied locally per device via Opacus, tested at two noise multiplier settings (0.5 and 1.5), with gradient clipping norm 10⁻⁴, delta 10⁻⁴, and epsilon reported per configuration. The paper does not state how many simulated clients/devices it used, how training data was partitioned across them, or how local preprocessing (specifically SMOTE) was handled per client versus globally.

## 2. Where we deviated, and why

### 2.1 We had to invent a client topology the paper never specifies

Nothing in the available material states how many clients were simulated. We chose 3, representing 3 separate devices/hospitals, as a plausible cross-silo FL setup (a small number of institutional participants, not thousands of individual consumer devices) — and made it a named constant (`NUM_CLIENTS`) specifically so it's a one-line, clearly-visible assumption rather than a number buried in logic. We also split training rows across clients i.i.d. (random shuffle), which is a simplification — real hospitals would likely have non-identically-distributed data (different local attack mixes, different patient populations), and we are not modeling that skew. We're testing the privacy *mechanism*, not simulating realistic institutional heterogeneity, and we say so rather than implying otherwise.

### 2.2 We deliberately treated scaling and SMOTE differently from each other — this is the most important design decision in Stage 3

If every client independently fits its own `StandardScaler` on only its own local data, the resulting per-client feature spaces are not guaranteed to be consistent with each other, which is a real problem for FedAvg: averaging model weights learned in different feature spaces doesn't cleanly mean what it should. Our resolution was to fit the scaler **once**, on the combined training set, and treat that as sharing *aggregate statistics* (per-feature mean/variance) across the federation — not raw records, not synthetic rows, just two numbers per feature. This is a recognized, low-risk pattern in federated learning practice (sometimes called federated normalization).

SMOTE is handled the opposite way, deliberately: it runs **independently per client**, strictly on that client's own already-scaled local shard. This is the piece that has to stay local for the privacy story to be real — SMOTE synthesizes new samples by interpolating between real neighbors, so a shared/global SMOTE (the choice Stage 2 makes, correctly, since Stage 2 has no privacy requirement to satisfy) would let one client's synthetic rows carry interpolated signal derived from another client's real, private data. Keeping SMOTE local is what makes Stage 3 an actual federated pipeline rather than centralized training wearing an FL-shaped wrapper.

### 2.3 We scaled down the paper's round/epoch configuration, and documented why rather than silently using a smaller number

The paper states "100 epochs" and "100 communication rounds" as separate configuration values; read together, that most likely means 100 local epochs per round repeated over 100 rounds per client — an unusually large amount of compute whose precise intent the paper doesn't fully clarify. We used `NUM_ROUNDS=20`, `LOCAL_EPOCHS=5` for Stage 3 (100 total local epochs per client — the same order of magnitude as Stage 2's 50-epoch centralized baseline, just distributed across rounds instead of run continuously), and even lower for Stage 4 (`NUM_ROUNDS_DP=10`, `LOCAL_EPOCHS_DP=3`) because Opacus's per-sample gradient computation is measurably more expensive than plain SGD — confirmed directly during development, where a synthetic 200-sample/2-epoch smoke test of the DP client pattern took over a minute on a 2-core CPU versus a sub-second equivalent without Opacus. Both sets of constants are one-line changes, intended to be scaled toward the paper's figures once running on faster hardware (e.g. Colab GPU) rather than treated as final.

### 2.4 We caught and fixed a subtle correctness bug in privacy accounting before it happened, not after

Stage 3's round structure creates a fresh local model copy from the global weights every round, trains it, and discards it. If Stage 4 mirrored that pattern exactly — wrapping a brand-new Opacus `PrivacyEngine` every round — the privacy accountant inside it would reset every round, and the epsilon it reports at the end would reflect only the most recent round's privacy cost, not the true cumulative cost of the entire training run. That's not a cosmetic difference: it would make the reported privacy guarantee look artificially strong, which is close to the worst kind of bug a DP implementation can have.

We addressed this before ever running Stage 4 for real, by designing each client as a persistent object (`DPClient`) whose model, optimizer, and `PrivacyEngine` are constructed once and live for the entire multi-round run — only the *weights* get overwritten from the global model at the start of each round, exactly matching what real federated learning does, while the privacy accountant keeps accumulating across every round it has ever trained. We verified this specific behavior with a synthetic smoke test (not the real dataset) before writing the real Stage 4 cells: epsilon correctly grew from 1.17 to 1.61 across two simulated rounds of the exact same client pattern, confirming the accountant persists rather than resets.

### 2.5 We implemented the paper's stated clipping norm as-is, and flagged it rather than "fixing" it ourselves

The paper states `max_grad_norm = 10⁻⁴`. Typical DP-SGD clipping norms in the literature are on the order of 1.0; a norm of 0.0001 would clip almost every per-sample gradient down to near-zero magnitude before noise is even added, which could plausibly prevent the model from learning much of anything, independent of the noise multiplier chosen. We implemented the value exactly as the paper states it rather than silently substituting a "more sensible" number, and left an explicit note in the code and documentation that if training collapses at this value, that's a prompt to double check the paper's figure by hand before assuming the implementation is at fault. Quietly changing a paper's stated hyperparameter to make our own results look better would undermine the point of a replication.

### 2.6 We carried only the DNN into Stages 3 and 4, not the CNN

The DNN was the stronger, more stable performer in Stage 2. Extending the same `DPClient`/FedAvg scaffolding to the CNN is a straightforward mechanical extension (swap the model class), but we scoped it out for now to get one architecture fully verified end-to-end (federated, then federated+DP) before doubling the surface area. This is a scope decision made under time/compute constraints, not a technical blocker.

## 3. Task log

| Task | Lead | Reviewer | Notes |
|---|---|---|---|
| Design the client-count and data-partitioning approach for FedAvg simulation | Sonpon Ye-shua Chief | Edem Doe Lawson | i.i.d. split, 3 clients — documented as simplifications |
| Design and justify the shared-scaler / local-SMOTE asymmetry | Sonpon Ye-shua Chief | Edem Doe Lawson | Core privacy-preserving design decision for Stage 3 |
| Implement FedAvg aggregation (weighted by client sample count) | Sonpon Ye-shua Chief | Edem Doe Lawson | Mirrors weighting rule of a standard FedAvg reference implementation |
| Scale down and document the round/local-epoch configuration | Edem Doe Lawson | Sonpon Ye-shua Chief | Flagged as a compute-driven deviation, not a scientific one |
| Integrate Opacus DP-SGD into per-client local training | Edem Doe Lawson | Sonpon Ye-shua Chief | |
| Identify and fix the fresh-PrivacyEngine-per-round accounting bug (persistent `DPClient` design) | Edem Doe Lawson | Sonpon Ye-shua Chief | Verified via synthetic smoke test before running on real data |
| Implement and flag the paper's stated `max_grad_norm=1e-4` | Edem Doe Lawson | Sonpon Ye-shua Chief | Implemented as-is; flagged rather than silently adjusted |
| Run both noise-multiplier settings (0.5, 1.5) and assemble the full comparison table | Edem Doe Lawson | Sonpon Ye-shua Chief | Centralized → federated → federated+DP, matching the paper's own ablation |
