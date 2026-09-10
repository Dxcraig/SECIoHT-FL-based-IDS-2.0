# Methodology Note 1: Preprocessing (Stage 1) — Why We Deviated From the Paper

**Covers:** dataset verification, column selection, data cleaning, class balancing
**Stage owner:** Sonpon Ye-shua Chief (lead) — reviewed by Edem Doe Lawson

---

## 1. What the paper actually specifies

The paper's preprocessing description is four sentences long. It states WUSTL-EHMS-2020 uses 36 features and ECU-IoHT uses 7, applies SMOTE for class balancing, and normalizes with a standard scaler. It does not publish a column list, does not describe how it handled non-numeric fields, and does not explain how the raw files (45 columns for WUSTL, 9 for ECU, as delivered by the dataset authors) were reduced to those counts.

This matters more than it sounds like it should: a preprocessing pipeline is not a detail you can approximate and still trust the downstream accuracy numbers. Every column you keep or drop changes what the model is allowed to learn from.

## 2. Where we deviated, and why

### 2.1 We rebuilt the column selection from scratch instead of trusting the paper's counts

We loaded both raw files and counted columns ourselves rather than assuming the paper's "36" and "7" were self-explanatory targets to hit by any means. WUSTL arrived with 45 columns; we identified and dropped 8 that are either identifiers with no generalizable signal (`SrcAddr`, `DstAddr`, `SrcMac`, `DstMac` — an IP or MAC address doesn't teach a model anything transferable, it just memorizes specific hosts), a row counter (`Packet_num`), text-only network flags not usable in this form (`Dir`, `Flgs`), or direct label leakage (`Attack Category` literally states the answer). That leaves 36 features plus the `Label` target — the same *count* the paper reports, but we cannot verify it is the same *set* of 36, because the paper never lists one.

We attempted to verify this independently: the authors' GitHub repository blocks raw file access for automated retrieval, so we cross-referenced a related paper using the same WUSTL dataset ("On the Performance of Cyber-Biomedical Features for Intrusion Detection in Healthcare 5.0"), which confirmed several of our drop decisions (`SrcMac`, `DstMac`, `Dir`, `Flgs`, `Attack Category`) independently. ECU's column drops (`No.`, `Info`, `Type`, `Type of attack`) have no equivalent external confirmation — this remains a documented, unverified assumption.

**Why we did it this way rather than just matching the paper's number:** matching a *count* by dropping whichever 9 columns get you to 36 is not the same as matching a *method*. If our reasoning for each drop is sound (leakage, non-generalizable identifiers, redundancy) independent of what number it lands on, the pipeline is defensible even where it can't be proven identical to the original.

### 2.2 We found and fixed a data quality problem the paper doesn't mention

WUSTL's `Sport` column is not purely numeric — it mixes literal port numbers (`'80'`) with service names (`'http'`, `'dircproxy'`) as text in the same column. This isn't something you'd guess from the paper's description; it only surfaces when you actually try to feed the column into a model and it crashes. We fixed it with `pd.to_numeric(errors='coerce')` followed by dropping the rows that don't convert.

A related paper using the same dataset states that only 3 anomalous rows needed removal for this exact issue. Our fix removed more rows than that (from ~16,318 down to 16,315, i.e. more than 3). **This is a flagged, unresolved discrepancy** — we do not know whether the related paper used a different/cleaner data snapshot, a different fix, or whether our coercion is more aggressive than necessary. It's noted here rather than silently reconciled, because papering over the difference would be worse than admitting we haven't closed it yet.

### 2.3 We parked ECU-IoHT instead of using it

ECU's label distribution is inverted relative to what a real network looks like: 87,754 attack rows versus 23,453 normal rows — attacks are the *majority* class. In any real deployment, attacks are rare events; a dataset where they dominate suggests either a synthetic construction process or a labeling scheme we don't yet understand.

We made a deliberate call not to proceed with modeling on ECU until this is understood, rather than let an unexplained anomaly silently shape (and potentially invalidate) model results downstream. This is a scope reduction relative to the paper (which reports results for both datasets), traded for not shipping a number we can't stand behind.

## 3. Task log

| Task | Lead | Reviewer | Notes |
|---|---|---|---|
| Acquire both raw datasets, initial shape/structure exploration | Sonpon Ye-shua Chief | Edem Doe Lawson | Confirmed 45/9 raw columns vs. paper's claimed 36/7 |
| Cross-reference WUSTL column drops against related literature | Edem Doe Lawson | Sonpon Ye-shua Chief | GitHub raw access blocked; used related-paper cross-check instead |
| Implement cleaning pipeline (dropna, dedup, column drops) | Sonpon Ye-shua Chief | Edem Doe Lawson | |
| Diagnose and fix the `Sport` mixed-type column | Sonpon Ye-shua Chief | Edem Doe Lawson | Discrepancy vs. related paper's "3 rows" left open, not resolved |
| Investigate ECU's inverted label distribution | Edem Doe Lawson | Sonpon Ye-shua Chief | Decision made to park ECU rather than proceed |
| Apply SMOTE + `StandardScaler`, document Stage 1 findings | Sonpon Ye-shua Chief | Edem Doe Lawson | Superseded in Stage 2 by the leak-free split methodology (see doc 2) |
