# Paper ↔ Code Mapping

Paper: *Wearable Multimodal Physiomarker Identification for Graded Alcohol Craving During VR-Based Cue Elicitation* (IEEE JBHI).

This document maps each part of the paper to the code that produced it.

## Section-to-code map

| Paper section | Code |
| --- | --- |
| II.A Experimental equipment, II.B VR craving stimuli (Fig. 1, 2) | `data/alcohol_scenarios.csv` (scenario/episode composition), no processing code — describes the acquisition protocol |
| III.A Participant flow (Fig. 3) | `notebooks/02_abnormal_modality_sessions.ipynb` (cell "Abnormal signal session results") — session/modality exclusion sets |
| III.C Craving self-report (Fig. 7, 8) | `notebooks/03_label_and_scenario_sanity_checks.ipynb` |
| III.C Data preprocessing (ECG/PPG/EDA filtering, R-peak fusion, Kubios correction, Elgendi PPG peaks, PTT pairing, Eq. 1) | `src/preprocessing/methods.py` |
| III.C Feature extraction, 152 features (Table II) | `src/preprocessing/methods.py` (`get_ECG_features`, `get_PPG_features`, `get_ECG_PPG_PTT`, `get_GSR_features`), orchestrated by `src/datasets/biosignal_dataset.py` (`BioSignalDataset.get_features`), driven from `notebooks/01_feature_extraction.ipynb` |
| III.E Statistical analysis / LMM (Eq. 2, 3) | `notebooks/05_ml_classification_lmm_shap.ipynb`, cells implementing `run_one_experiment` (`pymer4.Lmer`) |
| III.F ML classifiers + SHAP | `notebooks/05_ml_classification_lmm_shap.ipynb`, main sweep cell (6 classifiers: KNN/SVM/LR/MLP/XGB/RF) |
| IV.B Tables III/IV, Fig. 9/10 (classification performance) | `notebooks/05_ml_classification_lmm_shap.ipynb` main sweep, using `scripts/verify_ml_tables.py` for a standalone re-run |
| IV.B Table V/VI (modality ablation) | Same notebook, modality-loop branch; standalone re-run in `scripts/verify_ml_tables.py` |
| IV.D Fig. 11 (circos LMM+SHAP), 36 candidate features, Table VII | `notebooks/05_ml_classification_lmm_shap.ipynb`, circos-building cells; standalone re-run in `scripts/verify_circos.py` + `scripts/verify_table7.py` |
| PTT validity window sensitivity (Eq. 1, p.9) | `notebooks/lmm_results/ptt_05_70(_2,_3)`, `ptt_05_80/90/95`, `ptt_10_60/70_2/80` — `ptt_05_70_3` is the configuration used in the paper |
| LMM configuration sensitivity sweep | `notebooks/lmm_sensitivity_results/EXP001`–`EXP064` (+ ad-hoc `EXP031_*` variants) — `EXP031` is the configuration used in the paper (identical settings to `ptt_05_70_3`) |

## Key hyperparameter locations

| Parameter | Value used in paper | Where it lives in code |
| --- | --- | --- |
| Sampling rates | ECG 512 Hz, PPG/EDA 51.2 Hz | `src/preprocessing/methods.py` (`FS_ECG`, `FS_PPG`, `FS_GSR`) |
| Filter bands (lowpass) | ECG 40 Hz, PPG 5 Hz, EDA 1 Hz | `src/preprocessing/methods.py` (`ECG_BAND`, `PPG_BAND`, `GSR_BAND`) |
| Window / overlap | 10 s, 50% | `src/datasets/biosignal_dataset.py` (`BioSignalDataset.get_features` defaults) |
| PTT validity window (Eq. 1) | 0.05–0.70 × NNI | Passed explicitly as `ptt_range=(0.05, 0.7)` in `notebooks/01_feature_extraction.ipynb` |
| Random seed | 42 | `SEED = 42` in `notebooks/01` and `notebooks/05` |
| LMM config | agg=median, complete-session filter, no session fixed effect, subject+session random effects, no pairwise adjustment, no FDR, per-feature conditional log-transform | `EXP031` / `ptt_05_70_3` |
| ML classifiers | KNN, SVM, LR, MLP, XGB, RF | `notebooks/05_ml_classification_lmm_shap.ipynb` model dict |
| CV scheme | 5-fold `StratifiedGroupKFold`, grouped at the trial level | `notebooks/05_ml_classification_lmm_shap.ipynb`, `run_full_experiment` |

## Data split / label logic

- Craving groups: `Q_mean == 0` → Zero, `0 < Q_mean <= 3.5` → Low, `Q_mean > 3.5` → High. Implemented identically everywhere it's used (`notebooks/01`, `notebooks/05`).
- Labels themselves: `data/alcohol_Qscores.csv` (flattened) / `data/labels/alcohol.json` (nested, source of truth).

## Evaluation metrics

- Balanced accuracy, weighted F1, AUROC, AUPRC — computed once on pooled out-of-fold predictions (`compute_comprehensive_metrics` in `notebooks/05_ml_classification_lmm_shap.ipynb`).

## Reproducibility notes

- Tables III, IV, and VII, the LMM analysis, and the 36-candidate-feature selection behind Fig. 11 have all been independently re-executed and confirmed to reproduce the paper's reported values exactly.
- `notebooks/02_abnormal_modality_sessions.ipynb`'s computed EDA-session exclusion count differs slightly from the count reported in the paper's participant-flow figure; this is a minor reconciliation item on the authors' side and does not affect any of the tables above.
- **`MISSING_GSR` (abnormal-EDA session exclusion set) discrepancy**: `notebooks/02_abnormal_modality_sessions.ipynb`/`notebooks/04_biomarker_spearman_correlations.ipynb` independently determine 33 sessions have unreliable GSR, but the `MISSING_GSR` constant actually used by the ML/LMM pipeline (`src/ml/experiment_utils.py`, driven from `notebooks/05_ml_classification_lmm_shap.ipynb`) only nulls out 23 of those 33 — 10 sessions (`1_1_024_V1`, `1_1_024_V2`, `1_1_025_V2`, `1_1_027_V2`, `1_1_029_V1`, `1_1_029_V2`, `1_1_032_V2`, `1_1_035_V1`, `1_1_035_V2`, `1_1_037_V1`) are flagged as bad-GSR by notebooks 02/04 but are not excluded when the ML pipeline nulls out EDA features. Since notebook 05 is the exact code that produced the paper's reported Tables III–VII, this is reproduced here as-is (not "fixed") — changing it would change the paper's actual reported numbers. Flagged for the authors' awareness, same treatment as the Table V/VI RF/XGB labeling swap above.
- `notebooks/lmm_results/` and `notebooks/lmm_sensitivity_results/` (referenced above) are regenerated locally by running the corresponding notebook 05 cells — they are not tracked in this repository's git history (see `.gitignore`), since they're result data, not source code.
