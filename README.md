# Multimodal Candidate Physiomarkers of VR-Elicited Graded Alcohol Craving

Submitted to the IEEE Journal of Biomedical and Health Informatics. This repository contains signal preprocessing, feature extraction, statistical analysis, and machine-learning notebooks.

## Study and analysis

A clinician-supervised VR cue-exposure study in 50 patients with Alcohol Use Disorder (AUD). ECG, PPG, and EDA were recorded continuously during progressive alcohol-cue VR scenarios; 152 cardiovascular/electrodermal features were extracted per 10-second window. A linear mixed-effects model (LMM) and SHAP-based ML explainability jointly screened these features for **graded** (Zero/Low/High) craving physiomarkers.

## Repository structure

```
.
├── src/
│   ├── preprocessing/methods.py       # signal filtering, R-peak fusion, Kubios correction,
│   │                                  #   Elgendi PPG peaks, PTT pairing, HRV/EDA feature extraction
│   └── datasets/biosignal_dataset.py  # BioSignalDataset: orchestrates raw->preprocessed->features
├── notebooks/  # run in this numeric order -- it reflects true data dependency, not just topic
│   ├── 01_feature_extraction.ipynb                  # feature extraction from raw signals (10s windows,
│   │                                                 #   PTT window 0.05-0.70) -- run this first; everything
│   │                                                 #   below depends on this notebook's output
│   ├── 02_abnormal_modality_sessions.ipynb          # session/modality QC (paper Fig. 3) -- needs 01's PTT output
│   ├── 03_label_and_scenario_sanity_checks.ipynb    # craving-label validation (Fig. 7, 8)
│   ├── 04_biomarker_spearman_correlations.ipynb     # QC feature filtering (used downstream) +
│   │                                                 #   an exploratory correlation pass (not in the paper)
│   ├── 05_ml_classification_lmm_shap.ipynb          # LMM + ML + SHAP -> paper Tables III-VII, Fig. 9-12
│   └── lmm_results/, lmm_sensitivity_results/,       # LMM sensitivity sweeps + other regenerated result
│       feature_rank_summary.csv, eda_scr_peaks.png,  #   outputs -- produced by running the notebooks above;
│       feature_pair_comparison/, etc.                #   not tracked in git (see .gitignore), regenerate locally
├── scripts/
│   ├── verify_ml_tables.py   # standalone re-run of Tables III-VI (needs an extracted feature table)
│   ├── verify_lmm.py         # standalone re-run of the LMM analysis
│   ├── verify_circos.py      # rebuilds Fig. 11 + the 36-candidate-feature selection
│   └── verify_table7.py      # standalone re-run of Table VII
├── tests/
│   └── test_pipeline_smoke.py   # generates synthetic ECG/PPG/EDA signals and runs the full
│                                 #   raw -> preprocessed -> feature pipeline end-to-end (no real data needed)
├── data/            # labels, scenario metadata, Q-score CSVs (see data/README.md)
├── docs/paper_code_mapping.md   # section-by-section mapping from the paper to this code
└── legacy/          # quarantined unused/superseded/out-of-scope code (gitignored, not part of this release)
```

**No pre-extracted feature tables are included in this repository** — only the code that produces them. See "Running the pipeline" below.

## Environment setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The LMM analysis (`notebooks/05_ml_classification_lmm_shap.ipynb`) uses `pymer4==0.8.2` (pinned — newer releases rewrote the API), which wraps R's `lme4`/`emmeans` via `rpy2`. You additionally need:
- R (>= 4.x) installed and on `PATH`
- These R packages: `install.packages(c("lme4","emmeans","lmerTest","tibble","broom","broom.mixed","insight","parameters","performance","report"))`
- Build tools for `rpy2`/`lme4` from source: a C/C++ toolchain, `cmake`, BLAS/LAPACK + Fortran (`gfortran`), and R's own dev headers — see your distro's equivalents of `libpcre2-dev`, `libdeflate-dev`, `liblzma-dev`, `libbz2-dev`, `zlib1g-dev`, `libicu-dev`, `python3-dev`, `libblas-dev`, `liblapack-dev`.

## Verifying the code works (no real data required)

```bash
python tests/test_pipeline_smoke.py
```

This generates small synthetic (but physiologically realistic, via `neurokit2`'s own signal simulators) ECG/PPG/EDA recordings in the exact folder/column layout `BioSignalDataset` expects, and runs the complete raw -> preprocessing -> windowed-feature pipeline against them, asserting that HR/HRV/EDA/PTT feature families all come out non-empty. This does **not** validate the paper's actual numbers (it uses fabricated signals) — it only confirms the extraction code itself runs correctly end-to-end without needing access to the restricted patient dataset.

## Data preparation

See [`data/README.md`](data/README.md) for what's included vs. what must be supplied separately (raw physiological recordings are not redistributed here — see the IRB note there). In short:
1. Point the raw-signal loader at your own copy of the split-trial CSV export (see `data/README.md` for the expected folder layout — the same layout `tests/test_pipeline_smoke.py` synthesizes).
2. The tracked `data/alcohol_Qscores.csv` and `data/alcohol_scenarios.csv` provide metadata. Craving labels at `data/labels/alcohol.json` are not included and must be supplied separately; see `data/README.md`.

## Running the pipeline

The pipeline is notebook-driven. Run the notebooks in their numeric order — it reflects true data dependency (notebook 02 needs notebook 01's PTT output), not just topic:

1. **Feature extraction**: `notebooks/01_feature_extraction.ipynb` — loads raw signals via `BioSignalDataset`, extracts the 152 features per 10s/50%-overlap window with the paper's PTT window (`ptt_range=(0.05, 0.7)`), and writes `features_all_10s_05_70_3.csv` under `runs/07_final_report/` (git-ignored — this is regenerated locally, not shipped in the repo; the directory keeps its original internal name for continuity with earlier verification scripts). The notebook's own later cells (below the "Feature Extraction" boundary marked inside it) are a preliminary ML sweep superseded by step 5 below — kept for historical reference, not part of the reported results.
2. **QC**: `notebooks/02_abnormal_modality_sessions.ipynb` — identifies sessions/modalities to exclude (metadata errors, missing PPG/EDA); depends on step 1's PTT output.
3. **Label sanity checks**: `notebooks/03_label_and_scenario_sanity_checks.ipynb`.
4. **Correlation analysis**: `notebooks/04_biomarker_spearman_correlations.ipynb` — QC feature filtering (used downstream) plus an exploratory correlation pass (not in the paper).
5. **LMM + ML + SHAP + candidate physiomarker selection**: `notebooks/05_ml_classification_lmm_shap.ipynb` — reproduces paper Tables III-VII and Figures 9-12 from the feature table produced in step 1.

Each notebook computes `REPO_ROOT` relative to its own location (works whether you open it from the repo root or from `notebooks/`).

## Pretrained checkpoints

Not applicable — every classifier (KNN/SVM/LR/MLP/XGB/RF) is trained from scratch inside `notebooks/05_ml_classification_lmm_shap.ipynb` on the extracted feature table; there are no separate checkpoint files to download.

## Expected output

Running the notebooks reproduces, under a `runs/` directory (git-ignored, created on demand):
- Extracted feature CSVs (`runs/07_final_report/features_all_*.csv`)
- Classification metrics (`runs/08_final_report_v2/results/**/results_metrics_all.csv`) matching paper Tables III-VII
- SHAP importance CSVs/plots and the LMM+SHAP circos-style candidate-physiomarker figure (Fig. 11)

## Data / code availability

- **Code**: this repository.
- **Raw physiological data**: not publicly available (IRB-approved clinical data collection; see `data/README.md`). A public data-access process has not been specified.
- **Paper**: submitted; publication link and DOI to be added when available.

## License

A license file has not been added.

## Citation

Citation metadata will be added when publication details are available.
