# scripts/

Standalone verification scripts extracted from `notebooks/05_ml_classification_lmm_shap.ipynb`, used to confirm (by actually re-running the code) that this repository's code reproduces the paper's Tables III–VII and Fig. 11.

Run in this order from the repo root (each depends on the previous one's output):

```bash
python scripts/verify_ml_tables.py   # Tables III-VI (ML sweep + modality ablation)
python scripts/verify_lmm.py         # LMM analysis (needs R + lme4/emmeans/etc., see requirements.txt)
python scripts/verify_circos.py      # Fig. 11 + 36-candidate-feature selection (needs the two scripts above)
python scripts/verify_table7.py      # Table VII, 36-feature re-run (needs verify_circos.py's output)
```

Requires `runs/07_final_report/features_all_10s_05_70_3.csv` (see `notebooks/01_feature_extraction.ipynb` — the `runs/07_final_report` directory keeps its original internal name). Outputs go under `runs/08_final_report_v2/` and `notebooks/lmm_results_verify/` (both git-ignored).
