# data/

## What's included in this repository

| File | Description |
| --- | --- |
| `alcohol_Qscores.csv` | Flattened per-trial craving self-report scores (Q1, Q2, Q_mean) for all 50 participants × 2 sessions (`Nt = 1697` trials). Derived from `labels/alcohol.json`. |
| `alcohol_scenarios.csv` | Which of the 3 VR scenarios (Fig. 2 in the paper) was used in each of the 100 sessions. |
| `labels/alcohol.json` | Nested source-of-truth for craving labels (subject → session → trial → {Q1, Q2}). *Not currently tracked by git* (see `.gitignore`) — present only if you copy it in locally. |

## What's NOT included (and why)

- **Raw physiological recordings (ECG/PPG/EDA)**: these are original clinical/research recordings from AUD patients, collected under IRB approval (Kangdong Sacred Heart Hospital IRB No. 2024-08-011; Chuncheon Sacred Heart Hospital IRB No. 2025-02-002) and are not publicly redistributable under the current approvals. **TODO (author to confirm):** whether any de-identified subset can be released, and through what request process.
- `save/raw_data.pkl`, `save/preprocessed_data.pkl`: large (≈1.8 GB each) intermediate pickles produced from the raw recordings above. Excluded for the same reason, and also simply too large for git.
- Data from a related nicotine / control-group study that shares this codebase is not part of this release.

## Expected folder structure for re-running preprocessing

The preprocessing pipeline (`src/datasets/biosignal_dataset.py`, class `BioSignalDataset`) expects raw per-subject/session CSVs exported from the Shimmer3 ECG/GSR+ devices, laid out as:

```
<SPLIT_DATA_DIR>/
├── 1_1_001_V1/                  # <subject>_<session>
│   ├── ECG/
│   │   ├── low/{low1,low2,...}.csv     # 4-channel ECG (Shimmer_820D_ECG_*)
│   │   ├── mid/{mid1,...}.csv
│   │   └── high/{high1,...}.csv
│   └── PPG/                     # same trial csv also carries the GSR channel
│       ├── low/{low1,low2,...}.csv     # id95AE_PPG_A13_CAL, id95AE_GSR_Skin_Conductance_CAL
│       ├── mid/{mid1,...}.csv
│       └── high/{high1,...}.csv
├── 1_1_001_V2/
│   └── ...
└── ...
```

Point `SPLIT_DATA_DIR` at your own copy of this structure. `tests/test_pipeline_smoke.py` generates a tiny synthetic dataset in exactly this layout (with the exact column names `BioSignalDataset` expects) if you want a concrete, runnable example — see its `_write_synthetic_session` helper.

## Privacy / IRB note

Subject identifiers used throughout (`1_1_001`, etc.) are already de-identified codes, not names. No names, contact details, or other direct identifiers were found in the tracked data files. If you add any raw or intermediate data locally for your own analysis, keep it out of git (already covered by `.gitignore`) and follow your own institution's data handling requirements.
