"""
Smoke test for the raw-signal -> feature-extraction pipeline.

This repo does not (and should not) ship the original raw ECG/PPG/GSR
recordings (IRB-restricted patient data). To let anyone verify that the
extraction code itself actually works -- without needing access to the
real dataset -- this test generates a small set of physiologically
plausible synthetic ECG/PPG/EDA signals (via neurokit2's own simulators),
lays them out in the exact directory/column format `BioSignalDataset`
expects, and runs the full raw -> preprocessed -> windowed-features path.

Run with:  python tests/test_pipeline_smoke.py
       or:  pytest tests/test_pipeline_smoke.py
"""
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd
import neurokit2 as nk

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.datasets.biosignal_dataset import BioSignalDataset

FS_ECG, FS_PPG = 512, 51.2
DURATION_SEC = 25  # long enough for two 10s/50%-overlap windows per trial


def _fit_len(arr, n):
    if len(arr) >= n:
        return arr[:n]
    return np.pad(arr, (0, n - len(arr)), mode="edge")


def _write_synthetic_session(root, subject_session, trials):
    for trial_folder, trial_ids in trials.items():
        for trial_id in trial_ids:
            ecg_dir = os.path.join(root, subject_session, "ECG", trial_folder)
            ppg_dir = os.path.join(root, subject_session, "PPG", trial_folder)
            os.makedirs(ecg_dir, exist_ok=True)
            os.makedirs(ppg_dir, exist_ok=True)

            rng = np.random.default_rng(abs(hash((subject_session, trial_id))) % (2**32))
            hr = int(70 + rng.integers(-5, 5))

            base_ecg = nk.ecg_simulate(duration=DURATION_SEC, sampling_rate=FS_ECG, heart_rate=hr, random_state=1)
            leads = {
                lead: base_ecg * rng.uniform(0.8, 1.2) + rng.normal(0, 0.02, size=len(base_ecg))
                for lead in ["LA_RA", "LL_LA", "LL_RA", "Vx_RL"]
            }
            t_ecg = (np.arange(len(base_ecg)) * (1000.0 / FS_ECG)).astype(np.int64) + 1_700_000_000_000
            pd.DataFrame({
                "Shimmer_820D_Timestamp_Unix_CAL": t_ecg,
                "Shimmer_820D_ECG_LA-RA_24BIT_CAL": leads["LA_RA"],
                "Shimmer_820D_ECG_LL-LA_24BIT_CAL": leads["LL_LA"],
                "Shimmer_820D_ECG_LL-RA_24BIT_CAL": leads["LL_RA"],
                "Shimmer_820D_ECG_Vx-RL_24BIT_CAL": leads["Vx_RL"],
            }).to_csv(os.path.join(ecg_dir, f"{trial_id}.csv"), index=False)

            ppg = nk.ppg_simulate(duration=DURATION_SEC, sampling_rate=FS_PPG, heart_rate=hr, random_state=1)
            eda = _fit_len(nk.eda_simulate(duration=DURATION_SEC, sampling_rate=51, scr_number=3, random_state=1), len(ppg))
            t_ppg = (np.arange(len(ppg)) * (1000.0 / FS_PPG)).astype(np.int64) + 1_700_000_000_000
            pd.DataFrame({
                "id95AE_Timestamp_Unix_CAL": t_ppg,
                "id95AE_PPG_A13_CAL": ppg,
                "id95AE_GSR_Skin_Conductance_CAL": eda,
            }).to_csv(os.path.join(ppg_dir, f"{trial_id}.csv"), index=False)


def test_raw_to_feature_pipeline():
    tmp_dir = tempfile.mkdtemp(prefix="synthetic_raw_")
    try:
        _write_synthetic_session(tmp_dir, "1_1_001_V1", {"low": ["low1"], "mid": ["mid1"], "high": ["high1"]})
        _write_synthetic_session(tmp_dir, "1_1_002_V1", {"low": ["low1"], "mid": ["mid1"], "high": ["high1"]})

        ds = BioSignalDataset(tmp_dir, "ECG", "PPG", "GSR")
        raw = ds.get_raw_data()
        assert set(raw.keys()) == {"1_1_001", "1_1_002"}, f"unexpected subjects: {raw.keys()}"

        ds.get_preprocessed_data()
        ds.get_features(window_sec=10.0, overlap_ratio=0.5, ptt_range=(0.05, 0.7))

        df_long = ds.get_feature_table_long()
        assert not df_long.empty, "get_feature_table_long() returned no rows"

        df_wide = df_long.pivot_table(index=["subject", "session", "trial"], columns=["channel", "metric"], values="value")
        df_wide.columns = [f"{c}_{m}" for c, m in df_wide.columns]
        df_wide = df_wide.reset_index()

        n_feature_cols = df_wide.shape[1] - 3
        assert n_feature_cols > 100, f"expected >100 feature columns, got {n_feature_cols}"
        assert df_wide.isna().all().sum() == 0, "found an all-NaN feature column"

        for prefix in ["HR_ECG", "HR_PPG", "HRV_ECG_", "HRV_PPG_", "EDA_", "PTT_fused_PTT_"]:
            matches = [c for c in df_wide.columns if c.startswith(prefix)]
            assert matches, f"no columns found with prefix '{prefix}' -- {sorted(df_wide.columns)[:15]}..."

        print(f"OK: extracted {n_feature_cols} feature columns across {len(df_wide)} windows from synthetic data.")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_raw_to_feature_pipeline()
    print("Smoke test passed.")
