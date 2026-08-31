"""Shared utilities for the ML/LMM/SHAP notebooks (feature loading, metrics,
SHAP unwrapping). Extracted from notebook 08's inline definitions, which were
duplicated verbatim into notebook 09 -- see docs/paper_code_mapping.md for the
one known, deliberately-preserved discrepancy in the abnormal-session sets
below (MISSING_GSR does not match notebooks 00/02's fuller determination;
kept as-is because it reproduces the paper's actual reported numbers).

Not included here (left inline in the notebooks that use them, per the
code-cleanup audit): run_full_experiment/run_experiment_no_shap (differing
model rosters across notebooks -- not safe to merge), extract_pairwise_pvalues
(two versions with different return contracts already coexist in notebook 08),
and the plotting helpers (save_compare_figure, save_cm_refined, etc. -- not
currently duplicated anywhere, so extracting them isn't deduplication).
"""

import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from xgboost import DMatrix

# Abnormal signal session exclusion lists (from Notebook 02).
SAMPLING_RATE_ERROR = {'1_1_030_V1', '1_1_031_V1', '1_1_032_V1', '1_1_033_V1'}
MISSING_PPG = {'1_1_001_V1', '1_1_001_V2', '1_1_006_V2', '1_1_009_V1', '1_1_009_V2', '1_1_016_V2'}
MISSING_GSR = {
    '1_1_002_V1', '1_1_002_V2', '1_1_005_V1', '1_1_005_V2', '1_1_010_V1',
    '1_1_011_V1', '1_1_012_V2', '1_1_013_V2', '1_1_016_V1', '1_1_016_V2',
    '1_1_017_V1', '1_1_017_V2', '1_1_018_V2', '1_1_019_V1', '1_1_020_V2',
    '1_2_002_V2', '1_2_005_V1', '1_2_005_V2', '1_2_006_V1', '1_2_006_V2',
    '1_2_008_V2', '1_2_009_V1', '1_2_013_V1'
}


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def get_feature_cols_by_modality(all_features, modality):
    if modality == 'ALL': return all_features
    mod = modality.lower()
    if mod == 'ecg': return [c for c in all_features if 'ecg' in c.lower()]
    if mod == 'ppg': return [c for c in all_features if 'ppg' in c.lower()]
    if mod == 'eda': return [c for c in all_features if c.lower().startswith('eda')]
    if mod == 'ptt': return [c for c in all_features if c.lower().startswith('ptt')]
    return [c for c in all_features if mod in c.lower()]


def load_and_label(dur_name, repo_root, q_csv_path, n_classes=2, filter_abnormal=True, prefix="5", compare='z_vs_lh', selected_feature_csv=None):
    feat_csv = repo_root / 'runs' / '07_final_report' / f'features_all_{dur_name}{prefix}.csv'
    if not feat_csv.exists(): return None, None
    df_feat = pd.read_csv(feat_csv)

    df_feat.columns = df_feat.columns.str.replace('^EDA_EDA_', 'EDA_', regex=True)
    df_feat.columns = df_feat.columns.str.replace('^HRV_', '', regex=True)
    df_feat.columns = df_feat.columns.str.replace('^PTT_fused_PTT_', 'PTT_', regex=True)
    df_feat.columns = df_feat.columns.str.replace('^HR_ECG', 'ECG_HR', regex=True)
    df_feat.columns = df_feat.columns.str.replace('^RR_ECG', 'ECG_RR', regex=True)
    df_feat.columns = df_feat.columns.str.replace('^SI_ECG', 'ECG_SI', regex=True)
    df_feat.columns = df_feat.columns.str.replace('^HR_PPG', 'PPG_HR', regex=True)
    df_feat.columns = df_feat.columns.str.replace('^RR_PPG', 'PPG_RR', regex=True)
    df_feat.columns = df_feat.columns.str.replace('^SI_PPG', 'PPG_SI', regex=True)

    df_feat['subject'] = df_feat['subject'].astype(str).str.strip()
    df_feat['trial'] = df_feat['trial'].astype(str).str.strip().apply(lambda x: x.split('_w')[0] if '_w' in x else x)
    df_q = pd.read_csv(q_csv_path)
    df_q['subject'] = df_q['subject'].astype(str).str.strip()
    df_q['version'] = df_q['version'].astype(str).str.strip()
    df_q['trial'] = df_q['trial'].astype(str).str.strip()

    if 'session' in df_feat.columns:
        df_feat = df_feat.rename(columns={'session': 'version'})

    df_feat['version'] = df_feat['version'].astype(str).str.strip()
    df = df_feat.merge(df_q, on=['subject', 'version', 'trial'], how='inner')

    # -------------------------------
    # 1. Abnormal -> NULL out affected features only (no row drop)
    # -------------------------------
    df['session_id'] = df['subject'] + "_" + df['version']

    if filter_abnormal:
        # ECG sampling-rate error -> NULL ECG features
        bad_ecg_sessions = SAMPLING_RATE_ERROR
        ecg_cols = get_feature_cols_by_modality(df.columns, 'ecg')
        df.loc[df['session_id'].isin(bad_ecg_sessions), ecg_cols] = np.nan

        # Missing PPG -> NULL PPG features
        bad_ppg_sessions = MISSING_PPG
        ppg_cols = get_feature_cols_by_modality(df.columns, 'ppg')
        df.loc[df['session_id'].isin(bad_ppg_sessions), ppg_cols] = np.nan

        # Missing EDA(GSR) -> NULL EDA features
        bad_gsr_sessions = MISSING_GSR
        eda_cols = get_feature_cols_by_modality(df.columns, 'eda')
        df.loc[df['session_id'].isin(bad_gsr_sessions), eda_cols] = np.nan

        # PTT abnormal if ECG sampling error or PPG missing -> NULL PTT too
        bad_ptt_sessions = set(SAMPLING_RATE_ERROR) | set(MISSING_PPG)
        ptt_cols = get_feature_cols_by_modality(df.columns, 'ptt')
        df.loc[df['session_id'].isin(bad_ptt_sessions), ptt_cols] = np.nan

        print(f"  [Filter] NULL applied to abnormal sessions (no row drop)")

    # -------------------------------
    # 2. Label construction
    # -------------------------------
    q_vals = pd.to_numeric(df['Q_mean'], errors='coerce')
    df['y'] = np.nan
    if n_classes == 2:
        if compare == "z_vs_lh":
            df['y'] = np.where(q_vals == 0, 0, 1)
        elif compare == "zl_vs_h":
            df['y'] = np.where(q_vals <= 3.5, 0, 1)
        elif compare == "z_vs_h":
            df.loc[q_vals == 0, 'y'] = 0
            df.loc[q_vals > 3.5, 'y'] = 1

    else:
        df['y'] = np.where(q_vals == 0, 0, np.where(q_vals > 3.5, 2, 1))

    df['group_id'] = df['subject'].astype(str) + "_" + df['version'].astype(str) + "_" + df['trial'].astype(str)

    exclude = ['subject', 'version', 'trial', 'y', 'Q1', 'Q2', 'Q_mean', 'intensity',
               'trial_idx', 'group_id', 'trial_0base', 'orig_trial', 'session_id']
    feats = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude]

    # inf, -inf -> NaN
    df[feats] = df[feats].replace([np.inf, -np.inf], np.nan)

    # Drop feature columns that are mostly missing or constant.
    threshold = 0.5
    valid_feats = []

    for c in feats:
        if 'Symbolic' in c:
            print(f"Skipping {c}: Symbolic_EqualProb")
            continue

        nan_ratio = df[c].isna().mean()

        if nan_ratio > threshold or df[c].isna().all():
            print(f"Skipping {c}: NaN ratio {nan_ratio:.2%}")
            continue

        if df[c].dropna().nunique() <= 1:
            print(f"Skipping {c}: Constant value")
            continue

        valid_feats.append(c)

    # Optional additional filter against a pre-selected feature list.
    if selected_feature_csv is not None:
        df_sel = pd.read_csv(selected_feature_csv)

        if "feature" not in df_sel.columns:
            raise ValueError(f"'feature' column not found in {selected_feature_csv}")

        selected_feats = df_sel["feature"].dropna().astype(str).tolist()

        valid_feats = [c for c in valid_feats if c in selected_feats]

        if len(valid_feats) == 0:
            raise ValueError("No overlapping features between valid_feats and selected feature list.")

        print(f"[Selected feature filter] kept {len(valid_feats)} features from {selected_feature_csv}")

    # -------------------------------
    # 3. Row-level: drop if any valid feature is NaN
    # -------------------------------
    n_before = len(df)
    df = df.dropna(subset=valid_feats).reset_index(drop=True)
    n_after = len(df)
    df = df.dropna(subset=['y']).reset_index(drop=True)

    print(f"[Row drop] {n_before} -> {n_after} (dropped {n_before - n_after})")

    return df, valid_feats


def compute_comprehensive_metrics(y_true, y_pred, y_proba, classes):
    nc = len(classes)
    res = {
        'Acc': accuracy_score(y_true, y_pred),
        'BAcc': balanced_accuracy_score(y_true, y_pred),
        'MCC': matthews_corrcoef(y_true, y_pred),
        'F1_Macro': f1_score(y_true, y_pred, average='macro'),
        'F1_Weighted': f1_score(y_true, y_pred, average='weighted')
    }

    if y_proba is not None:
        try:
            if nc == 2:
                p1 = y_proba[:, 1]
                res['AUROC'] = roc_auc_score(y_true, p1)
                res['AUPRC'] = average_precision_score(y_true, p1)
                res['Brier'] = brier_score_loss(y_true, p1)
                res['LogLoss'] = log_loss(y_true, y_proba, labels=classes)
            else:
                Y = label_binarize(y_true, classes=classes)
                res['AUROC'] = roc_auc_score(
                    Y, y_proba, average='macro', multi_class='ovr'
                )
                res['AUPRC'] = average_precision_score(
                    Y, y_proba, average='macro'
                )
                res['Brier'] = np.mean(np.sum((y_proba - Y) ** 2, axis=1))
                res['LogLoss'] = log_loss(y_true, y_proba, labels=classes)
        except Exception:
            pass

    return res


def map_y_to_group(y):
    s = str(y).strip().lower()
    mapping = {
        "0": "0", "zero": "0",
        "1": "Low", "low": "Low",
        "2": "High", "high": "High",
    }
    return mapping.get(s, np.nan)


def unwrap_shap(s):
    if hasattr(s, "values"): s = s.values
    if isinstance(s, (list, np.ndarray)) and len(s) > 0:
        first = s[0];
        while isinstance(first, (list, np.ndarray)) and len(first) > 0: first = first[0]
        if isinstance(first, str) and "[" in first:
            def p(o):
                if not isinstance(o, str): return o
                ns = re.findall(r"[-+]?\d*\.\d+[eE][-+]?\d+|[-+]?\d+\.\d+|[-+]?\d+", o)
                return float(ns[0]) if len(ns) == 1 else [float(x) for x in ns]
            f = np.array(s).ravel(); px = np.array([p(z) for z in f])
            if px.ndim == 2 and px.shape[1] > 1: s = px.reshape(s.shape[0], -1, px.shape[1])
            else: s = px.reshape(s.shape)
    if isinstance(s, np.ndarray):
        if s.ndim == 2: return [s]
        if s.ndim == 3: return [s[:, :, i] for i in range(s.shape[2])]
    if isinstance(s, list): return [np.asarray(v.values if hasattr(v, "values") else v) for v in s]
    return [s]


def get_xgb_shap(pipe, Xs):
    cl = pipe.named_steps['clf']; dt = DMatrix(Xs); sv = cl.get_booster().predict(dt, pred_contribs=True)
    if sv.ndim == 2: return [sv[:, :-1]]
    return [sv[:, i, :-1] for i in range(sv.shape[1])]
