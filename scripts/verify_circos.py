import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import shap
import warnings
import re
from tqdm.notebook import tqdm

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, 
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, precision_recall_curve, 
    average_precision_score, brier_score_loss, matthews_corrcoef, log_loss,
    precision_recall_fscore_support
)
from sklearn.model_selection import StratifiedGroupKFold, LeaveOneGroupOut
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier, DMatrix
from sklearn.calibration import calibration_curve

warnings.filterwarnings('ignore')
SEED = 42
def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(SEED)

# Project Paths
# notebook is assumed to be in repo root or under notebooks/
NB_DIR = Path.cwd()
REPO_ROOT = NB_DIR if (NB_DIR / "src").exists() else NB_DIR.parent
DATA_DIR = REPO_ROOT / 'data'
Q_CSV = DATA_DIR / 'alcohol_Qscores.csv'
RUNS_DIR = REPO_ROOT / 'runs' / '08_final_report_v2'
RUNS_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_ROOT = RUNS_DIR / 'results'
EXPORT_ROOT.mkdir(parents=True, exist_ok=True)

def get_save_paths(nc, modality):
    base = EXPORT_ROOT / f"{nc}class" / modality
    paths = {
        'base': base,
        'plots': base / 'plots',
        'shap': base / 'shap',
        'csvs': base / 'csvs'
    }
    for p in paths.values(): p.mkdir(parents=True, exist_ok=True)
    return paths

import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

# Performance Plot Aesthetics (Large Fonts)
plt.rcParams.update({'font.size': 14, 'axes.labelsize': 16, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12})


# ==== next cell ====

# Abnormal signal session exclusion lists (from Notebook 02)
SAMPLING_RATE_ERROR = {'1_1_030_V1', '1_1_031_V1', '1_1_032_V1', '1_1_033_V1'}
MISSING_PPG = {'1_1_001_V1', '1_1_001_V2', '1_1_006_V2', '1_1_009_V1', '1_1_009_V2', '1_1_016_V2'}
MISSING_GSR = {
    '1_1_002_V1', '1_1_002_V2', '1_1_005_V1', '1_1_005_V2', '1_1_010_V1',
    '1_1_011_V1', '1_1_012_V2', '1_1_013_V2', '1_1_016_V1', '1_1_016_V2',
    '1_1_017_V1', '1_1_017_V2', '1_1_018_V2', '1_1_019_V1', '1_1_020_V2',
    '1_2_002_V2', '1_2_005_V1', '1_2_005_V2', '1_2_006_V1', '1_2_006_V2',
    '1_2_008_V2', '1_2_009_V1', '1_2_013_V1'
}

def get_feature_cols_by_modality(all_features, modality):
    if modality == 'ALL': return all_features
    mod = modality.lower()
    if mod == 'ecg': return [c for c in all_features if 'ecg' in c.lower()]
    if mod == 'ppg': return [c for c in all_features if 'ppg' in c.lower()]
    if mod == 'eda': return [c for c in all_features if c.lower().startswith('eda')]
    if mod == 'ptt': return [c for c in all_features if c.lower().startswith('ptt')]
    return [c for c in all_features if mod in c.lower()]

def load_and_label(dur_name, n_classes=2, filter_abnormal=True, prefix="5", compare='z_vs_lh', selected_feature_csv=None):
    feat_csv = REPO_ROOT / 'runs' / '07_final_report' / f'features_all_{dur_name}{prefix}.csv'
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
    df_q = pd.read_csv(Q_CSV)
    df_q['subject'] = df_q['subject'].astype(str).str.strip()
    df_q['version'] = df_q['version'].astype(str).str.strip()
    df_q['trial'] = df_q['trial'].astype(str).str.strip()

    if 'session' in df_feat.columns:
        df_feat = df_feat.rename(columns={'session': 'version'})

    df_feat['version'] = df_feat['version'].astype(str).str.strip()
    df = df_feat.merge(df_q, on=['subject', 'version', 'trial'], how='inner')

    # -------------------------------
    # 1. Abnormal -> 특정 feature만 NULL 처리
    # -------------------------------
    df['session_id'] = df['subject'] + "_" + df['version']

    if filter_abnormal:
        # ECG sampling rate 오류 → ECG feature를 NULL
        bad_ecg_sessions = SAMPLING_RATE_ERROR
        ecg_cols = get_feature_cols_by_modality(df.columns, 'ecg')
        df.loc[df['session_id'].isin(bad_ecg_sessions), ecg_cols] = np.nan

        # PPG 누락 → PPG feature NULL
        bad_ppg_sessions = MISSING_PPG
        ppg_cols = get_feature_cols_by_modality(df.columns, 'ppg')
        df.loc[df['session_id'].isin(bad_ppg_sessions), ppg_cols] = np.nan

        # EDA(GSR) 누락 → EDA feature NULL
        bad_gsr_sessions = MISSING_GSR
        eda_cols = get_feature_cols_by_modality(df.columns, 'eda')
        df.loc[df['session_id'].isin(bad_gsr_sessions), eda_cols] = np.nan

        # PTT abnormal -> ECG sampling error 또는 PPG missing 이면 PTT도 NULL
        bad_ptt_sessions = set(SAMPLING_RATE_ERROR) | set(MISSING_PPG)
        ptt_cols = get_feature_cols_by_modality(df.columns, 'ptt')
        df.loc[df['session_id'].isin(bad_ptt_sessions), ptt_cols] = np.nan

        print(f"  [Filter] NULL applied to abnormal sessions (row drop 없음)")

    # -------------------------------
    # 2. Label 생성
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
        df['y'] = np.where(q_vals == 0, 0, np.where(q_vals>3.5, 2, 1))

    df['group_id'] = df['subject'].astype(str) + "_" + df['version'].astype(str) + "_" + df['trial'].astype(str)

    
    exclude = ['subject','version','trial','y','Q1','Q2','Q_mean','intensity',
               'trial_idx','group_id','trial_0base','orig_trial','session_id']
    feats = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude]

    # inf, -inf → NaN
    df[feats] = df[feats].replace([np.inf, -np.inf], np.nan)

    # feature-wise fill
    # 설정값: 결측치가 몇 % 이상일 때 버릴 것인가? (예: 0.5 = 50%)
    threshold = 0.5
    valid_feats = []

    for c in feats:
        if 'Symbolic' in c: 
            print(f"Skipping {c}: Symbolic_EqualProb")
            continue

         # 1. 해당 컬럼의 결측치 비율 계산
        nan_ratio = df[c].isna().mean()
        
        # 2. 결측치가 너무 많거나, 모든 값이 NaN인 경우 제외
        if nan_ratio > threshold or df[c].isna().all():
            print(f"Skipping {c}: NaN ratio {nan_ratio:.2%}")
            continue
        
        # 3. (선택사항) 모든 값이 동일한 경우도 분석 가치가 없으므로 제외
        if df[c].dropna().nunique() <= 1:
            print(f"Skipping {c}: Constant value")
            continue

        # 4. 결측치 채우기 및 유효 리스트 추가
        # df[c] = df[c].fillna(df[c].median())
        valid_feats.append(c)

    # ---------------------------------------
    # 여기서 selected feature list로 한 번 더 필터
    # ---------------------------------------
    if selected_feature_csv is not None:
        df_sel = pd.read_csv(selected_feature_csv)

        if "feature" not in df_sel.columns:
            raise ValueError(f"'feature' column not found in {selected_feature_csv}")

        selected_feats = df_sel["feature"].dropna().astype(str).tolist()

        # 현재 데이터에 실제 존재하고, 기존 valid_feats에도 포함된 feature만 유지
        valid_feats = [c for c in valid_feats if c in selected_feats]

        if len(valid_feats) == 0:
            raise ValueError("No overlapping features between valid_feats and selected feature list.")

        print(f"[Selected feature filter] kept {len(valid_feats)} features from {selected_feature_csv}")

    # -------------------------------
    # row 기준: valid_feats 중 하나라도 NaN이면 제거
    # -------------------------------
    n_before = len(df)
    df = df.dropna(subset=valid_feats).reset_index(drop=True)
    n_after = len(df)
    df = df.dropna(subset=['y']).reset_index(drop=True)

    print(f"[Row drop] {n_before} -> {n_after} (dropped {n_before - n_after})")
    
    return df, valid_feats

# ============================================================
#  Metric computation
# ============================================================
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
                res['Brier'] = np.mean(np.sum((y_proba - Y)**2, axis=1))
                res['LogLoss'] = log_loss(y_true, y_proba, labels=classes)
        except:
            pass

    return res

# ============================================================
#  Plotting (Binary)
# ============================================================
def _plot_curves_binary(ax_roc, ax_pr, ax_cal, y_true, proba, label):
    if isinstance(proba, list):
        proba = np.array(proba)

    p1 = proba[:, 1]

    # ROC
    if ax_roc is not None:
        fpr, tpr, _ = roc_curve(y_true, p1)
        auc = roc_auc_score(y_true, p1)
        ax_roc.plot(fpr, tpr, lw=2, label=f"{label} (AUROC={auc:.2f})")

    # PR
    if ax_pr is not None:
        prec, rec, _ = precision_recall_curve(y_true, p1)
        ap = average_precision_score(y_true, p1)
        ax_pr.plot(rec, prec, lw=2, label=f"{label} (AUPRC={ap:.2f})")

    # Calibration
    if ax_cal is not None:
        f_pos, m_pred = calibration_curve(y_true, p1, n_bins=10)
        brier = brier_score_loss(y_true, p1)
        ax_cal.plot(m_pred, f_pos, marker="o", lw=1.5, label=f"{label} (Brier score loss={brier:.3f})")

# ============================================================
#  Plotting (Multiclass)
# ============================================================
def _plot_curves_multiclass(ax_roc, ax_pr, ax_cal, y_true, proba, classes, label, show_classwise=False):
    if isinstance(proba, list):
        proba = np.array(proba)

    Y = label_binarize(y_true, classes=classes)
    n_classes = len(classes)

    # ---------------------------
    # OVR 세부 선 (옵션)
    # ---------------------------
    if show_classwise:
        for ci in range(n_classes):
            if np.unique(Y[:, ci]).size < 2:
                continue
            # ROC
            if ax_roc is not None:
                fpr, tpr, _ = roc_curve(Y[:, ci], proba[:, ci])
                ax_roc.plot(fpr, tpr, lw=1, alpha=0.12, color='gray')

            # PR
            if ax_pr is not None:
                prec, rec, _ = precision_recall_curve(Y[:, ci], proba[:, ci])
                ax_pr.plot(rec, prec, lw=1, alpha=0.12, color='gray')

    # ---------------------------
    # Macro ROC
    #---------------------------
    if ax_roc is not None:
        all_fpr = np.linspace(0, 1, 200)
        mean_tpr = np.zeros_like(all_fpr)

        for ci in range(n_classes):
            if np.unique(Y[:, ci]).size < 2:
                continue
            fpr, tpr, _ = roc_curve(Y[:, ci], proba[:, ci])
            mean_tpr += np.interp(all_fpr, fpr, tpr)

        mean_tpr /= n_classes
        mauc = roc_auc_score(Y, proba, average='macro', multi_class='ovr')

        ax_roc.plot(all_fpr, mean_tpr, lw=2, label=f"{label} (AUROC={mauc:.2f})")

    # ---------------------------
    # Macro PR
    #---------------------------
    if ax_pr is not None:
        all_rec = np.linspace(0, 1, 200)
        mean_prec = np.zeros_like(all_rec)

        for ci in range(n_classes):
            if np.unique(Y[:, ci]).size < 2:
                continue
            prec, rec, _ = precision_recall_curve(Y[:, ci], proba[:, ci])
            mean_prec += np.interp(all_rec, rec[::-1], prec[::-1])

        mean_prec /= n_classes
        mapc = average_precision_score(Y, proba, average='macro')

        ax_pr.plot(all_rec, mean_prec, lw=2, label=f"{label} (AUPRC={mapc:.2f})")

    # Calibration (multiclass는 표시 불가 - Brier score만 표시)
    if ax_cal is not None:
        brier = np.mean(np.sum((proba - Y)**2, axis=1))
        ax_cal.plot([], [], label=f"{label} (Brier score loss={brier:.3f})")

# ============================================================
#  Save Figure
# ============================================================
def save_compare_figure(curves_dict, classes, out_path_base, title_prefix=""):
    # --- 핵심 패치: 모든 y_true 를 numpy array로 변환 ---
    new_dict = {}
    for k, (yt, pb) in curves_dict.items():
        new_dict[k] = (np.asarray(yt), np.asarray(pb))
    curves_dict = new_dict

    # baseline 준비
    yt_any = list(curves_dict.values())[0][0]
    pos_rate = (yt_any == 1).mean() if len(classes) == 2 else label_binarize(yt_any, classes=classes).mean(axis=0).mean()

    # 1. Combined Figure (Legacy)
    plt.figure(figsize=(15, 5), dpi=600)
    ax1, ax2, ax3 = plt.subplot(1, 3, 1), plt.subplot(1, 3, 2), plt.subplot(1, 3, 3)
    ax1.plot([0, 1], [0, 1], "--", color='gray', label='Random Classifier')
    ax2.hlines(pos_rate, 0, 1, colors='gray', linestyles='--', label='Random Classifier')
    ax3.plot([0, 1], [0, 1], "--", color='gray', label='Perfectly Calibrated')

    for m, (yt, pb) in curves_dict.items():
        if len(classes) == 2:
            _plot_curves_binary(ax1, ax2, ax3, yt, pb, m)
        else:
            _plot_curves_multiclass(ax1, ax2, ax3, yt, pb, classes, m)

    # ax1.set_title("ROC Curve")
    ax1.set_xlabel("False Positive Rate", fontsize=16); ax1.set_ylabel("True Positive Rate", fontsize=16); ax1.legend(fontsize=14)
    # ax2.set_title("Precision-Recall Curve"); 
    ax2.set_xlabel("Recall", fontsize=16); ax2.set_ylabel("Precision", fontsize=16); ax2.legend(fontsize=14)
    # ax3.set_title("Calibration Curve"); 
    ax3.set_xlabel("Mean predicted value", fontsize=16); ax3.set_ylabel("Fraction of positives", fontsize=16); ax3.legend(fontsize=14)

    # if title_prefix:
        # plt.suptitle(title_prefix)

    plt.tight_layout()
    plt.savefig(out_path_base / 'curves_comparison.png')
    plt.close()

    # 2. Individual Figures
    # ROC
    plt.figure(figsize=(5, 5), dpi=600)
    ax = plt.gca()
    ax.plot([0, 1], [0, 1], "--", color='gray', label='Random Classifier')
    for m, (yt, pb) in curves_dict.items():
        if len(classes) == 2: _plot_curves_binary(ax, None, None, yt, pb, m)
        else: _plot_curves_multiclass(ax, None, None, yt, pb, classes, m)
    # ax.set_title(f"ROC Curve\n{title_prefix}"); 
    ax.set_xlabel("False Positive Rate", fontsize=16); ax.set_ylabel("True Positive Rate", fontsize=16); ax.legend(fontsize=14)
    plt.tight_layout(); plt.savefig(out_path_base / 'roc_curve.png', bbox_inches="tight"); plt.close()

    # PR
    plt.figure(figsize=(5, 5), dpi=600)
    ax = plt.gca()
    ax.hlines(pos_rate, 0, 1, colors='gray', linestyles='--', label='Random Classifier')
    for m, (yt, pb) in curves_dict.items():
        if len(classes) == 2: _plot_curves_binary(None, ax, None, yt, pb, m)
        else: _plot_curves_multiclass(None, ax, None, yt, pb, classes, m)
    # ax.set_title(f"Precision-Recall Curve\n{title_prefix}"); 
    ax.set_xlabel("Recall", fontsize=18); ax.set_ylabel("Precision", fontsize=18); ax.legend(fontsize=12)
    plt.tight_layout(); plt.savefig(out_path_base / 'pr_curve.png', bbox_inches="tight"); plt.close()

    # Calibration
    plt.figure(figsize=(5, 5), dpi=600)
    ax = plt.gca()
    ax.plot([0, 1], [0, 1], "--", color='gray', label='Perfectly Calibrated')
    for m, (yt, pb) in curves_dict.items():
        if len(classes) == 2: _plot_curves_binary(None, None, ax, yt, pb, m)
        else: _plot_curves_multiclass(None, None, ax, yt, pb, classes, m)
    # ax.set_title(f"Calibration Curve\n{title_prefix}"); 
    ax.set_xlabel("Mean predicted value", fontsize=18); ax.set_ylabel("Fraction of positives", fontsize=18); ax.legend(fontsize=12)
    plt.tight_layout(); plt.savefig(out_path_base / 'calibration_curve.png'); plt.close()

# ============================================================
#  Confusion matrix refined
# ============================================================
def save_cm_refined(y_true, y_pred, classes, path, title, display_labels=None):
    conf_mat = confusion_matrix(y_true, y_pred, labels=classes)
    cm_norm = conf_mat / np.maximum(conf_mat.sum(axis=1, keepdims=True), 1e-12) * 100

    tick_labels = display_labels if display_labels is not None else classes

    plt.figure(figsize=(5, 5), dpi=600)
    annot = np.empty_like(conf_mat).astype(str)

    for i in range(conf_mat.shape[0]):
        row_vals = cm_norm[i].copy()
        rounded = np.round(row_vals, 1)

        # 반올림 오차 보정: 마지막 칸에 차이 반영
        diff = np.round(100.0 - rounded.sum(), 1)
        rounded[-1] = np.round(rounded[-1] + diff, 1)

        for j in range(conf_mat.shape[1]):
            annot[i, j] = f"{rounded[j]:.1f}%"

    ax = sns.heatmap(
        data=cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        annot_kws={"size": 22, "weight": "bold"},
        cbar=False
    )

    plt.xlabel("Predicted", fontsize=18)
    plt.ylabel("True", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    
def _unwrap_shap(s, nc):
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

def run_full_experiment(df, feats, model_name, paths, nc_name, mod_name):
           
    X, y, groups = df[feats], df['y'], df['group_id']; classes_list = sorted(np.unique(y)); nc = len(classes_list); 
    
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)

    yt_all, yp_all, pb_all, s_all = [], [], [], []
    for f, (tr, te) in enumerate(cv.split(X, y, groups)):
        Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
        # 모델 설정 부분
        if model_name == 'LR':
            clf = LogisticRegression(solver='saga', penalty='l2', max_iter=8000, random_state=SEED, class_weight='balanced')

        elif model_name == 'MLP':
            # 2-layer FC layer (예: 64개 노드 -> 32개 노드)
            clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, early_stopping=True, n_iter_no_change=30, random_state=SEED)

        elif model_name == 'RF':
            clf = RandomForestClassifier(n_estimators=500, random_state=SEED, class_weight='balanced_subsample', n_jobs=-1)

        elif model_name == 'XGB':
            clf = XGBClassifier(n_estimators=100, random_state=SEED, n_jobs=-1, objective='multi:softprob' if nc > 2 else 'binary:logistic')

        elif model_name == 'SVM':
            # Calibration Curve를 그리려면 probability=True 설정이 반드시 필요합니다.
            clf = SVC(kernel='rbf', probability=True, random_state=SEED, class_weight='balanced')

        elif model_name == 'KNN':
            # KNN은 별도의 random_state가 없으며, n_neighbors는 데이터 특성에 따라 조절이 필요합니다.
            clf = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        pe = Pipeline([('scaler', StandardScaler()), ('clf', clf)]); pe.fit(Xtr, ytr)
        yt_all.extend(yte); yp_all.extend(pe.predict(Xte)); pb_all.extend(pe.predict_proba(Xte))
        if model_name in ['LR', 'XGB', 'RF']:
            try:
                Xtes = pe.named_steps['scaler'].transform(Xte)
                if model_name == 'XGB': sl = get_xgb_shap(pe, Xtes)
                elif model_name == 'RF':
                    sl_raw = shap.TreeExplainer(pe.named_steps['clf']).shap_values(Xtes, approximate=True, check_additivity=False)
                    sl = _unwrap_shap(sl_raw, nc)
                else: 
                    bX = pe.named_steps['scaler'].transform(Xtr.sample(min(100, len(Xtr)), random_state=SEED))
                    sl = _unwrap_shap(shap.LinearExplainer(pe.named_steps['clf'], bX).shap_values(Xtes), nc)
                s_all.append({'X': Xtes, 'S': sl})
            except Exception as e: print(f"  [SHAP Error Fold {f+1}] {e}")
    
    pb_all = np.array(pb_all); 
    
    display_names = ['zero', 'low', 'high'] if nc==3 else ['zero', 'non-zero']

    save_cm_refined(yt_all, yp_all, classes_list, 
                    paths['plots'] / f'cm_{model_name}.png', 
                    f"{model_name} | {nc_name} | {mod_name}",
                    display_labels=display_names)
                    
    if s_all:
        Xc = np.concatenate([d['X'] for d in s_all], axis=0)
        nsc = len(s_all[0]['S'])

        # --------------------------------------------------
        # Consolidated SHAP importance CSV (all folds pooled)
        # --------------------------------------------------
        if nc > 2 and nsc > 1:
            list_Sc = [np.concatenate([d['S'][ci] for d in s_all], axis=0) for ci in range(nsc)]
            mabs_all = np.mean([np.mean(np.abs(s), axis=0) for s in list_Sc], axis=0)
        else:
            Sc_all = np.concatenate([d['S'][0] for d in s_all], axis=0)
            mabs_all = np.mean(np.abs(Sc_all), axis=0)

        pd.DataFrame({
            'feature': feats,
            'mean_abs_shap': mabs_all
        }).sort_values(
            'mean_abs_shap',
            ascending=False
        ).to_csv(
            paths['csvs'] / f'shap_importance_{model_name}_all_classes.csv',
            index=False
        )

        # Multiclass Consolidated Plot (Bar)
        if nc > 2 and nsc > 1:
            list_Sc = [np.concatenate([d['S'][ci] for d in s_all], axis=0) for ci in range(nsc)]

            desired_classes = [0, 1, 2]
            name_map = {
                0: "zero",
                1: "low",
                2: "high",
            }

            class_to_idx = {c: i for i, c in enumerate(classes_list)}
            order_idx = [class_to_idx[c] for c in desired_classes]

            list_Sc_ordered = [list_Sc[i] for i in order_idx]
            class_names_ordered = [name_map[c] for c in desired_classes]

            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                list_Sc_ordered,
                Xc,
                feature_names=feats,
                show=False,
                max_display=20,
                plot_type="bar",
                class_names=class_names_ordered,
                class_inds="original"
            )
            plt.tight_layout()
            plt.savefig(paths['shap'] / f'shap_bar_{model_name}_all_classes.png', dpi=600)
            plt.close()
        
        # Per-class plots (Dots always, Bar/CSV only if not consolidated)
        for ci in range(nsc):
            Sc = np.concatenate([d['S'][ci] for d in s_all], axis=0)
            target = classes_list[ci] if nsc > 1 else (classes_list[1] if nc == 2 else classes_list[0])
            mabs = np.mean(np.abs(Sc), axis=0)
            
            if not (nc > 2): # For binary, still save the main class bar/csv
                pd.DataFrame({'feature': feats, 'mean_abs_shap': mabs}).sort_values('mean_abs_shap', ascending=False).to_csv(paths['csvs'] / f'shap_importance_{model_name}_class{target}.csv', index=False)
                plt.figure(figsize=(10, 8)); shap.summary_plot(Sc, Xc, feature_names=feats, show=False, max_display=20, plot_type="bar"); 
                # plt.title(f"SHAP Importance: {model_name} | {mod_name} | Class {target}"); 
                plt.tight_layout(); plt.savefig(paths['shap'] / f'shap_bar_{model_name}_class{target}.png', dpi=600); plt.close()

                # # Consolidated CSV 
                # mabs_all = np.mean([np.mean(np.abs(s), axis=0) for s in list_Sc], axis=0) 
                # pd.DataFrame({'feature': feats, 'mean_abs_shap': mabs_all}).sort_values('mean_abs_shap', ascending=False).to_csv(paths['csvs'] / f'shap_importance_{model_name}_all_classes.csv', index=False)
            
            # Beeswarm is always useful per class
            plt.figure(figsize=(10, 8)); shap.summary_plot(Sc, Xc, feature_names=feats, show=False, max_display=20, plot_type="dot"); 
            # plt.title(f"SHAP Beeswarm: {model_name} | {mod_name} | Class {target}"); 
            plt.tight_layout(); plt.savefig(paths['shap'] / f'shap_dot_{model_name}_class{target}.png', dpi=600); plt.close()
            plt.figure(figsize=(10, 8)); shap.summary_plot(Sc, Xc, feature_names=feats, show=False, max_display=20, plot_type="violin"); 
            # plt.title(f"SHAP Beeswarm: {model_name} | {mod_name} | Class {target}"); 
            plt.tight_layout(); plt.savefig(paths['shap'] / f'shap_violin_{model_name}_class{target}.png', dpi=600); plt.close()

    return compute_comprehensive_metrics(yt_all, yp_all, pb_all, classes_list), yt_all, pb_all
    

# ==== next cell ====

import numpy as np
import pandas as pd

EDA_GROUPS = [
    ("Raw statistics", [
        "EDA_Raw_Mean", "EDA_Raw_Median", "EDA_Raw_Min", "EDA_Raw_Max", "EDA_Raw_SD", "EDA_Raw_Range", "EDA_Raw_IQR"
    ]),
    ("Tonic component", [
        "EDA_Tonic_Mean", "EDA_Tonic_Median", "EDA_Tonic_Min", "EDA_Tonic_Max", "EDA_Tonic_SD", "EDA_Tonic_Range", "EDA_Tonic_IQR", 
        "EDA_Tonic_Slope"
    ]),
    ("Phasic component", [
        "EDA_Phasic_Mean", "EDA_Phasic_Median", "EDA_Phasic_Min", "EDA_Phasic_Max", "EDA_Phasic_SD", "EDA_Phasic_Range", "EDA_Phasic_IQR", "EDA_Phasic_RMS", 
        "EDA_Phasic_AUC_abs"
    ]),
    ("SCR peaks", [
        "EDA_SCR_Peaks_Amp_Mean", "EDA_SCR_Peaks_Amp_Max", "EDA_SCR_Peaks_Amp_SD", 
        "EDA_SCR_Peaks_N",
        "EDA_SCR_Peaks_Rate"
    ]),
]

ECGPPG_BASE_GROUPS = [
    ("Basic", ["HR", "RR", "SI"]),
    ("Time-domain", [
        "MeanNN", "MedianNN", "MinNN", "MaxNN",
        "SDNN", 
        "RMSSD", "SDSD", 
        "SDRMSSD",
        "pNN20", "pNN50", 
        "CVNN", "CVSD",
        "HTI", "TINN",
        "IQRNN", "MadNN", "MCVNN", 
        "Prc20NN", "Prc80NN"
    ]),
    ("Entropy", ["ApEn", "FuzzyEn", "ShanEn"]),
    ("Fractal Dimension", ["CD", "KFD", "LZC"]),
    ("Poincare geometry", ["SD1", "SD2", "SD1SD2", "S"]), 
    ("Poincare-derived autonomic indices", ["CSI", "CVI", "CSI_Modified"]),
    ("Heart rate fragmentation", ["PIP", "PAS", "PSS", "IALS"]),
    ("Heart rate asymmetry", [
        "AI", "PI", "GI", "SI",
        "SD1a", "SD1d", "SD2a", "SD2d",
        "C1a", "C1d", "C2a", "C2d", "SDNNa", "SDNNd", "Ca", "Cd",
    ]),
]

PTT_GROUPS = [
    ("Statistical", ["PTT_Mean", "PTT_Median", "PTT_Min", "PTT_Max", "PTT_SD", "PTT_Range"]),
    # ("Percentile", ["PTT_Prc05", "PTT_Prc10", "PTT_Prc25", "PTT_Prc75", "PTT_Prc90", "PTT_Prc95"]),
    ("Percentile", ["PTT_Prc20", "PTT_Prc80"]),

    ("Robust", ["PTT_IQR", "PTT_MAD"]),
    ("Shape", ["PTT_Skewness", "PTT_Kurtosis"]),
    ("Variability", ["PTT_CV"]),
    ("Count", ["PTT_N"]),
]

def build_feature_order_and_group():
    order_map = {}
    group_map = {}
    idx = 0

    def add_feature(name, group_name, idx_value):
        order_map[name] = idx_value
        group_map[name] = group_name

    # EDA
    for group_name, feats in EDA_GROUPS:
        for f in feats:
            add_feature(f, group_name, idx)
            idx += 1

    # ECG / PPG
    for modality in ["ECG", "PPG"]:
        for group_name, base_feats in ECGPPG_BASE_GROUPS:
            if group_name == 'Basic':
                for base in base_feats:
                    names = []

                    names.append(f"{modality}_{base}")

                    for name in names:
                        add_feature(name, group_name, idx)

                    idx += 1
            else:
                for base in base_feats:
                    names = []

                    names.append(f"{modality}_HRV_{base}")

                    for name in names:
                        add_feature(name, group_name, idx)

                    idx += 1

    # PTT
    for group_name, feats in PTT_GROUPS:
        for f in feats:
            add_feature(f, group_name, idx)
            idx += 1

    return order_map, group_map

def shorten_feature_label(feat):
    for prefix in ["ECG_", "PPG_", "EDA_", "PTT_"]:
        if feat.startswith(prefix):
            return feat[len(prefix):]
    return feat

# ==== next cell ====

import pandas as pd

# ------------------------------------------------------------
# 1) base result
# ------------------------------------------------------------
res_df = pd.read_csv(
    str(REPO_ROOT / "notebooks" / "lmm_results" / "ptt_05_70_3" / "ptt_05_70_3_strict_sorted.csv")
)

# res_df = pd.read_csv(
#     str(REPO_ROOT / "notebooks" / "lmm_sensitivity_results" / "EXP031_centered_session_mean_nolog" / "EXP031_centered_session_mean_nolog_full_results.csv")
# )

# ------------------------------------------------------------
# 2) SHAP importances
# ------------------------------------------------------------
xgb_shap = pd.read_csv(
    str(EXPORT_ROOT / "3class" / "ALL" / "csvs" / "shap_importance_XGB_all_classes.csv")
)[["feature", "mean_abs_shap"]].rename(columns={"mean_abs_shap": "mean_abs_shap_xgb"})

rf_shap = pd.read_csv(
    str(EXPORT_ROOT / "3class" / "ALL" / "csvs" / "shap_importance_RF_all_classes.csv")
)[["feature", "mean_abs_shap"]].rename(columns={"mean_abs_shap": "mean_abs_shap_rf"})

lr_shap = pd.read_csv(
    str(EXPORT_ROOT / "3class" / "ALL" / "csvs" / "shap_importance_LR_all_classes.csv")
)[["feature", "mean_abs_shap"]].rename(columns={"mean_abs_shap": "mean_abs_shap_lr"})

# ------------------------------------------------------------
# 3) merge
# ------------------------------------------------------------
res_df = res_df.merge(xgb_shap, on="feature", how="left")
res_df = res_df.merge(rf_shap, on="feature", how="left")
res_df = res_df.merge(lr_shap, on="feature", how="left")

# 혹시 NaN 있으면 0으로
for c in ["mean_abs_shap_xgb", "mean_abs_shap_rf", "mean_abs_shap_lr"]:
    res_df[c] = pd.to_numeric(res_df[c], errors="coerce").fillna(0.0)

# ==== next cell ====

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from pycirclize import Circos
import warnings
from pycirclize import config
config.R_PLOT_MARGIN = 0


warnings.simplefilter(action='ignore', category=FutureWarning)


def darken_color(color, factor=0.8):
    rgb = np.array(to_rgb(color))
    return tuple(np.clip(rgb * factor, 0, 1))


def plot_flexible_circos(
    res_df,
    save_path,
    track_configs,
    shap_df=None,
    equal_sector_width=False,
    title="Flexible Circos Profile",
    center_text=None,
    p_sig_col="p_max_used",
    p_sig_threshold=0.05,
    candidate_threshold=20,
):
    """
    track_configs example:
    [
        {"col": "mean_abs_shap_xgb", "kind": "shap", "label": "SHAP (XGB)", "transform": "none"},
        {"col": "mean_abs_shap_rf",  "kind": "shap", "label": "SHAP (RF)",  "transform": "none"},
        {"col": "mean_abs_shap_lr",  "kind": "shap", "label": "SHAP (LR)",  "transform": "none"},
        {"col": "p_max_used",        "kind": "p",    "label": "p_max",      "transform": "none"},
    ]
    """

    order_map, group_map = build_feature_order_and_group()

    df = res_df.copy()
    df["modality"] = pd.Categorical(
        df["feature"].astype(str).str.split("_").str[0],
        categories=["ECG", "PPG", "EDA", "PTT"],
        ordered=True
    )

    df = df[df["modality"].notna()].reset_index(drop=True)

    # merge shap if provided
    if shap_df is not None:
        shap_df = shap_df.copy()
        shap_df["mean_abs_shap"] = pd.to_numeric(shap_df["mean_abs_shap"], errors="coerce")
        df = df.merge(shap_df[["feature", "mean_abs_shap"]], on="feature", how="left")

    # ordering / labels
    df["plot_order"] = df["feature"].map(order_map).fillna(999999)
    df["group_name"] = df["feature"].map(group_map).fillna("Other")
    df["label_short"] = df["feature"].apply(shorten_feature_label)
    df = df.sort_values(by=["modality", "plot_order", "feature"]).reset_index(drop=True)

    mod_colors = {
        "ECG": "#d62728",
        "PPG": "#1f77b4",
        "EDA": "#2ca02c",
        "PTT": "#ff7f0e",
    }

    group_sizes = df.groupby("modality", observed=False).size().to_dict()
    sectors = {k: v for k, v in group_sizes.items() if v > 0}

    if equal_sector_width:
        max_n = max(sectors.values())
        sectors = {k: max_n for k in sectors.keys()}

    circos = Circos(sectors, space=10)

    # ---------- preprocess track values ----------
    processed = []
    feature_summary_rows = []
    for cfg in track_configs:
        col = cfg["col"]
        kind = cfg.get("kind", "z")
        transform = cfg.get("transform", None)
        label = cfg.get("label", col)

        vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy()

        if kind == "z":
            plot_vals = vals.copy()
            threshold = cfg.get("threshold", 1.96)
            vmax = cfg.get("vmax", max(4.0, float(np.ceil(np.nanmax(np.abs(plot_vals)) * 2) / 2)))
            vmin = -vmax
            tick_vals = [vmin, -threshold, threshold, vmax]
            tick_labels = [f"{vmin:g}", f"{-threshold:g}", f"{threshold:g}", f"{vmax:g}"]
            q_top = q70 = q80 = q90 = None
            scale_min = scale_max = None

        elif kind == "p":
            if transform is None:
                transform = "neglog10"

            vals = np.clip(vals, 1e-300, 1.0)

            if transform == "neglog10":
                plot_vals = -np.log10(vals)
                alpha = cfg.get("alpha", 0.05)
                threshold = -np.log10(alpha)
                vmax = cfg.get("vmax", max(2.0, float(np.ceil(np.nanmax(plot_vals) * 2) / 2)))
                vmin = 0.0
                tick_vals = [0.0, threshold, vmax]
                tick_labels = ["0", f"{threshold:.2f}", f"{vmax:g}"]
            else:
                plot_vals = vals.copy()
                alpha = cfg.get("alpha", 0.05)
                threshold = alpha
                vmax = cfg.get("vmax", 1.0)
                vmin = 0.0
                tick_vals = [0.0, alpha, vmax]
                tick_labels = ["0", f"{alpha:g}", f"{vmax:g}"]

            q_top = q70 = q80 = q90 = None
            scale_min = scale_max = None

        elif kind == "shap":
            vals = np.clip(vals, 0.0, None)

            if transform == "sqrt":
                vals_t = np.sqrt(vals)
            elif transform == "log1p":
                vals_t = np.log1p(vals)
            elif transform == "rank":
                vals_t = pd.Series(vals).rank(pct=True, method="average").to_numpy()
            else:
                vals_t = vals.copy()

            scale_min = float(np.nanmin(vals_t)) if np.isfinite(np.nanmin(vals_t)) else 0.0
            scale_max = float(np.nanmax(vals_t)) if np.isfinite(np.nanmax(vals_t)) else 1.0
            print(f"[preprocess] {col}: scale_min={scale_min:.6f}, scale_max={scale_max:.6f}")

            if scale_max > scale_min:
                plot_vals = (vals_t - scale_min) / (scale_max - scale_min)
            else:
                plot_vals = np.zeros_like(vals_t, dtype=float)

            print(f"[{col}] scaled min={plot_vals.min():.6f}, scaled max={plot_vals.max():.6f}")
            print("argmax feature:", df.loc[np.nanargmax(plot_vals), "feature"])

            vmin = 0.0
            vmax = 1.0
            tick_vals = [0.0, 0.5, 1.0]
            tick_labels = ["0", "0.5", "1.0"]
            threshold = None

            if candidate_threshold > 1:
                top_percentile = (152 - candidate_threshold) / 152 * 100
                q_top = np.nanpercentile(plot_vals, top_percentile)
            else:
                q_top = None

            q70 = np.nanpercentile(plot_vals, 70)
            q80 = np.nanpercentile(plot_vals, 80)
            q90 = np.nanpercentile(plot_vals, 90)

        else:
            raise ValueError(f"Unsupported kind: {kind}")

        processed.append({
            "col": col,
            "kind": kind,
            "label": label,
            "values": plot_vals,
            "vmin": vmin,
            "vmax": vmax,
            "tick_vals": tick_vals,
            "tick_labels": tick_labels,
            "threshold": threshold,
            "transform": transform,
            "q70": q70,
            "q80": q80,
            "q90": q90,
            "q_top": q_top if candidate_threshold > 1 else 0,
            "scale_min": scale_min,
            "scale_max": scale_max,
        })

    # ---------- layout ----------
    n_tracks = len(processed)
    outer_r, inner_r = 112, 45
    track_width = (outer_r - inner_r) / n_tracks

    for s_idx, sector in enumerate(circos.sectors):
        mod = sector.name
        mod_df = df[df["modality"] == mod].reset_index(drop=True)
        n_feat = len(mod_df)
        if n_feat == 0:
            continue

        sector.text(mod, size=15, r=inner_r - 8, color=mod_colors[mod], weight="bold")

        x = np.linspace(sector.start + 0.5, sector.end - 0.5, n_feat)
        labels = mod_df["label_short"].astype(str).tolist()
        group_names = mod_df["group_name"].tolist()

        # ---------------------------------------------------
        # label condition masks
        # red   : 모든 조건 만족
        # bold  : 2개 이상 만족
        # ---------------------------------------------------
        pvals_for_label = pd.to_numeric(mod_df[p_sig_col], errors="coerce").fillna(1.0).to_numpy()
        p_sig_mask = pvals_for_label <= p_sig_threshold

        shap_sig_masks = []
        for meta in processed:
            if meta["kind"] == "shap":
                raw_vals = pd.to_numeric(mod_df[meta["col"]], errors="coerce").fillna(0.0).to_numpy()
                raw_vals = np.clip(raw_vals, 0.0, None)

                if meta["transform"] == "sqrt":
                    shap_tmp = np.sqrt(raw_vals)
                elif meta["transform"] == "log1p":
                    shap_tmp = np.log1p(raw_vals)
                elif meta["transform"] == "rank":
                    shap_tmp = pd.Series(raw_vals).rank(pct=True, method="average").to_numpy()
                else:
                    shap_tmp = raw_vals.copy()

                if meta["scale_max"] is not None and meta["scale_max"] > meta["scale_min"]:
                    shap_scaled = (shap_tmp - meta["scale_min"]) / (meta["scale_max"] - meta["scale_min"])
                else:
                    shap_scaled = np.zeros_like(shap_tmp, dtype=float)

                if candidate_threshold == 0.3:
                    shap_sig_masks.append(shap_scaled >= meta["q70"])
                elif candidate_threshold == 0.2:
                    shap_sig_masks.append(shap_scaled >= meta["q80"])
                elif candidate_threshold == 0.1:
                    shap_sig_masks.append(shap_scaled >= meta["q90"])
                elif candidate_threshold > 1:
                    shap_sig_masks.append(shap_scaled >= meta["q_top"])

        cond_list = [p_sig_mask] + shap_sig_masks

        if len(cond_list) > 0:
            cond_sum = np.sum(np.vstack(cond_list), axis=0)
            bold_required = min(2, len(cond_list))
            label_bold = cond_sum >= bold_required
            label_red = cond_sum == len(cond_list)
        else:
            label_bold = np.zeros(len(mod_df), dtype=bool)
            label_red = np.zeros(len(mod_df), dtype=bool)

        # ---------------------------------------------------
        # save summary rows for this modality
        # ---------------------------------------------------
        summary_df = mod_df[["feature", "label_short", "modality"]].copy()
        summary_df["p_sig"] = p_sig_mask.astype(int)

        for j, shap_mask in enumerate(shap_sig_masks, start=1):
            summary_df[f"shap_sig_{j}"] = shap_mask.astype(int)

        summary_df["n_conditions_met"] = cond_sum if len(cond_list) > 0 else 0
        summary_df["all_3_met"] = label_red.astype(int)          # 3개 다 만족
        summary_df["at_least_2_met"] = label_bold.astype(int)   # 2개 이상 만족

        feature_summary_rows.append(summary_df)

        for idx, meta in enumerate(processed):
            t_outer = outer_r - (idx * track_width)
            t_inner = t_outer - track_width
            track = sector.add_track((t_inner, t_outer), r_pad_ratio=0.08)
            track.axis()

            y_vals = pd.to_numeric(mod_df[meta["col"]], errors="coerce").fillna(0.0).to_numpy(dtype=float)

            # apply transform
            if meta["kind"] == "z":
                y_plot = y_vals

            elif meta["kind"] == "p":
                y_vals = np.clip(y_vals, 1e-300, 1.0)
                if meta["transform"] == "neglog10":
                    y_plot = -np.log10(y_vals)
                else:
                    y_plot = y_vals

            elif meta["kind"] == "shap":
                y_vals = np.clip(y_vals, 0.0, None)

                if meta["transform"] == "sqrt":
                    y_vals_t = np.sqrt(y_vals)
                elif meta["transform"] == "log1p":
                    y_vals_t = np.log1p(y_vals)
                elif meta["transform"] == "rank":
                    y_vals_t = pd.Series(y_vals).rank(pct=True, method="average").to_numpy()
                else:
                    y_vals_t = y_vals

                if meta["scale_max"] is not None and meta["scale_max"] > meta["scale_min"]:
                    y_plot = (y_vals_t - meta["scale_min"]) / (meta["scale_max"] - meta["scale_min"])
                else:
                    y_plot = np.zeros_like(y_vals_t, dtype=float)
                    print("fail")

            else:
                y_plot = y_vals

            # bar 강조 스타일
            for xi, yi, raw_y, feat_red in zip(x, y_plot, y_vals, label_red):
                base_color = mod_colors[mod]
                face_color = base_color
                alpha = 0.45
                ec = "none"
                lw = 0.0

                if meta["kind"] == "p":
                    is_sig = raw_y <= p_sig_threshold
                    if is_sig:
                        face_color = darken_color(base_color, 0.55)
                        alpha = 1.0
                        ec = "black"
                    else:
                        face_color = base_color
                        alpha = 0.50

                elif meta["kind"] == "shap":
                    if yi >= meta["q90"] and (candidate_threshold == 0.1 or candidate_threshold == 0.2 or candidate_threshold == 0.3):
                        face_color = darken_color(base_color, 0.55)
                        alpha = 1.0
                        ec = "black"
                    elif yi >= meta["q80"] and (candidate_threshold == 0.2 or candidate_threshold == 0.3):
                        face_color = darken_color(base_color, 0.70)
                        alpha = 0.95
                        ec = "black"
                    elif yi >= meta["q70"] and candidate_threshold == 0.3:
                        face_color = darken_color(base_color, 0.82)
                        alpha = 0.90
                        ec = "none"
                    elif yi >= meta["q_top"] and candidate_threshold > 1:
                        face_color = darken_color(base_color, 0.55)
                        alpha = 1.0
                        ec = "black"
                    else:
                        face_color = base_color
                        alpha = 0.50

                else:
                    if feat_red:
                        face_color = darken_color(base_color, 0.75)
                        alpha = 0.95
                        ec = "black"
                    else:
                        face_color = base_color
                        alpha = 0.5

                track.bar(
                    [xi], [yi],
                    vmin=meta["vmin"], vmax=meta["vmax"],
                    color=face_color,
                    alpha=alpha,
                    ec=ec,
                    lw=lw,
                    width=0.82
                )

            # baseline / thresholds
            if meta["kind"] == "z":
                track.line(
                    x, np.zeros_like(x),
                    vmin=meta["vmin"], vmax=meta["vmax"],
                    color="black", lw=0.7
                )
                track.line(
                    x, np.full_like(x, meta["threshold"]),
                    vmin=meta["vmin"], vmax=meta["vmax"],
                    color="grey", lw=0.5, ls="--"
                )
                track.line(
                    x, np.full_like(x, -meta["threshold"]),
                    vmin=meta["vmin"], vmax=meta["vmax"],
                    color="grey", lw=0.5, ls="--"
                )

            elif meta["kind"] == "p":
                track.line(
                    x, np.zeros_like(x),
                    vmin=meta["vmin"], vmax=meta["vmax"],
                    color="black", lw=0.7
                )
                if meta["threshold"] is not None:
                    track.line(
                        x, np.full_like(x, meta["threshold"]),
                        vmin=meta["vmin"], vmax=meta["vmax"],
                        color="grey", lw=0.6, ls="--"
                    )

            elif meta["kind"] == "shap":
                track.line(
                    x, np.zeros_like(x),
                    vmin=meta["vmin"], vmax=meta["vmax"],
                    color="black", lw=0.7
                )
                temp = (
                    meta["q70"] if candidate_threshold == 0.3
                    else meta["q80"] if candidate_threshold == 0.2
                    else meta["q90"] if candidate_threshold == 0.1
                    else meta["q_top"]
                )
                for q, c, ls, lw_ in [
                    (temp, "grey", "--", 0.60),
                ]:
                    track.line(
                        x, np.full_like(x, q),
                        vmin=meta["vmin"], vmax=meta["vmax"],
                        color=c, lw=lw_, ls=ls
                    )

            # subgroup separators
            for i in range(1, len(group_names)):
                if group_names[i] != group_names[i - 1]:
                    x_sep = (x[i - 1] + x[i]) / 2.0
                    y0 = meta["vmin"]
                    y1 = meta["vmax"]
                    track.line(
                        [x_sep, x_sep], [y0, y1],
                        vmin=meta["vmin"], vmax=meta["vmax"],
                        color="lightgrey", lw=1.2, ls=":"
                    )

            # y axis ticks
            track.yticks(
                meta["tick_vals"],
                labels=meta["tick_labels"],
                vmin=meta["vmin"],
                vmax=meta["vmax"],
                label_size=4,
                tick_length=1
            )

            # outermost: feature labels
            if idx == 0:
                for x_pos, label, is_bold, is_red in zip(x, labels, label_bold, label_red):
                    track.xticks(
                        [x_pos], [label],
                        tick_length=1,
                        show_bottom_line=False,
                        label_orientation="vertical",
                        label_size=6.8 if is_bold else 6,
                        line_kws=dict(ec="grey", lw=0.4),
                        text_kws=dict(
                            weight="bold" if is_bold else "normal",
                            color="red" if is_red else "black"
                        )
                    )

            if s_idx == 0:
                
                track.text(
                    meta["label"],
                    x=sector.start + 0.5,
                    r=(t_inner + t_outer) / 2 + 3.5*idx-1,
                    size=10,
                    ha="left",
                    ignore_range_error=True
                )

    fig = circos.plotfig()
    ax = fig.axes[0]
    ax.set_rlim(0, outer_r)

    if center_text is not None:
        fig.text(0.5, 0.5, center_text, ha="center", va="center", fontsize=6)

    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] Flexible circos saved to {save_path}")

    # ---------------------------------------------------
    # save feature lists
    # ---------------------------------------------------
    if len(feature_summary_rows) > 0:
        feature_summary_df = pd.concat(feature_summary_rows, axis=0, ignore_index=True)
        feature_summary_df = feature_summary_df.sort_values(
            by=["modality", "n_conditions_met", "feature"],
            ascending=[True, False, True]
        ).reset_index(drop=True)

        all3_df = feature_summary_df[feature_summary_df["all_3_met"] == 1].copy()
        ge2_df = feature_summary_df[feature_summary_df["at_least_2_met"] == 1].copy()

        save_dir = Path(save_path).parent
        stem = Path(save_path).stem

        summary_csv = save_dir / f"{stem}_feature_condition_summary.csv"
        all3_csv = save_dir / f"{stem}_features_all_3_met.csv"
        ge2_csv = save_dir / f"{stem}_features_at_least_2_met.csv"

        feature_summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
        all3_df.to_csv(all3_csv, index=False, encoding="utf-8-sig")
        ge2_df.to_csv(ge2_csv, index=False, encoding="utf-8-sig")

        print(f"[Saved] summary      : {summary_csv}")
        print(f"[Saved] all 3 met    : {all3_csv}")
        print(f"[Saved] >=2 met      : {ge2_csv}")

    tmp = df[["feature", "modality", "mean_abs_shap_xgb"]].copy()
    vals = pd.to_numeric(tmp["mean_abs_shap_xgb"], errors="coerce").fillna(0.0).to_numpy()
    scale_min = float(np.nanmin(vals))
    scale_max = float(np.nanmax(vals))
    scaled = (vals - scale_min) / (scale_max - scale_min)
    tmp["scaled_xgb"] = scaled
    print(tmp.loc[tmp["feature"] == "EDA_Raw_Min"])

    print(f"\n[{mod}]")
    for idx, meta in enumerate(processed):
        t_outer = outer_r - (idx * track_width)
        t_inner = t_outer - track_width
        print(
            idx, meta["label"],
            "t_inner=", round(t_inner, 3),
            "t_outer=", round(t_outer, 3),
            "width=", round(t_outer - t_inner, 3)
        )

plot_flexible_circos(
    res_df=res_df,
    save_path=EXPORT_ROOT / "circos_shap_allmodels_lmm_highlighted_30.png",
    track_configs=[
        {"col": "mean_abs_shap_xgb", "kind": "shap", "label": "SHAP (XGB)", "transform": "none"},
        {"col": "mean_abs_shap_rf",  "kind": "shap", "label": "SHAP (RF)",  "transform": "none"},
        {"col": "p_max_used",        "kind": "p",    "label": r"$\tilde{p}$ (LMM)"},
    ],
    center_text=(
        # "Inner track: $-\\log_{10}(p_{max})$.\n\n"
        # "Outer tracks: rescaled SHAP values\n(0-1, model-wise min-max scaling).\n\n"
        # "Red bold labels:features satisfying\n all criteria simultaneously.\n\n"
        # "Bars are darker for more salient features."
        " "
    ),
    p_sig_col="p_max_used",
    p_sig_threshold=0.05,
    candidate_threshold = 0.3,
)
