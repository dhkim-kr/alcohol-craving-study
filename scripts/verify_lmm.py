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

from pathlib import Path
import pandas as pd
import numpy as np
from pymer4.models import Lmer
from scipy.stats import shapiro
from statsmodels.stats.multitest import multipletests
import warnings
import json



def map_y_to_group(y):
    s = str(y).strip().lower()
    mapping = {
        "0": "0", "zero": "0",
        "1": "Low", "low": "Low",
        "2": "High", "high": "High",
    }
    return mapping.get(s, np.nan)



# ============================================================
# 2) helper
# ============================================================
def make_session_summary(df_input, feats, agg="median", center_mode=None):
    group_cols = ["Subject", "Session", "Group"]

    if agg == "median":
        sdf = df_input.groupby(group_cols)[feats].median().reset_index()
    elif agg == "mean":
        sdf = df_input.groupby(group_cols)[feats].mean().reset_index()
    else:
        raise ValueError("agg must be 'median' or 'mean'")

    # session-level centering after aggregation
    if center_mode is not None:
        for feat in feats:
            if center_mode == "session_mean":
                center_val = sdf.groupby(["Subject", "Session"])[feat].transform("mean")
                sdf[feat] = sdf[feat] - center_val

            elif center_mode == "session_median":
                center_val = sdf.groupby(["Subject", "Session"])[feat].transform("median")
                sdf[feat] = sdf[feat] - center_val

            elif center_mode == "group0":
                base = (
                    sdf[sdf["Group"] == "0"][["Subject", "Session", feat]]
                    .rename(columns={feat: f"{feat}_baseline0"})
                )
                sdf = sdf.merge(base, on=["Subject", "Session"], how="left")
                sdf[feat] = sdf[feat] - sdf[f"{feat}_baseline0"]
                sdf = sdf.drop(columns=[f"{feat}_baseline0"])

            else:
                raise ValueError("center_mode must be one of: None, session_mean, session_median, group0")

    return sdf

def keep_complete_sessions(df_feat):
    grp_counts = df_feat.groupby(["Subject", "Session"])["Group"].nunique()
    keep_idx = grp_counts[grp_counts == 3].index
    if len(keep_idx) == 0:
        return df_feat.iloc[0:0].copy()
    return (
        df_feat.set_index(["Subject", "Session"])
        .loc[keep_idx]
        .reset_index()
    )

def extract_variance_components(model):
    if not hasattr(model, "ranef_var"):
        return np.nan, np.nan, np.nan

    rv = model.ranef_var.copy()
    num_cols = rv.select_dtypes(include=[np.number]).columns
    if len(num_cols) == 0:
        return np.nan, np.nan, np.nan

    col = num_cols[0]
    idx = [str(x).lower() for x in rv.index]

    subj_var = np.nan
    subj_sess_var = np.nan
    resid_var = np.nan

    for i, name in enumerate(idx):
        if "subject:session" in name or "session:subject" in name:
            subj_sess_var = float(rv.iloc[i][col])
        elif "subject" in name and "session" not in name:
            subj_var = float(rv.iloc[i][col])
        elif "resid" in name or "residual" in name:
            resid_var = float(rv.iloc[i][col])

    if np.isnan(subj_var) and rv.shape[0] >= 1:
        subj_var = float(rv.iloc[0][col])
    if np.isnan(resid_var) and rv.shape[0] >= 2:
        resid_var = float(rv.iloc[-1][col])

    return subj_var, subj_sess_var, resid_var

def fit_lmer_with_fallback(formula, data, factor_spec):
    model = Lmer(formula, data=data)

    try:
        try:
            model.set_factors(factor_spec)
            model.fit(summary=False, conf_method="satterthwaite")
        except AttributeError:
            model.fit(
                factors=factor_spec,
                summary=False,
                conf_method="satterthwaite"
            )
    except Exception:
        model = Lmer(formula, data=data)
        try:
            try:
                model.set_factors(factor_spec)
                model.fit(
                    summary=False,
                    conf_method="satterthwaite",
                    control="bobyqa"
                )
            except AttributeError:
                model.fit(
                    factors=factor_spec,
                    summary=False,
                    conf_method="satterthwaite",
                    control="bobyqa"
                )
        except Exception as e:
            raise e

    return model

def normalize_contrast_name(x: str) -> str:
    s = str(x).strip()
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    s = " ".join(s.split())
    return s

def get_contrast_columns(contrasts: pd.DataFrame):
    p_col = None
    for candidate in ["P-val", "Pr(>|t|)", "p-value", "P-value"]:
        if candidate in contrasts.columns:
            p_col = candidate
            break
    if p_col is None:
        raise KeyError(f"p-value column not found: {contrasts.columns.tolist()}")

    contrast_col = None
    for candidate in ["Contrast", "contrast", "Comparison", "comparison"]:
        if candidate in contrasts.columns:
            contrast_col = candidate
            break
    if contrast_col is None:
        raise KeyError(f"contrast column not found: {contrasts.columns.tolist()}")

    return contrast_col, p_col

def extract_pairwise_pvalues(contrasts: pd.DataFrame):
    contrast_col, p_col = get_contrast_columns(contrasts)

    p_map = {}
    for _, row in contrasts.iterrows():
        cname = normalize_contrast_name(row[contrast_col])
        p_map[cname] = float(row[p_col])

    def get_p(a, b):
        a = normalize_contrast_name(a)
        b = normalize_contrast_name(b)
        if a in p_map:
            return p_map[a]
        if b in p_map:
            return p_map[b]
        return np.nan

    p_zero_vs_low  = get_p("0 - Low", "Low - 0")
    p_zero_vs_high = get_p("0 - High", "High - 0")
    p_low_vs_high  = get_p("Low - High", "High - Low")

    p_triplet = {
        "0 - Low": p_zero_vs_low,
        "0 - High": p_zero_vs_high,
        "Low - High": p_low_vs_high,
    }

    p_max = np.nanmax(list(p_triplet.values()))
    p_max_contrast = max(p_triplet, key=p_triplet.get)

    return {
        "p_zero_vs_low": p_zero_vs_low,
        "p_zero_vs_high": p_zero_vs_high,
        "p_low_vs_high": p_low_vs_high,
        "p_max": p_max,
        "p_max_contrast": p_max_contrast,
    }

def build_formula(use_session_fixed: bool, random_structure: str):
    fixed_terms = ["Group"]
    if use_session_fixed:
        fixed_terms.append("Session")

    fixed_part = " + ".join(fixed_terms)

    if random_structure == "subject_only":
        random_part = "(1|Subject)"
    elif random_structure == "subject_session":
        random_part = "(1|Subject) + (1|Subject:Session)"
    else:
        raise ValueError("random_structure must be 'subject_only' or 'subject_session'")

    return f"Target ~ {fixed_part} + {random_part}"

def get_stat_note(cfg):
    notes = []
    if not cfg["use_feature_fdr"]:
        notes.append("exploratory_no_feature_fdr")
    if cfg["pairwise_adjust"] == "none":
        notes.append("no_within_feature_pairwise_adjust")
    if not cfg["use_session_fixed"]:
        notes.append("session_fixed_removed")
    if cfg["random_structure"] == "subject_only":
        notes.append("simpler_random_effect")
    if cfg["random_structure"] == "subject_session":
        notes.append("check_near_singular_for_summary_data")
    return "|".join(notes) if notes else "current_like"

# ============================================================
# 3) 단일 실험 실행
# ============================================================
def run_one_experiment(df_raw, feats, cfg):
    sdf = make_session_summary(
        df_raw,
        feats,
        agg=cfg["agg"],
        center_mode=cfg.get("center_mode", None),
    )
    results = []

    for feat in feats:
        d = sdf[["Subject", "Session", "Group", feat]].dropna().copy()

        if cfg["complete_session_filter"]:
            d = keep_complete_sessions(d)

        d["Target"] = pd.to_numeric(d[feat], errors="coerce")
        d = d.dropna(subset=["Target"]).reset_index(drop=True)

        if len(d) == 0 or d["Group"].nunique() < 2:
            continue
        if d["Subject"].nunique() < 2:
            continue

        factor_spec = {
            "Group": ["0", "Low", "High"],
            "Session": sorted(d["Session"].astype(str).unique().tolist())
        }

        formula = build_formula(
            use_session_fixed=cfg["use_session_fixed"],
            random_structure=cfg["random_structure"]
        )

        # initial fit
        model = fit_lmer_with_fallback(formula, d, factor_spec)
        residuals = np.asarray(model.residuals).ravel()
        _, p_shap = shapiro(residuals)
        p_shap_post = np.nan
        log_transformed = False

        if cfg["use_log_transform"] and (p_shap < 0.05):
            min_val = d["Target"].min()
            if min_val <= 0:
                shift = -min_val + 1e-9
                d["Target"] = np.log1p(d["Target"] + shift)
            else:
                d["Target"] = np.log(d["Target"])
            log_transformed = True

            model = fit_lmer_with_fallback(formula, d, factor_spec)
            residuals = np.asarray(model.residuals).ravel()
            _, p_shap_post = shapiro(residuals)

        # residual variance ratio
        vars_list = []
        for g in ["0", "Low", "High"]:
            rv = residuals[d["Group"] == g]
            if len(rv) > 1:
                vars_list.append(np.var(rv, ddof=1))
            else:
                vars_list.append(np.nan)
        valid_vars = [v for v in vars_list if pd.notna(v) and v > 0]
        var_ratio = max(valid_vars) / min(valid_vars) if len(valid_vars) >= 2 else np.nan

        # random effects
        subj_var, subj_sess_var, resid_var = extract_variance_components(model)
        total_var = np.nansum([subj_var, subj_sess_var, resid_var])

        subj_icc = subj_var / total_var if pd.notna(subj_var) and total_var > 0 else np.nan
        sess_icc = subj_sess_var / total_var if pd.notna(subj_sess_var) and total_var > 0 else np.nan

        near_singular_subject = bool(pd.notna(subj_var) and subj_var < 1e-6)
        near_singular_subject_session = bool(pd.notna(subj_sess_var) and subj_sess_var < 1e-6)

        # pairwise contrasts
        pairwise_adjust = cfg["pairwise_adjust"]
        if hasattr(model, "emmeans"):
            contrasts = model.emmeans("Group", contrasts="pairwise", p_adjust=pairwise_adjust)
        else:
            _, contrasts = model.post_hoc("Group", p_adjust=pairwise_adjust)

        pinfo = extract_pairwise_pvalues(contrasts)

        results.append({
            "Feature": feat,
            "Formula": formula,
            "Agg": cfg["agg"],
            "CenterMode": cfg.get("center_mode", None),
            "CompleteSessionFilter": cfg["complete_session_filter"],
            "UseSessionFixed": cfg["use_session_fixed"],
            "RandomStructure": cfg["random_structure"],
            "PairwiseAdjust": cfg["pairwise_adjust"],
            "UseFeatureFDR": cfg["use_feature_fdr"],
            "UseLogTransform": cfg["use_log_transform"],
            "LogTransformed": log_transformed,

            "Rows": len(d),
            "Subjects": d["Subject"].nunique(),
            "Sessions": d[["Subject", "Session"]].drop_duplicates().shape[0],

            "Subject_ICC": subj_icc,
            "Session_ICC": sess_icc,
            "NearSingular_Subject": near_singular_subject,
            "NearSingular_SubjectSession": near_singular_subject_session,
            "Shapiro_p_pre": p_shap,
            "Shapiro_p_post": p_shap_post,
            "ResidualVarRatio": var_ratio,

            "p_zero_vs_low": pinfo["p_zero_vs_low"],
            "p_zero_vs_high": pinfo["p_zero_vs_high"],
            "p_low_vs_high": pinfo["p_low_vs_high"],
            "p_max": pinfo["p_max"],
            "p_max_driver": pinfo["p_max_contrast"],
        })

    if len(results) == 0:
        return pd.DataFrame()

    res = pd.DataFrame(results)

    # feature-level FDR
    if cfg["use_feature_fdr"]:
        for col in ["p_zero_vs_low", "p_zero_vs_high", "p_low_vs_high", "p_max"]:
            pvals = res[col].to_numpy(dtype=float)
            mask = np.isfinite(pvals)
            adj = np.full_like(pvals, np.nan, dtype=float)
            if mask.any():
                adj[mask] = multipletests(pvals[mask], method="fdr_bh")[1]
            res[f"{col}_used"] = adj
    else:
        for col in ["p_zero_vs_low", "p_zero_vs_high", "p_low_vs_high", "p_max"]:
            res[f"{col}_used"] = res[col].astype(float)

    # significance flags based on "used" p-values
    res["sig_zero_vs_low"] = res["p_zero_vs_low_used"] < 0.05
    res["sig_zero_vs_high"] = res["p_zero_vs_high_used"] < 0.05
    res["sig_low_vs_high"] = res["p_low_vs_high_used"] < 0.05

    res["n_pairwise_sig"] = (
        res["sig_zero_vs_low"].astype(int)
        + res["sig_zero_vs_high"].astype(int)
        + res["sig_low_vs_high"].astype(int)
    )
    res["AnyPairwiseSig"] = res["n_pairwise_sig"] >= 1
    res["AllThreeDiffer"] = res["p_max_used"] < 0.05

    # 정렬: 유의 feature 상단
    res = res.sort_values(
        by=[
            "AllThreeDiffer",
            "AnyPairwiseSig",
            "n_pairwise_sig",
            "p_zero_vs_high_used",
            "p_zero_vs_low_used",
            "p_low_vs_high_used",
            "p_max_used",
        ],
        ascending=[False, False, False, True, True, True, True]
    ).reset_index(drop=True)

    return res

from itertools import product

# ============================================================
# 4) 실험 설정: 모든 조합 자동 생성
# ============================================================

# 지금은 median만 사용
# 나중에 mean도 같이 돌리고 싶으면 ["median", "mean"] 로 바꾸면 됨
AGG_OPTIONS = ["median", "mean"]

# complete-session filter는 일단 유지 권장
COMPLETE_SESSION_FILTER_OPTIONS = [True]

# fixed effect
USE_SESSION_FIXED_OPTIONS = [True, False]

# random effect
RANDOM_STRUCTURE_OPTIONS = [
    "subject_only",      # (1|Subject)
    "subject_session",   # (1|Subject) + (1|Subject:Session)
]

# pairwise 내부 보정
PAIRWISE_ADJUST_OPTIONS = [
    "tukey",
    "none",
]

# feature-level FDR
USE_FEATURE_FDR_OPTIONS = [True, False]

# log transform
USE_LOG_TRANSFORM_OPTIONS = [True, False]


def yn(flag: bool) -> str:
    return "yes" if flag else "no"


def build_experiment_name(
    idx,
    agg,
    complete_session_filter,
    use_session_fixed,
    random_structure,
    pairwise_adjust,
    use_feature_fdr,
    use_log_transform,
):
    random_label_map = {
        "subject_only": "rand-subject",
        "subject_session": "rand-subject+subjectsession",
    }

    return (
        f"EXP{idx:03d}"
        f"__agg-{agg}"
        f"__complete-{yn(complete_session_filter)}"
        f"__sessionfix-{yn(use_session_fixed)}"
        f"__{random_label_map[random_structure]}"
        f"__pairadj-{pairwise_adjust}"
        f"__fdr-{yn(use_feature_fdr)}"
        f"__log-{yn(use_log_transform)}"
    )

"""
EXPERIMENTS = []
exp_idx = 1

for (
    agg,
    complete_session_filter,
    use_session_fixed,
    random_structure,
    pairwise_adjust,
    use_feature_fdr,
    use_log_transform,
) in product(
    AGG_OPTIONS,
    COMPLETE_SESSION_FILTER_OPTIONS,
    USE_SESSION_FIXED_OPTIONS,
    RANDOM_STRUCTURE_OPTIONS,
    PAIRWISE_ADJUST_OPTIONS,
    USE_FEATURE_FDR_OPTIONS,
    USE_LOG_TRANSFORM_OPTIONS,
):
    cfg = {
        "name": build_experiment_name(
            exp_idx,
            agg,
            complete_session_filter,
            use_session_fixed,
            random_structure,
            pairwise_adjust,
            use_feature_fdr,
            use_log_transform,
        ),
        "agg": agg,
        "complete_session_filter": complete_session_filter,
        "use_session_fixed": use_session_fixed,
        "random_structure": random_structure,
        "pairwise_adjust": pairwise_adjust,
        "use_feature_fdr": use_feature_fdr,
        "use_log_transform": use_log_transform,
    }
    EXPERIMENTS.append(cfg)
    exp_idx += 1

print(f"Total experiments: {len(EXPERIMENTS)}")
for e in EXPERIMENTS[:5]:
    print(e["name"])

# ============================================================
# 5) 전체 실험 실행 + CSV 저장
# ============================================================
overview_rows = []

for cfg in EXPERIMENTS:
    exp_name = cfg["name"]
    print("\n" + "=" * 80)
    print(f"Running experiment: {exp_name}")
    print("=" * 80)

    exp_dir = OUT_DIR / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # config 저장
    with open(exp_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    res = run_one_experiment(df_raw, feats_3c, cfg)

    if res.empty:
        overview_rows.append({
            "experiment": exp_name,
            "n_features_tested": 0,
            "n_any_pairwise_sig": 0,
            "n_all_three_sig": 0,
            "n_sig_zero_vs_low": 0,
            "n_sig_zero_vs_high": 0,
            "n_sig_low_vs_high": 0,
            "n_near_singular_subject": 0,
            "n_near_singular_subject_session": 0,
            "stat_note": get_stat_note(cfg),
        })
        continue

    # 전체 저장
    res.to_csv(exp_dir / f"{exp_name}_full_results.csv", index=False)

    # 유의 feature만 저장
    sig_res = res[res["AnyPairwiseSig"] | res["AllThreeDiffer"]].copy()
    sig_res = sig_res.sort_values(
        by=[
            "AllThreeDiffer",
            "AnyPairwiseSig",
            "n_pairwise_sig",
            "p_zero_vs_high_used",
            "p_zero_vs_low_used",
            "p_low_vs_high_used",
            "p_max_used",
        ],
        ascending=[False, False, False, True, True, True, True]
    )
    sig_res.to_csv(exp_dir / f"{exp_name}_significant_first.csv", index=False)

    # strict top list
    strict_res = res.sort_values(
        by=["AllThreeDiffer", "p_max_used", "p_zero_vs_high_used"],
        ascending=[False, True, True]
    )
    strict_res.to_csv(exp_dir / f"{exp_name}_strict_sorted.csv", index=False)

    overview_rows.append({
        "experiment": exp_name,
        "n_features_tested": len(res),
        "n_any_pairwise_sig": int(res["AnyPairwiseSig"].sum()),
        "n_all_three_sig": int(res["AllThreeDiffer"].sum()),
        "n_sig_zero_vs_low": int(res["sig_zero_vs_low"].sum()),
        "n_sig_zero_vs_high": int(res["sig_zero_vs_high"].sum()),
        "n_sig_low_vs_high": int(res["sig_low_vs_high"].sum()),
        "n_near_singular_subject": int(res["NearSingular_Subject"].sum()),
        "n_near_singular_subject_session": int(res["NearSingular_SubjectSession"].sum()),
        "median_pmax_used": float(np.nanmedian(res["p_max_used"])),
        "best_feature_pmax_used": float(np.nanmin(res["p_max_used"])),
        "stat_note": get_stat_note(cfg),
    })

# overview 저장
overview = pd.DataFrame(overview_rows).sort_values(
    by=["n_all_three_sig", "n_any_pairwise_sig", "n_sig_zero_vs_high", "best_feature_pmax_used"],
    ascending=[False, False, False, True]
).reset_index(drop=True)

overview.to_csv(OUT_DIR / "experiment_overview.csv", index=False)

print("\nSaved to:", OUT_DIR.resolve())
print(overview.to_string(index=False))
"""

# ==== next cell ====

prefix = "_05_70_3"
db, af = load_and_label('10s', n_classes=3, filter_abnormal=True, prefix=prefix)

mff = get_feature_cols_by_modality(af, 'ALL')

# print(mff)
mod = 'ALL'

# ------------------------------------------------------
# 🔥 모달리티별 abnormal session 제거 (가장 중요한 부분)
# ------------------------------------------------------
if mod != 'ALL':
    if mod == 'PPG':
        bad = MISSING_PPG
    elif mod == 'EDA':
        bad = MISSING_GSR
    elif mod == 'ECG':
        bad = SAMPLING_RATE_ERROR
    elif mod == 'PTT':
        bad = list(set(MISSING_PPG) | set(SAMPLING_RATE_ERROR))

    db_mod = db[~db['session_id'].isin(bad)].reset_index(drop=True)
else:
    db_mod = db.copy()

print(f" - Samples: {len(db_mod)}")
# Preserve data for LMM analysis
df_3c = db_mod.copy()
feats_3c = list(mff)

# ============================================================
# 0) 사용자 설정
# ============================================================
OUT_DIR = Path("./lmm_results_verify")  # redirected: do not overwrite tracked notebooks/lmm_results/
OUT_DIR.mkdir(parents=True, exist_ok=True)

# df_3c, feats_3c가 이미 메모리에 있다고 가정
# df_3c: raw feature dataframe
# feats_3c: feature column list

# ============================================================
# 1) 데이터 준비
# ============================================================
df_raw = df_3c.reset_index(drop=True).copy()
df_raw["Subject"] = df_raw["subject"].astype(str)
df_raw["Session"] = df_raw["version"].astype(str)

df_raw["Group"] = df_raw["y"].apply(map_y_to_group)
df_raw = df_raw[df_raw["Group"].notna()].copy()
df_raw["Group"] = pd.Categorical(
    df_raw["Group"], categories=["0", "Low", "High"], ordered=True
)

experiment = {
    "name": f"ptt{prefix}",
    "agg": "median",
    "complete_session_filter": True,
    "use_session_fixed": False,
    "random_structure": "subject_session",
    "pairwise_adjust": "none",
    "use_feature_fdr": False,
    "use_log_transform": True,
    "center_mode": None,
}

exp_dir = OUT_DIR / experiment["name"]
exp_dir.mkdir(parents=True, exist_ok=True)

with open(exp_dir / "config.json", "w", encoding="utf-8") as f:
    json.dump(experiment, f, indent=2, ensure_ascii=False)

res = run_one_experiment(df_raw, feats_3c, experiment)

res.to_csv(
    exp_dir / f"{experiment['name']}_full_results.csv",
    index=False
)

sig_res = res[
    res["AnyPairwiseSig"] | res["AllThreeDiffer"]
].copy()

sig_res = sig_res.sort_values(
    by=[
        "AllThreeDiffer",
        "AnyPairwiseSig",
        "n_pairwise_sig",
        "p_zero_vs_high_used",
        "p_zero_vs_low_used",
        "p_low_vs_high_used",
        "p_max_used",
    ],
    ascending=[False, False, False, True, True, True, True]
)

sig_res.to_csv(
    exp_dir / f"{experiment['name']}_significant_first.csv",
    index=False
)

strict_res = res.sort_values(
    by=["AllThreeDiffer", "p_max_used", "p_zero_vs_high_used"],
    ascending=[False, True, True]
)

strict_res.to_csv(
    exp_dir / f"{experiment['name']}_strict_sorted.csv",
    index=False
)

print("Saved to:", exp_dir.resolve())
print(res.head())

# ==== next cell ====


# ==== verification: compare fresh LMM re-run against the tracked archive ====
import pandas as pd
orig = pd.read_csv(Path("./lmm_results") / "ptt_05_70_3" / "ptt_05_70_3_strict_sorted.csv").set_index("Feature").sort_index()
fresh = pd.read_csv(Path("./lmm_results_verify") / "ptt_05_70_3" / "ptt_05_70_3_strict_sorted.csv").set_index("Feature").sort_index()
common_cols = [c for c in ["p_zero_vs_low","p_zero_vs_high","p_low_vs_high","p_max","AllThreeDiffer"] if c in orig.columns and c in fresh.columns]
print("tracked rows:", len(orig), "| fresh rows:", len(fresh))
merged = orig[common_cols].join(fresh[common_cols], lsuffix="_orig", rsuffix="_fresh", how="inner")
print("matched features:", len(merged))
n_allthree_match = (merged["AllThreeDiffer_orig"] == merged["AllThreeDiffer_fresh"]).sum()
print(f"AllThreeDiffer flag matches: {n_allthree_match}/{len(merged)}")
import numpy as np
pmax_diff = (merged["p_max_orig"] - merged["p_max_fresh"]).abs()
print("max |p_max diff|:", pmax_diff.max(), "| mean |p_max diff|:", pmax_diff.mean())
print("Features where AllThreeDiffer differs:")
print(merged[merged["AllThreeDiffer_orig"] != merged["AllThreeDiffer_fresh"]])
merged.to_csv("lmm_verification_comparison.csv")
