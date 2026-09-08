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

selected_feature_csv = str(EXPORT_ROOT / "circos_shap_allmodels_lmm_highlighted_30_features_at_least_2_met.csv")
table7_results = []

if not Path(selected_feature_csv).exists():
    print(f"Skipping Table VII: {selected_feature_csv} not found. "
          f"Run the 'Combined Circos Plot' cells above first, then re-run this cell.")
else:
    db36, af36 = load_and_label('10s', n_classes=3, filter_abnormal=True, prefix="_05_70_3",
                                 compare='z_vs_lh', selected_feature_csv=selected_feature_csv)
    if db36 is not None:
        seed_everything(SEED)
        paths36 = get_save_paths(3, 'ALL_36feat')
        cur_curves = {}
        for m_name in ['KNN', 'SVM', 'LR', 'MLP', 'XGB', 'RF']:
            try:
                m_res, yt, pb = run_full_experiment(db36, af36, m_name, paths36, "3class", "ALL_36feat")
                m_res.update({'n_classes': 3, 'duration': '10s', 'modality': 'ALL_36feat',
                              'model': m_name, 'samples': len(db36)})
                table7_results.append(m_res)
                cur_curves[m_name] = (yt, pb)
            except Exception as e:
                print(f"  [Error {m_name}] {e}")
        if cur_curves:
            save_compare_figure(cur_curves, np.unique(db36['y']), paths36['plots'],
                                 "Models Comparisons | 3c | 10s | ALL_36feat")
        pd.DataFrame(table7_results).to_csv(EXPORT_ROOT / 'results_metrics_36feat.csv', index=False)
        print("\n✅ Table VII (36 candidate features) complete.")
