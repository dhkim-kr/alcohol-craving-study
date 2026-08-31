# biosignal_dataset.py (formerly src/util/data.py)

import pandas as pd
import os
import numpy as np
import joblib
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from numpy.fft import rfft, rfftfreq
from src.preprocessing.methods import preprocess_ecg, preprocess_gsr, preprocess_ppg, get_ECG_features, get_PPG_features, get_ECG_PPG_PTT, get_GSR_features

signal_columns = {
    'ECG': {
        'timestamp': 'Shimmer_820D_Timestamp_Unix_CAL',
        'LA_RA': 'Shimmer_820D_ECG_LA-RA_24BIT_CAL',
        'LL_LA': 'Shimmer_820D_ECG_LL-LA_24BIT_CAL',
        'LL_RA': 'Shimmer_820D_ECG_LL-RA_24BIT_CAL',
        'Vx_RL': 'Shimmer_820D_ECG_Vx-RL_24BIT_CAL'
    },
    'PPG': {
        'timestamp': 'id95AE_Timestamp_Unix_CAL',
        'ppg': 'id95AE_PPG_A13_CAL'
    },
    'GSR': {
        'timestamp': 'id95AE_Timestamp_Unix_CAL',
        'gsr': 'id95AE_GSR_Skin_Conductance_CAL'
    }
}

def extract_signal(df, data_type: str):
    """
        Extracts biosignal columns from a DataFrame based on data_type.

        Args:
            df (pd.DataFrame): The raw DataFrame containing Shimmer sensor data.
            data_type (str): The type of signal to extract.
                             One of 'ECG', 'PPG', 'GSR'.

        Returns:
            dict: A dictionary where keys are channel names (e.g., 'LA_RA') and values are the signals as 1D NumPy arrays.
                  Returns an empty dict if data_type is invalid.
    """
    if data_type not in signal_columns:
        raise Exception(f"Error: Invalid data_type '{data_type}'. Valid types are {list(signal_columns.keys())}")

    # signal Extraction
    signal = {key: df[col].values for key, col in signal_columns[data_type].items()}
    return signal

class BioSignalDataset:
    """
    Container class for managing biosignal data (ECG, PPG, GSR) for multiple
    subjects, sessions, and trials.

    This class provides utilities for:
      - Loading raw CSV data into a nested dictionary structure.
      - Preprocessing raw signals using filter-based methods.
      - Extracting features (HR, HRV, RR, SI, PTT, EDA) for each trial.
      - Loading label metadata from JSON files.

    Attributes:
        dir_path (str): Root directory where the subject/session/trial folders are stored.
        data_type (tuple[str]): Tuple of enabled signal types, subset of ('ECG', 'PPG', 'GSR').
        raw_data (dict): Nested dictionary storing raw signals.
        preprocessed_data (dict): Nested dictionary storing preprocessed signals.
        features (dict): Nested dictionary storing extracted features.
        labels (dict): Nested dictionary storing labels/annotations.
    """

    def __init__(self, dir_path: str, *data_type):
        """
        Initialize the BioSignalDataset object.

        Args:
            dir_path (str):
                Root directory path containing per-subject subfolders.
                Each subfolder is typically named like '1_1_001_V1' and may
                contain 'ECG', 'PPG', 'GSR' folders.
            *data_type (str):
                One or more signal types to handle. Each entry must be one of:
                'ECG', 'PPG', 'GSR'.

        Raises:
            ValueError:
                If any element in data_type is not one of ('ECG', 'PPG', 'GSR').

        Returns:
            None
        """
        if any(d not in ['ECG', 'PPG', 'GSR'] for d in data_type):
            raise ValueError(
                f"Invalid data type is included in {data_type}. "
                f"Must be one or multiple types among ('ECG', 'PPG', 'GSR')"
            )

        self.dir_path = dir_path
        self.data_type = data_type

        self.raw_data = {}          # Nested dict: subject → session → trial → signals
        self.preprocessed_data = {} # Nested dict: same structure, but filtered signals
        self.features = {}          # Nested dict: same structure, but feature values
        self.labels = {}            # Nested dict: label metadata (e.g., Q1/Q2, craving level)

    # ------------------------
    # Load raw data from CSV
    # ------------------------
    def get_raw_data(self, save_pkl: str = None, load_pkl: str = None):
        """
        Load raw biosignal data from disk into a nested dictionary.

        Data structure (example):
            raw_data[subject_id][session_id][trial_id] = {
                'ECG': { 'timestamp': ..., 'LA_RA': ..., ... },
                'PPG': { 'timestamp': ..., 'ppg': ... },
                'GSR': { 'timestamp': ..., 'gsr': ... },
            }

        The function can either:
          - Load an existing serialized dictionary from a pickle file (load_pkl), or
          - Scan the directory tree, read CSV files, and build the dictionary from scratch.

        Args:
            save_pkl (str, optional):
                Path to a pickle file where the loaded raw_data dictionary will be saved.
                If None, the dictionary is not saved to disk.
            load_pkl (str, optional):
                Path to a pickle file containing a previously saved raw_data dictionary.
                If provided, the dictionary is loaded from this file instead of reading CSVs.

        Returns:
            dict:
                Nested dictionary of raw signals, indexed by:
                [subject_id][session_id][trial_id][signal_type].

        Side Effects:
            - Updates self.raw_data with the loaded dictionary.
            - Optionally writes the dictionary to `save_pkl` using joblib.
        """
        raw_data = {}

        if load_pkl:
            # Load pre-saved raw data from pickle
            try:
                raw_data = joblib.load(load_pkl)
                self.raw_data = raw_data

                return raw_data
            except Exception as e:
                print("pickle file load error")

        biosignal_data_file_name = []
        if 'ECG' in self.data_type:
            biosignal_data_file_name.append('ECG')
        if 'PPG' in self.data_type or 'GSR' in self.data_type:
            biosignal_data_file_name.append('PPG')

        # Iterate over subject/session directories (e.g., 1_1_001_V1)
        for sub_dir in tqdm(os.listdir(self.dir_path)):
            sub_dir_path = os.path.join(self.dir_path, sub_dir)
            if not os.path.exists(sub_dir_path):
                print(f'Skipping {sub_dir_path}, does not exist')
                continue

            # Example: subject_id: 1_1_001, session_id: V1
            subject_id, session_id = sub_dir.rsplit("_", 1)

            for fname in biosignal_data_file_name:
                data_dir_path = os.path.join(sub_dir_path, fname)
                if not os.path.exists(data_dir_path):
                    print(f'Skipping {data_dir_path}, does not exist')
                    continue

                # Iterate over trial folders (e.g., low, mid, high)
                for trial_name in os.listdir(data_dir_path):
                    trial_dir_path = os.path.join(data_dir_path, trial_name)
                    if not os.path.exists(trial_dir_path):
                        print(f'Skipping {trial_dir_path}, does not exist')
                        continue

                    # Iterate over CSV files (e.g., low1.csv, mid2.csv, ...)
                    for trial_file_name in os.listdir(trial_dir_path):
                        if not trial_file_name.endswith('.csv'):
                            continue

                        file_path = os.path.join(trial_dir_path, trial_file_name)
                        raw_df = pd.read_csv(file_path)

                        trial_id = trial_file_name.split('.')[0]

                        # Initialize nested structure
                        if subject_id not in raw_data:
                            raw_data[subject_id] = {}
                        if session_id not in raw_data[subject_id]:
                            raw_data[subject_id][session_id] = {}
                        if trial_id not in raw_data[subject_id][session_id]:
                            raw_data[subject_id][session_id][trial_id] = {}

                        # Extract signals depending on folder type
                        if fname == 'ECG':
                            raw_data[subject_id][session_id][trial_id]['ECG'] = extract_signal(raw_df, 'ECG')
                        if fname == 'PPG':
                            if 'PPG' in self.data_type:
                                raw_data[subject_id][session_id][trial_id]['PPG'] = extract_signal(raw_df, 'PPG')
                            if 'GSR' in self.data_type:
                                raw_data[subject_id][session_id][trial_id]['GSR'] = extract_signal(raw_df, 'GSR')

        self.raw_data = raw_data

        if save_pkl:
            joblib.dump(raw_data, save_pkl)

        return raw_data

    # ------------------------
    # Preprocess raw data
    # ------------------------
    def get_preprocessed_data(self, save_pkl: str = None, load_pkl: str = None, filter_type = 'lowpass'):
        """
        Apply preprocessing (e.g., bandpass/lowpass filtering) to the raw data.

        The output has the same nested structure as self.raw_data but with
        each signal replaced by its preprocessed version.

        Args:
            save_pkl (str, optional):
                Path to a pickle file where the preprocessed_data dictionary
                will be saved. If None, nothing is saved to disk.
            load_pkl (str, optional):
                Path to a pickle file from which to load preprocessed data.
                If provided, the method will skip processing and simply load
                the dictionary from this file.

        Returns:
            dict:
                Nested dictionary of preprocessed signals, indexed by:
                [subject_id][session_id][trial_id][signal_type].

        Side Effects:
            - Updates self.preprocessed_data.
            - Optionally serializes the dictionary to `save_pkl`.
        """
        preprocessed_data = {}

        if load_pkl:
            # Load pre-saved raw data from pickle
            try:
                preprocessed_data = joblib.load(load_pkl)
                self.preprocessed_data = preprocessed_data

                return preprocessed_data
            except Exception as e:
                print("pickle file load error")

        # Iterate: subject → session → trial
        for subj_id, sessions in tqdm(self.raw_data.items()):
            preprocessed_data[subj_id] = {}

            for sess_id, trials in sessions.items():
                preprocessed_data[subj_id][sess_id] = {}

                for trial_id, trial_dict in trials.items():
                    # trial_dict: {'ECG': {...}, 'PPG': {...}, 'GSR': {...}}
                    out_trial = {}

                    if 'ECG' in trial_dict:
                        out_trial['ECG'] = preprocess_ecg(trial_dict['ECG'], filter_type= filter_type)

                    if 'PPG' in trial_dict:
                        out_trial['PPG'] = preprocess_ppg(trial_dict['PPG'], filter_type= filter_type)

                    if 'GSR' in trial_dict:
                        out_trial['GSR'] = preprocess_gsr(trial_dict['GSR'])

                    preprocessed_data[subj_id][sess_id][trial_id] = out_trial

        self.preprocessed_data = preprocessed_data

        if save_pkl:
            joblib.dump(preprocessed_data, save_pkl)

        return preprocessed_data

    # ------------------------
    # Feature Extraction
    # ------------------------
    def get_features(self, save_pkl: str = None, load_pkl: str = None, window_sec: float = 10.0, overlap_ratio: float = 0.5, save_ptt_raw_csv = None, ptt_range=(0.1,0.6)):
        """
        Extract features (HR, HRV, RR, SI, PTT, EDA-related metrics) per trial or segment
        from preprocessed signals.

        If window_sec is specified (e.g., 10.0, 20.0, 30.0), the signal is divided into 
        sliding windows with the given overlap_ratio (e.g., 0.5 for 50%).
        
        The resulting structure is:
            features[subject_id][session_id][trial_id_seg_id] = {
                'HR':  {...},
                'HRV': {...},
                'SI':  {...},
                'RR':  {...},
                'PTT': {...},
                'EDA': {...},
            }

        Args:
            save_pkl (str, optional): Path to save pickle.
            load_pkl (str, optional): Path to load pickle.
            window_sec (float, optional): Window size in seconds. If None, process whole trial.
            overlap_ratio (float, optional): Overlap ratio between consecutive windows.
            save_ptt_raw_csv (str, optional): Optional. Path to save unified raw PTT bead-level data (.csv).
        """
        features = {}
        all_ptt_debug_dfs = []

        if load_pkl:
            try:
                features = joblib.load(load_pkl)
                self.features = features
                return features
            except Exception as e:
                print("pickle file load error")

        for subj_id, sessions in tqdm(self.preprocessed_data.items()):
            features[subj_id] = {}

            for sess_id, trials in sessions.items():
                features[subj_id][sess_id] = {}

                for trial_id, trial_dict in trials.items():
                    
                    t_start = float('inf')
                    t_end = float('-inf')
                    for sig_type in ['ECG', 'PPG', 'GSR']:
                        if sig_type in trial_dict and 'timestamp' in trial_dict[sig_type]:
                            ts = trial_dict[sig_type]['timestamp']
                            if len(ts) > 0:
                                t_start = min(t_start, ts[0])
                                t_end = max(t_end, ts[-1])
                                
                    if t_start == float('inf'):
                        continue
                        
                    if window_sec is not None:
                        win_ms = int(window_sec * 1000)
                        step_ms = int(win_ms * (1.0 - overlap_ratio))
                    else:
                        win_ms = int(t_end - t_start) + 1
                        step_ms = win_ms
                        
                    current_start = t_start
                    win_idx = 0
                    
                    while current_start + win_ms <= t_end or (win_idx == 0 and current_start <= t_end and window_sec is None):
                        if current_start + win_ms > t_end and window_sec is not None:
                            break # Drop incomplete trailing window
                            
                        current_end = current_start + win_ms
                        
                        windowed_trial = {}
                        for sig_type in ['ECG', 'PPG', 'GSR']:
                            if sig_type in trial_dict and 'timestamp' in trial_dict[sig_type]:
                                ts = trial_dict[sig_type]['timestamp']
                                mask = (ts >= current_start) & (ts < current_end)
                                if np.sum(mask) == 0:
                                    continue
                                
                                windowed_trial[sig_type] = {}
                                for ch, val in trial_dict[sig_type].items():
                                    windowed_trial[sig_type][ch] = np.asarray(val)[mask]
                        
                        if window_sec is not None:
                            window_id = f"{trial_id}_w{win_idx:03d}"
                        else:
                            window_id = trial_id

                        out_trial = {
                            "HR": {},
                            "HRV": {},
                            "SI": {},
                            "RR": {},
                        }
                        # ----- ECG features -----
                        if 'ECG' in windowed_trial:
                            hr_ecg, hrv_ecg, si_ecg, rr_ecg, _ = get_ECG_features(
                                windowed_trial['ECG'],
                                preprocess=False,
                                use_peak_fusion=True,
                                fusion_cluster_ms=100,
                                fusion_min_agree_leads=2,
                                fusion_outlier_ms=40,
                                correct_artifacts_per_lead=True,
                                post_fix_fused_peaks=True,
                                fused_fix_method="Kubios",
                            )
                            out_trial['HR']['ECG'] = hr_ecg
                            out_trial['HRV']['ECG'] = hrv_ecg
                            out_trial['SI']['ECG'] = si_ecg
                            out_trial['RR']['ECG'] = rr_ecg

                        # ----- PPG features -----
                        if 'PPG' in windowed_trial:
                            hr_ppg, hrv_ppg, si_ppg, rr_ppg = get_PPG_features(
                                windowed_trial['PPG'],
                                preprocess=False,
                                post_fix_ppg_peaks=True,
                                ppg_fix_method="Kubios",
                            )
                            out_trial['HR']['PPG'] = hr_ppg
                            out_trial['HRV']['PPG'] = hrv_ppg
                            out_trial['SI']['PPG'] = si_ppg
                            out_trial['RR']['PPG'] = rr_ppg

                        # PTT (requires both ECG and PPG)
                        if 'ECG' in windowed_trial and 'PPG' in windowed_trial:
                            ptt_res = get_ECG_PPG_PTT(
                                windowed_trial['ECG'],
                                windowed_trial['PPG'],
                                preprocess=False,
                                use_peak_fusion=True,
                                fusion_cluster_ms=100,
                                fusion_min_agree_leads=2,
                                fusion_outlier_ms=40,
                                correct_artifacts_per_lead=True,
                                post_fix_fused_peaks=True,
                                fused_fix_method="Kubios",
                                ppg_method="elgendi",
                                post_fix_ppg_peaks=True,
                                ppg_fix_method="Kubios",
                                rr_ref_method="median",   # recommended
                                return_debug_df=(save_ptt_raw_csv is not None),
                                pair_lo_ratio = ptt_range[0],
                                pair_hi_ratio = ptt_range[1],
                            )

                            if save_ptt_raw_csv is not None:
                                if "fused_debug_df" in ptt_res:
                                    debug_df = ptt_res.pop("fused_debug_df")
                                    debug_df.insert(0, "window", window_id)
                                    debug_df.insert(0, "trial", trial_id)
                                    debug_df.insert(0, "session", sess_id)
                                    debug_df.insert(0, "subject", subj_id)
                                    all_ptt_debug_dfs.append(debug_df)

                            out_trial['PTT'] = ptt_res

                        # EDA / GSR features
                        if 'GSR' in windowed_trial:
                            out_trial['EDA'] = get_GSR_features(
                                windowed_trial['GSR'], preprocess=False
                            )
                            
                        features[subj_id][sess_id][window_id] = out_trial
                        
                        current_start += step_ms
                        win_idx += 1

            # NOTE: 'break' here limits processing to the first session only.
            # Remove this if you want to process all sessions.
            # break

        self.features = features

        if save_pkl:
            joblib.dump(features, save_pkl)
            
        if save_ptt_raw_csv is not None and len(all_ptt_debug_dfs) > 0:
            pd.concat(all_ptt_debug_dfs, ignore_index=True).to_csv(save_ptt_raw_csv, index=False)
            print(f"Saved raw PTT dataset to {save_ptt_raw_csv}")

        return features

    # ------------------------
    # Label Loading
    # ------------------------
    def get_labels(self, label_json_path: str = r"../data/labels/alcohol.json"):
        """
        Load label information (e.g., craving scores, group labels) from a JSON file.

        The JSON file is expected to contain a nested structure that can be
        indexed by subject/session/trial, or some other agreed-upon format.

        Args:
            label_json_path (str, optional):
                Path to the JSON file containing label information.
                Default is '../data/labels/alcohol.json'.

        Returns:
            dict:
                Parsed JSON object representing label information.

        Side Effects:
            - Updates self.labels with the loaded dictionary.
        """
        with open(label_json_path, 'r') as f:
            data = json.load(f)

        self.labels = data
        return data

    def show_plot(self, subject_name, session_id, trial_id, sig_type, channel, fs, unit, type = 'time'):
        """Quick-look plot (time-domain or FFT) of a raw vs. preprocessed signal for one subject/session/trial/channel. Debugging aid, not used by the paper pipeline."""
        # os.makedirs(save_dir_time, exist_ok=True)
        # os.makedirs(save_dir_fft, exist_ok=True)
        raw = self.raw_data[subject_name][session_id][trial_id][sig_type][channel]
        filtered = self.preprocessed_data[subject_name][session_id][trial_id][sig_type][channel]

        # ---- Time-domain ----
        if type == 'time':
            t = np.arange(len(raw)) / fs
            plt.figure(figsize=(10, 4))
            # plt.plot(t, raw, label="Raw", alpha=0.5)
            plt.plot(t, filtered, label=f"Filtered")
            plt.xlabel("Time (s)")
            plt.ylabel(f"Amplitude ({unit})")
            plt.title(f"{subject_name}_{session_id}_{trial_id}_{channel} (Time)")
            plt.legend()
            plt.tight_layout()
            plt.ion()
            plt.show()
            
            # plt.savefig(os.path.join(save_dir_time, f"{subject_name}_{filename_prefix}_time_{cutoff:.2f}.png"), dpi=200)
            plt.close()
        elif type == 'frequency_log':
            # ---- FFT (Improved version: log scale + limited freq) ----
            N = len(filtered)
            freq = rfftfreq(N, 1/fs)

            fft_raw = np.abs(rfft(raw))
            fft_filt = np.abs(rfft(filtered))

            # log amplitude (use log10, avoiding zeros)
            fft_raw_log = np.log10(fft_raw + 1e-8)
            fft_filt_log = np.log10(fft_filt + 1e-8)

            plt.figure(figsize=(10, 4))
            plt.plot(freq, fft_raw_log, label="Raw", alpha=0.5)
            plt.plot(freq, fft_filt_log, label=f"Filtered Hz")

            # ---- Frequency range limit ----
            plt.xlim(0, 10)  # <==== adjust here (0~10Hz)

            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Log Magnitude (log10)")
            plt.title(f"{subject_name}_{session_id}_{trial_id}_{channel} (FFT, log scale)")
            plt.legend()
            plt.tight_layout()
            plt.ion()
            plt.show()
            # plt.savefig(os.path.join(save_dir_fft, f"{subject_name}_{filename_prefix}_fft_{cutoff:.2f}.png"), dpi=200)
            plt.close()
        elif type == 'frequency':
            # ---- FFT ----
            N = len(filtered)
            freq = rfftfreq(N, 1/fs)

            # fft_raw = np.abs(rfft(raw))
            fft_filt = np.abs(rfft(filtered))


            plt.figure(figsize=(10, 4))
            # plt.plot(freq, fft_raw, label="Raw", alpha=0.5)
            plt.plot(freq, fft_filt, label=f"Filtered Hz")

            # ---- Frequency range limit ----
            plt.xlim(0, 10)  # <==== adjust here (0~10Hz)

            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Magnitude")
            plt.title(f"{subject_name}_{session_id}_{trial_id}_{channel} (FFT)")
            plt.legend()
            plt.tight_layout()
            plt.ion()
            plt.show()
            # plt.savefig(os.path.join(save_dir_fft, f"{subject_name}_{filename_prefix}_fft_{cutoff:.2f}.png"), dpi=200)
            plt.close()

    def get_feature_table_long(self, feature: str = None) -> pd.DataFrame:
        """
        Extract features into a unified long-format DataFrame.

        If `feature` is provided (e.g., 'ECG'), only that specific feature set is extracted.
        If `feature` is None, all features in the trial dictionary are collected.

        This function supports heterogeneous feature representations:
        - scalar values (e.g., HR)
        - dictionaries of metrics
        - pandas Series
        - pandas DataFrames (e.g., HRV with multiple metrics)

        Output schema:
            subject | session | trial | channel | metric | value
        """
        rows = []

        for subj_id, subj_dict in self.features.items():
            for sess_id, sess_dict in subj_dict.items():
                for trial_id, trial_dict in sess_dict.items():

                    # If specific feature not provided, extract everything available
                    target_features = [feature] if feature is not None else list(trial_dict.keys())

                    for f_name in target_features:
                        if f_name not in trial_dict:
                            continue

                        feat_obj = trial_dict[f_name]

                        # Case A: feature is stored as a dictionary
                        # (channel -> scalar OR channel -> dict/Series/DataFrame)
                        if isinstance(feat_obj, dict):
                            for ch, v in feat_obj.items():

                                # Case A-1: scalar feature value (e.g., HR)
                                if isinstance(v, (int, float, np.number)) and not isinstance(v, (pd.DataFrame, pd.Series)):
                                    rows.append({
                                        "subject": subj_id,
                                        "session": sess_id,
                                        "trial": trial_id,
                                        "channel": f_name,
                                        "metric": ch,
                                        "value": float(v)
                                    })
                                    continue

                                # Case A-2: dictionary of metrics (metric -> scalar OR metric -> DataFrame/Series)
                                if isinstance(v, dict):
                                    for m, mv in v.items():
                                        # Handle scalar
                                        if isinstance(mv, (int, float, np.number)) and not isinstance(mv, (pd.DataFrame, pd.Series)):
                                            rows.append({
                                                "subject": subj_id,
                                                "session": sess_id,
                                                "trial": trial_id,
                                                "channel": f"{f_name}_{ch}",
                                                "metric": m,
                                                "value": float(mv)
                                            })
                                        # Handle nested DataFrame (Common in PTT)
                                        elif isinstance(mv, pd.DataFrame):
                                            if len(mv) == 1:
                                                s = mv.iloc[0]
                                            else:
                                                s = mv.select_dtypes(include="number").mean(axis=0)
                                            for sub_m, sub_mv in s.items():
                                                if pd.notna(sub_mv):
                                                    rows.append({
                                                        "subject": subj_id,
                                                        "session": sess_id,
                                                        "trial": trial_id,
                                                        "channel": f"{f_name}_{ch}_{m}",
                                                        "metric": sub_m,
                                                        "value": float(sub_mv)
                                                    })
                                        # Handle nested Series
                                        elif isinstance(mv, pd.Series):
                                            for sub_m, sub_mv in mv.items():
                                                if pd.notna(sub_mv) and isinstance(sub_mv, (int, float, np.number)):
                                                    rows.append({
                                                        "subject": subj_id,
                                                        "session": sess_id,
                                                        "trial": trial_id,
                                                        "channel": f"{f_name}_{ch}_{m}",
                                                        "metric": sub_m,
                                                        "value": float(sub_mv)
                                                    })
                                    continue

                                # Case A-3: DataFrame of metrics (Common for HRV/PTT)
                                if isinstance(v, pd.DataFrame):
                                    if len(v) == 1:
                                        s = v.iloc[0]
                                    else:
                                        s = v.select_dtypes(include="number").mean(axis=0)
                                    
                                    for m, mv in s.items():
                                        if pd.notna(mv) and isinstance(mv, (int, float, np.number)):
                                            rows.append({
                                                "subject": subj_id,
                                                "session": sess_id,
                                                "trial": trial_id,
                                                "channel": f"{f_name}_{ch}",
                                                "metric": m,
                                                "value": float(mv)
                                            })
                                    continue

                                # Case A-4: Series of metrics
                                if isinstance(v, pd.Series):
                                    for m, mv in v.items():
                                        if isinstance(mv, (int, float, np.number)) and pd.notna(mv):
                                            rows.append({
                                                "subject": subj_id,
                                                "session": sess_id,
                                                "trial": trial_id,
                                                "channel": f"{f_name}_{ch}",
                                                "metric": m,
                                                "value": float(mv)
                                            })
                                    continue

                        # Case B: feature is stored as a scalar directly
                        elif isinstance(feat_obj, (int, float, np.number)):
                            rows.append({
                                "subject": subj_id,
                                "session": sess_id,
                                "trial": trial_id,
                                "channel": f_name,
                                "metric": "value",
                                "value": float(feat_obj)
                            })
                        
                        # Case C: feature is stored as a DataFrame directly
                        elif isinstance(feat_obj, pd.DataFrame):
                            if len(feat_obj) == 1:
                                s = feat_obj.iloc[0]
                            else:
                                s = feat_obj.select_dtypes(include="number").mean(axis=0)
                            for m, mv in s.items():
                                if pd.notna(mv) and isinstance(mv, (int, float, np.number)):
                                    rows.append({
                                        "subject": subj_id,
                                        "session": sess_id,
                                        "trial": trial_id,
                                        "channel": f_name,
                                        "metric": m,
                                        "value": float(mv)
                                    })

        return pd.DataFrame(rows)

    def get_feature_table_wide(self, feature: str) -> pd.DataFrame:
        """
        Convert extracted features into a wide-format DataFrame.

        - If only a single metric exists (e.g., HR),
        channels become columns.
        - If multiple metrics exist (e.g., HRV),
        columns are expanded as (metric, channel) pairs
        and optionally flattened as 'metric__channel'.
        """
        df_long = self.get_feature_table_long(feature)

        # Case 1: single metric (e.g., HR) → channels as columns
        if df_long["metric"].nunique() == 1:
            return df_long.pivot_table(
                index=["subject", "session", "trial"],
                columns="channel",
                values="value"
            ).reset_index()

        # Case 2: multiple metrics (e.g., HRV) → (metric, channel) columns
        df_wide = df_long.pivot_table(
            index=["subject", "session", "trial"],
            columns=["metric", "channel"],
            values="value"
        )

        # Flatten MultiIndex columns for easier downstream processing
        df_wide.columns = [f"{m}__{ch}" for m, ch in df_wide.columns]

        return df_wide.reset_index()
