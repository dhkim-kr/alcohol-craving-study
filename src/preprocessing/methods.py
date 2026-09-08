# methods.py

import os
import pandas as pd
import numpy as np
from scipy.signal import lombscargle, butter, filtfilt, iirfilter, periodogram
import neurokit2 as nk
import warnings
from neurokit2.misc import NeuroKitWarning
from scipy.stats import pearsonr, ConstantInputWarning
from collections import Counter
from math import sqrt

warnings.filterwarnings(action='ignore')
warnings.filterwarnings('ignore', category=ConstantInputWarning)
warnings.filterwarnings("ignore", category=NeuroKitWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

FS_ECG = 512
FS_PPG = 51.2
FS_GSR = 51.2
ECG_BAND = (0.5, 40)
PPG_BAND = (0.3, 5)
GSR_BAND = (0.1, 1)
RR_BAND = (0.1, 0.4)
ORDER = 4


def zscore_normalize(data: np.ndarray) -> np.ndarray:
    """
    Apply Z-score normalization to a 1D signal.

    Args:
        data (np.ndarray):
            Input 1D array representing a single-channel signal.

    Returns:
        np.ndarray:
            Z-score normalized signal. If the standard deviation is zero,
            the mean is removed but the scale is not changed.
    """
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return data - mean
    return (data - mean) / std


def butter_bandpass(sig: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    """
    Apply a Butterworth band-pass filter to a signal.

    Args:
        sig (np.ndarray):
            Input 1D signal to be filtered.
        fs (float):
            Sampling frequency of the signal (Hz).
        low (float):
            Low cut-off frequency of the band-pass filter (Hz).
        high (float):
            High cut-off frequency of the band-pass filter (Hz).
        order (int, optional):
            Order of the Butterworth filter. Default is 4.

    Returns:
        np.ndarray:
            Band-pass filtered signal.
    """
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, sig)


def butter_lowpass(sig: np.ndarray, fs: float, cutoff: float, order: int = 4) -> np.ndarray:
    """
    Apply a Butterworth low-pass filter to a signal.

    Args:
        sig (np.ndarray):
            Input 1D signal to be filtered.
        fs (float):
            Sampling frequency of the signal (Hz).
        cutoff (float):
            Cut-off frequency of the low-pass filter (Hz).
        order (int, optional):
            Order of the Butterworth filter. Default is 4.

    Returns:
        np.ndarray:
            Low-pass filtered signal.
    """
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, sig)


# ---- ECG preprocessing ----
def preprocess_ecg(ecg_dict: dict,
                   fs_ecg: float = FS_ECG,
                   lowcut: float = ECG_BAND[0],
                   highcut: float = ECG_BAND[1],
                   filter_type: str = 'lowpass',
                   fix_polarity=True) -> dict:
    """
    Filter and z-score normalize each ECG lead.

    Args:
        ecg_dict (dict):
            {"timestamp": np.ndarray, "<lead_name>": np.ndarray, ...}
        fs_ecg (float, optional):
            ECG sampling rate (Hz). Default is FS_ECG.
        lowcut (float, optional):
            Low cut-off frequency (Hz), used when filter_type="bandpass".
        highcut (float, optional):
            High/low-pass cut-off frequency (Hz).
        filter_type (str, optional):
            "lowpass" or "bandpass". Default is "lowpass".
        fix_polarity (bool, optional):
            If True, invert the "Vx_RL" lead (its raw polarity is flipped
            relative to the other leads for this electrode configuration).

    Returns:
        dict:
            Same structure as `ecg_dict`, with each lead filtered and
            z-score normalized.
    """
    out = {}
    out['timestamp'] = ecg_dict['timestamp']

    for ch_name, sig in ecg_dict.items():
        if ch_name == 'timestamp':
            continue

        if filter_type == 'lowpass':
            filtered = butter_lowpass(sig, fs_ecg, highcut)
        elif filter_type == 'bandpass':
            filtered = butter_bandpass(sig, fs_ecg, lowcut, highcut)
        else:
            raise ValueError(f"Unsupported ECG filter_type: {filter_type}")

        out[ch_name] = zscore_normalize(np.asarray(filtered, dtype=float))

        if fix_polarity and ch_name == 'Vx_RL':
            out[ch_name] = out[ch_name] * -1

    return out


# ---- PPG preprocessing ----
def preprocess_ppg(ppg_dict: dict,
                   fs_ppg: float = FS_PPG,
                   lowcut: float = PPG_BAND[0],
                   highcut: float = PPG_BAND[1],
                   filter_type: str = 'lowpass') -> dict:
    """
    Filter and z-score normalize the PPG signal.

    Args:
        ppg_dict (dict):
            {"timestamp": np.ndarray, "ppg": np.ndarray}
        fs_ppg (float, optional):
            PPG sampling rate (Hz). Default is FS_PPG.
        lowcut (float, optional):
            Low cut-off frequency (Hz), used when filter_type="bandpass".
        highcut (float, optional):
            High/low-pass cut-off frequency (Hz).
        filter_type (str, optional):
            "lowpass" or "bandpass". Default is "lowpass".

    Returns:
        dict:
            Same structure as `ppg_dict`, with "ppg" filtered and z-score
            normalized.
    """
    out = {}
    out['timestamp'] = ppg_dict['timestamp']

    if filter_type == 'lowpass':
        filtered = butter_lowpass(ppg_dict['ppg'], fs_ppg, highcut)
    elif filter_type == 'bandpass':
        filtered = butter_bandpass(ppg_dict['ppg'], fs_ppg, lowcut, highcut)
    else:
        raise ValueError(f"Unsupported PPG filter_type: {filter_type}")

    out['ppg'] = zscore_normalize(np.asarray(filtered, dtype=float))
    return out

# ---- GSR preprocessing ----
def preprocess_gsr(gsr_dict: dict,
                   fs_gsr: float = FS_GSR,
                   highcut: float = GSR_BAND[1]) -> dict:
    """
    Preprocess GSR signal using a low-pass filter to keep slow tonic changes.

    Args:
        gsr_dict (dict):
            Dictionary of the form:
            {
                'timestamp': np.ndarray,
                'gsr': np.ndarray
            }
        fs_gsr (float, optional):
            GSR sampling rate (Hz). Default is FS_GSR.
        highcut (float, optional):
            Low-pass cut-off frequency (Hz). Default is GSR_BAND[1].

    Returns:
        dict:
            Dictionary with the same structure as `gsr_dict`, where 'gsr'
            is replaced by its low-pass filtered version.
    """
    out = {}
    out['timestamp'] = gsr_dict['timestamp']
    out['gsr'] = butter_lowpass(gsr_dict['gsr'], fs_gsr, cutoff=highcut)
    return out


# ---- HR ----
def get_hr(peaks, fs: float, len_sig: int) -> float:
    """
    Compute mean heart rate (HR) from detected peaks.

    Args:
        peaks:
            Peak representation passed to `nk.signal_rate`. Typically the
            binary peak array or the peak dict produced by NeuroKit2 (e.g. info).
        fs (float):
            Sampling rate of the signal (Hz).
        len_sig (int):
            Length of the underlying signal, used as desired_length
            in `nk.signal_rate`.

    Returns:
        float:
            Mean heart rate in beats per minute (bpm). If computation fails,
            returns `np.nan`.
    """
    try:
        rate = nk.signal_rate(peaks, sampling_rate=fs, desired_length=len_sig)
        hr = np.mean(rate)
    except Exception as e:
        print(f"Error during HR processing: {e}")
        return np.nan

    return hr


# ---- HRV ----
def get_hrv(peaks, fs: float, show=False):
    """
    Compute HRV metrics from detected peaks using NeuroKit2.

    Args:
        peaks:
            Peak representation compatible with `nk.hrv`. For NeuroKit2,
            this is typically the dict returned by `nk.ecg_peaks` or
            `nk.ppg_peaks` (e.g., the `peaks` output).
        fs (float):
            Sampling rate of the signal (Hz).

    Returns:
        pandas.DataFrame | None:
            DataFrame of HRV metrics if successful, None otherwise.
    """
    try:
        hrv = nk.hrv(peaks, sampling_rate=fs, show=show)
    except Exception as e:
        print(f"Error during HRV processing: {e}")
        return None

    return hrv


# ---- SI ----
def get_si(peaks, fs: float) -> float:
    """
    Compute Baevsky's Stress Index (SI) from peak locations.

    Args:
        peaks:
            Peak locations or indices used to compute inter-beat intervals.
            Commonly an array-like of R-peak indices or PPG peak indices.
        fs (float):
            Sampling rate of the signal (Hz).

    Returns:
        float:
            Baevsky Stress Index. Returns `np.nan` on failure or if not
            enough beats are available.
    """
    try:
        peak_locs = np.asarray(peaks, dtype=int)

        # Calculate IBI (Inter-Beat Interval) in seconds
        ibi = np.diff(peak_locs) / fs
        if len(ibi) < 2:  # Need at least 2 IBIs to calculate the range
            return np.nan

        # Baevsky's SI requires binning IBIs into 50ms (0.05s) bins
        ibi_binned = np.round(ibi / 0.05) * 0.05

        # Calculate Stress Index (SI) components
        ibi_counter = Counter(ibi_binned)  # Count occurrences of each binned IBI
        M0, M0_count = ibi_counter.most_common(1)[0]  # Mode of IBI
        AM0 = (M0_count / len(ibi)) * 100  # Amplitude of Mode (%)
        MxDMn = max(ibi) - min(ibi)        # Variation Range

        if M0 == 0 or MxDMn == 0:  # Avoid division by zero
            return np.nan

        si = sqrt(AM0 / (2 * M0 * MxDMn))

    except Exception as e:
        print(f"Error during SI processing: {e}")
        return np.nan

    return si


# ---- RR ----
def get_rr(peaks, fs: float, len_sig: int, rr_band: tuple = RR_BAND) -> float:
    """
    Estimate respiration rate (RR) in breaths per minute using Lomb-Scargle
    analysis on inter-beat intervals (IBI).

    Args:
        peaks:
            Peak locations or indices used to compute inter-beat intervals.
        fs (float):
            Sampling rate of the signal (Hz).
        len_sig (int):
            Length of the original signal. Used to check minimum duration
            (e.g., 5 seconds).

    Returns:
        float:
            Estimated respiration rate (breaths per minute). Returns `np.nan`
            if the signal is too short, too few valid IBIs are found, or
            the spectral analysis fails.
    """
    # Check for minimum signal length
    if len_sig < fs * 5:
        print("Warning: Input signal is too short for RR calculation (less than 5 seconds).")
        return np.nan

    try:
        peak_times = np.asarray(peaks) / fs

        # IBI filtering
        ibi = np.diff(peak_times)
        ibi_times = peak_times[1:]

        # Physiologically valid IBI range
        valid_mask = (ibi >= 0.4) & (ibi <= 1.33)
        ibi = ibi[valid_mask]
        ibi_times = ibi_times[valid_mask]

        # Check IBI count after filtering
        if len(ibi) < 4:
            print(f"Warning: Not enough valid IBIs after filtering. len(ibi) is {len(ibi)}.")
            return np.nan

        # Respiratory Frequency Estimation using the Lomb-Scargle Periodogram
        try:
            f_min, f_max = rr_band  # ex) (0.1, 0.4)
            freqs = np.linspace(f_min, f_max, 1000)
            angular_freqs = 2 * np.pi * freqs

            ibi_mean_removed = ibi - np.mean(ibi)
            psd = lombscargle(ibi_times, ibi_mean_removed, angular_freqs)

            if len(psd) == 0:
                return np.nan

            peak_idx = np.argmax(psd)
            peak_freq = freqs[peak_idx]

            rr = peak_freq * 60.0  # Hz -> breaths per minute

        except Exception as e:
            print(f"Lomb-Scargle calculation failed: {e}")
            return np.nan

    except Exception as e:
        print(f"Error during RR processing: {e}")
        return np.nan

    return rr

def _cluster_rpeaks_multilead(rpeaks_dict, fs_ecg, cluster_ms=100):
    """
    Cluster R-peak indices from multiple leads by proximity in time.
    Returns:
        clusters: list[list[tuple]]
            Each element is one beat cluster; each inner element is a
            (lead_name, peak_idx) tuple.
    """
    all_peaks = []
    for lead_name, arr in rpeaks_dict.items():
        arr = np.asarray(arr, dtype=int)
        for p in arr:
            all_peaks.append((lead_name, int(p)))

    if len(all_peaks) == 0:
        return []

    all_peaks = sorted(all_peaks, key=lambda x: x[1])

    thr = int(round(cluster_ms * fs_ecg / 1000.0))
    clusters = []
    current = [all_peaks[0]]
    last_idx = all_peaks[0][1]

    for item in all_peaks[1:]:
        _, cur_idx = item

        # Comparing against the last peak (rather than the cluster's first peak) is less sensitive to cluster breaks
        if abs(cur_idx - last_idx) <= thr:
            current.append(item)
        else:
            clusters.append(current)
            current = [item]

        last_idx = cur_idx

    clusters.append(current)
    return clusters


def _fuse_rpeaks_from_clusters(clusters, fs_ecg, min_agree_leads=2, outlier_ms=40):
    """
    Produce one final fused R-peak per cluster.
    Rules:
    1) If the same lead has duplicate detections in a cluster, keep the one closest to the cluster median.
    2) Discard clusters with fewer than min_agree_leads agreeing leads.
    3) Remove outliers relative to the median.
    4) Use the median of the remaining peaks as the fused peak.
    """
    fused_peaks = []
    outlier_thr = int(round(outlier_ms * fs_ecg / 1000.0))

    for cluster in clusters:
        if len(cluster) == 0:
            continue

        raw_times = np.array([x[1] for x in cluster], dtype=int)
        raw_med = int(np.median(raw_times))

        # Remove duplicate detections from the same lead
        best_per_lead = {}
        for lead_name, peak_idx in cluster:
            if lead_name not in best_per_lead:
                best_per_lead[lead_name] = peak_idx
            else:
                if abs(peak_idx - raw_med) < abs(best_per_lead[lead_name] - raw_med):
                    best_per_lead[lead_name] = peak_idx

        unique_times = np.array(list(best_per_lead.values()), dtype=int)

        if len(unique_times) < min_agree_leads:
            continue

        med = int(np.median(unique_times))
        keep = np.abs(unique_times - med) <= outlier_thr
        kept_times = unique_times[keep]

        if len(kept_times) < min_agree_leads:
            continue

        fused_idx = int(np.median(kept_times))
        fused_peaks.append(fused_idx)

    if len(fused_peaks) == 0:
        return np.array([], dtype=int)

    fused_peaks = np.array(sorted(set(fused_peaks)), dtype=int)
    return fused_peaks


def _build_peaks_dataframe_from_indices(rpeak_indices, signal_length):
    """
    Build a binary DataFrame matching NeuroKit2's peaks format.
    """
    arr = np.zeros(signal_length, dtype=int)
    rpeak_indices = np.asarray(rpeak_indices, dtype=int)
    rpeak_indices = rpeak_indices[(rpeak_indices >= 0) & (rpeak_indices < signal_length)]
    arr[rpeak_indices] = 1
    return pd.DataFrame({"ECG_R_Peaks": arr})


def _extract_rpeaks_dict_from_info(info_dict):
    """
    Extract each lead's ECG_R_Peaks from the per-lead info dicts.
    """
    rpeaks_dict = {}
    for ch_name, info in info_dict.items():
        if info is None:
            continue
        if not isinstance(info, dict):
            continue
        if "ECG_R_Peaks" not in info:
            continue

        arr = np.asarray(info["ECG_R_Peaks"], dtype=int)
        if len(arr) == 0:
            continue

        arr = np.unique(arr)
        rpeaks_dict[ch_name] = arr

    return rpeaks_dict


def _post_fix_fused_rpeaks(rpeak_indices, fs_ecg, method="Kubios", iterative=True, show=False):
    """
    Post-process fused R-peaks with NeuroKit2's signal_fixpeaks.
    """
    rpeak_indices = np.asarray(rpeak_indices, dtype=int)
    rpeak_indices = np.unique(rpeak_indices)
    rpeak_indices = rpeak_indices[rpeak_indices >= 0]

    if len(rpeak_indices) < 3:
        return rpeak_indices, {}

    try:
        artifacts, fixed = nk.signal_fixpeaks(
            rpeak_indices,
            sampling_rate=fs_ecg,
            method=method,
            iterative=iterative,
            show=show,
        )
        fixed = np.asarray(fixed, dtype=int)
        fixed = np.unique(fixed)
        fixed = fixed[fixed >= 0]
        return fixed, artifacts

    except Exception as e:
        print(f"Warning: signal_fixpeaks failed on fused peaks: {e}")
        return rpeak_indices, {}

def _post_fix_peak_indices(peak_indices, fs, method="Kubios", iterative=True, show=False):
    """
    Generic peak post-fix for ECG/PPG peak indices using NeuroKit2 signal_fixpeaks.
    """
    peak_indices = np.asarray(peak_indices, dtype=int)
    peak_indices = np.unique(peak_indices)
    peak_indices = peak_indices[peak_indices >= 0]

    if len(peak_indices) < 3:
        return peak_indices, {}

    try:
        artifacts, fixed = nk.signal_fixpeaks(
            peak_indices,
            sampling_rate=fs,
            method=method,
            iterative=iterative,
            show=show,
        )
        fixed = np.asarray(fixed, dtype=int)
        fixed = np.unique(fixed)
        fixed = fixed[fixed >= 0]
        return fixed, artifacts
    except Exception as e:
        print(f"Warning: signal_fixpeaks failed: {e}")
        return peak_indices, {}

def _fuse_multilead_rpeaks_from_info(
    info_dict,
    signal_length,
    fs_ecg,
    cluster_ms=100,
    min_agree_leads=2,
    outlier_ms=40
):
    """
    Take each lead's info['ECG_R_Peaks'] and produce the fused R-peaks.
    """
    rpeaks_dict = _extract_rpeaks_dict_from_info(info_dict)

    if len(rpeaks_dict) == 0:
        return pd.DataFrame({"ECG_R_Peaks": np.zeros(signal_length, dtype=int)}), {
            "ECG_R_Peaks": np.array([], dtype=int)
        }

    clusters = _cluster_rpeaks_multilead(
        rpeaks_dict=rpeaks_dict,
        fs_ecg=fs_ecg,
        cluster_ms=cluster_ms
    )

    fused_indices = _fuse_rpeaks_from_clusters(
        clusters=clusters,
        fs_ecg=fs_ecg,
        min_agree_leads=min_agree_leads,
        outlier_ms=outlier_ms
    )

    fused_peaks = _build_peaks_dataframe_from_indices(fused_indices, signal_length)
    fused_info = {"ECG_R_Peaks": fused_indices}

    return fused_peaks, fused_info


def get_ECG_features(
    ecg_dict: dict,
    fs_ecg: float = FS_ECG,
    preprocess: bool = True,
    filter_type: str = "lowpass",
    fix_polarity: bool = True,
    show: bool = False,
    use_peak_fusion: bool = True,
    fusion_cluster_ms: float = 100,
    fusion_min_agree_leads: int = 2,
    fusion_outlier_ms: float = 40,
    correct_artifacts_per_lead: bool = True,
    post_fix_fused_peaks: bool = True,
    fused_fix_method: str = "Kubios",
):
    """
    Extract ECG-based features (HR, HRV, SI, RR) from multi-lead ECG.

    Args:
        ecg_dict (dict):
            {
                "timestamp": ...,
                "LA_RA": np.ndarray,
                "LL_LA": np.ndarray,
                ...
            }
        fs_ecg (float):
            ECG sampling rate.
        preprocess (bool):
            Whether to preprocess ECG.
        filter_type (str):
            Filter type used in preprocess_ecg().
        fix_polarity (bool):
            Whether to fix ECG polarity during preprocessing.
        show (bool):
            Passed to NeuroKit2 functions.
        use_peak_fusion (bool):
            If True, fuse R-peaks across leads and return fused features only.
            If False, return per-lead feature dicts.
        fusion_cluster_ms (float):
            Time window for clustering R-peaks across leads.
        fusion_min_agree_leads (int):
            Minimum number of agreeing leads for a valid fused beat.
        fusion_outlier_ms (float):
            Outlier rejection threshold around cluster median.
        correct_artifacts_per_lead (bool):
            If True, apply NeuroKit2 artifact correction in ecg_peaks() per lead.
        post_fix_fused_peaks (bool):
            If True, run nk.signal_fixpeaks() on fused R-peaks.
        fused_fix_method (str):
            Method for nk.signal_fixpeaks(). Usually "Kubios".

    Returns:
        If use_peak_fusion is False:
            HR, HRV, SI, RR, per_lead_info

        If use_peak_fusion is True:
            fused_hr, fused_hrv, fused_si, fused_rr, per_lead_info
    """
    HR = {}
    HRV = {}
    SI = {}
    RR = {}

    if ecg_dict is None:
        if use_peak_fusion:
            return np.nan, None, np.nan, np.nan, {}
        return HR, HRV, SI, RR, {}

    # --------------------------------------------------
    # Preprocess
    # --------------------------------------------------
    if preprocess:
        preprocessed_dict = preprocess_ecg(
            ecg_dict,
            filter_type=filter_type,
            fix_polarity=fix_polarity
        )
    else:
        preprocessed_dict = ecg_dict

    # --------------------------------------------------
    # Per-lead R-peak detection
    # --------------------------------------------------
    per_lead_info = {}
    per_lead_peaks = {}
    signal_length = None

    for ch_name, sig in preprocessed_dict.items():
        if ch_name == "timestamp":
            continue

        sig = np.asarray(sig, dtype=float)
        if signal_length is None:
            signal_length = len(sig)

        try:
            peaks, info = nk.ecg_peaks(
                sig,
                sampling_rate=fs_ecg,
                show=show,
                correct_artifacts=correct_artifacts_per_lead,
            )

            per_lead_peaks[ch_name] = peaks
            per_lead_info[ch_name] = info

        except Exception as e:
            print(f"Error during ECG peak detection ({ch_name}): {e}")
            per_lead_peaks[ch_name] = None
            per_lead_info[ch_name] = None

    # Case where no signal was available at all
    if signal_length is None:
        if use_peak_fusion:
            return np.nan, None, np.nan, np.nan, {}
        return HR, HRV, SI, RR, {}

    # --------------------------------------------------
    # Per-lead feature extraction
    # --------------------------------------------------
    if not use_peak_fusion:
        for ch_name in per_lead_info.keys():
            info = per_lead_info[ch_name]
            peaks = per_lead_peaks[ch_name]

            try:
                if info is None or peaks is None:
                    raise ValueError("Peak detection failed.")

                mean_hr = get_hr(info, fs_ecg, signal_length)
                hrv = get_hrv(peaks, fs_ecg, show=show)
                si = get_si(info["ECG_R_Peaks"], fs_ecg)
                rr = get_rr(info["ECG_R_Peaks"], fs_ecg, signal_length)

            except Exception as e:
                print(f"Error during ECG features processing ({ch_name}): {e}")
                mean_hr = np.nan
                hrv = None
                si = np.nan
                rr = np.nan

            HR[ch_name] = mean_hr
            HRV[ch_name] = hrv
            SI[ch_name] = si
            RR[ch_name] = rr

        return HR, HRV, SI, RR, per_lead_info

    # --------------------------------------------------
    # Fused feature extraction
    # --------------------------------------------------
    try:
        # 1) Fuse across leads
        fused_peaks, fused_info = _fuse_multilead_rpeaks_from_info(
            info_dict=per_lead_info,
            signal_length=signal_length,
            fs_ecg=fs_ecg,
            cluster_ms=fusion_cluster_ms,
            min_agree_leads=fusion_min_agree_leads,
            outlier_ms=fusion_outlier_ms,
        )

        fused_indices = np.asarray(fused_info["ECG_R_Peaks"], dtype=int)

        # 2) RR artifact correction after fusion
        fused_artifacts = {}
        if post_fix_fused_peaks and len(fused_indices) >= 3:
            fused_indices, fused_artifacts = _post_fix_fused_rpeaks(
                fused_indices,
                fs_ecg=fs_ecg,
                method=fused_fix_method,
                iterative=True,
                show=False,
            )

        # 3) Rebuild from the corrected fused peaks
        fused_info = {"ECG_R_Peaks": fused_indices}
        fused_peaks = _build_peaks_dataframe_from_indices(fused_indices, signal_length)

        # For record-keeping
        per_lead_info["Fused_R_Peaks"] = fused_indices
        per_lead_info["Fused_Artifacts"] = fused_artifacts

        # 4) Compute features
        if len(fused_indices) < 2:
            raise ValueError("Too few fused peaks after correction.")

        fused_hr = get_hr(fused_info, fs_ecg, signal_length)
        fused_hrv = get_hrv(fused_peaks, fs_ecg, show=show)
        fused_si = get_si(fused_info["ECG_R_Peaks"], fs_ecg)
        fused_rr = get_rr(fused_info["ECG_R_Peaks"], fs_ecg, signal_length)

    except Exception as e:
        print(f"Error during fused ECG features processing: {e}")
        fused_hr = np.nan
        fused_hrv = None
        fused_si = np.nan
        fused_rr = np.nan

    return fused_hr, fused_hrv, fused_si, fused_rr, per_lead_info

def get_PPG_features(ppg_dict: dict,
                     fs_ppg: float = FS_PPG,
                     preprocess: bool = True,
                     filter_type: str = 'lowpass',
                     method='elgendi',
                     post_fix_ppg_peaks: bool = True,
                     ppg_fix_method: str = "Kubios"
):
    """
    Extract PPG-based features (HR, HRV, SI, RR) from a single PPG channel.

    Args:
        ppg_dict (dict):
            {"timestamp": np.ndarray, "ppg": np.ndarray}
        fs_ppg (float, optional):
            PPG sampling rate. Default is FS_PPG.
        preprocess (bool, optional):
            Whether to run preprocess_ppg() first. Default True.
        filter_type (str, optional):
            Filter type passed to preprocess_ppg(). Default "lowpass".
        method (str, optional):
            Systolic-peak detection method passed to nk.ppg_peaks().
            Default "elgendi".
        post_fix_ppg_peaks (bool, optional):
            If True, run nk.signal_fixpeaks() on the detected peaks.
        ppg_fix_method (str, optional):
            Method for nk.signal_fixpeaks(). Usually "Kubios".

    Returns:
        tuple: (mean_hr, hrv, si, rr) -- same feature families as
        get_ECG_features(), computed from the single PPG channel.
        Falls back to (nan, None, nan, nan) if peak detection fails.
    """
    if preprocess:
        preprocessed_dict = preprocess_ppg(ppg_dict, filter_type=filter_type)
    else:
        preprocessed_dict = ppg_dict

    sig = preprocessed_dict['ppg']

    try:
        peaks, info = nk.ppg_peaks(
            sig,
            sampling_rate=fs_ppg,
            method=method,
            correct_artifacts=True
        )

        ppg_idx = np.asarray(info.get("PPG_Peaks", []), dtype=int)
        ppg_artifacts = {}

        if post_fix_ppg_peaks and len(ppg_idx) >= 3:
            ppg_idx, ppg_artifacts = _post_fix_peak_indices(
                ppg_idx,
                fs=fs_ppg,
                method=ppg_fix_method,
                iterative=True,
                show=False,
            )

        peaks = pd.DataFrame({"PPG_Peaks": np.zeros(len(sig), dtype=int)})
        ppg_idx = ppg_idx[(ppg_idx >= 0) & (ppg_idx < len(sig))]
        peaks.loc[ppg_idx, "PPG_Peaks"] = 1
        info = {"PPG_Peaks": ppg_idx}

        mean_hr = get_hr(info['PPG_Peaks'], fs_ppg, len(sig))
        hrv = get_hrv(peaks, fs_ppg)
        si = get_si(info['PPG_Peaks'], fs_ppg)
        rr = get_rr(info['PPG_Peaks'], fs_ppg, len(sig))

    except Exception as e:
        print(f"Error during PPG features processing: {e}")
        mean_hr = np.nan
        hrv = None
        si = np.nan
        rr = np.nan

    return mean_hr, hrv, si, rr

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_ECG_PPG_PTT(
    ecg_dict: dict,
    ppg_dict: dict,
    preprocess: bool = True,
    filter_type: str = "lowpass",
    resample_method: str = "cubic",
    min_pairs: int = 3,
    use_peak_fusion: bool = True,
    fusion_cluster_ms: float = 100.0,
    fusion_min_agree_leads: int = 2,
    fusion_outlier_ms: float = 40.0,
    correct_artifacts_per_lead: bool = True,
    post_fix_fused_peaks: bool = True,
    fused_fix_method: str = "Kubios",
    ppg_method: str = "elgendi",
    post_fix_ppg_peaks: bool = True,
    ppg_fix_method: str = "Kubios",
    pair_lo_ratio: float = 0.05,
    pair_hi_ratio: float = 0.95,
    rr_ref_method: str = "median",   # "median" or "mean"
    save_debug_plots: bool = False,
    debug_plot_dir: str | Path | None = None,
    save_debug_csv: bool = False,
    plot_prefix: str = "ptt",
    return_debug_df: bool = False,
) -> dict:
    """
    Extract PTT (pulse transit time) features from ECG R-peaks and PPG
    systolic peaks. (The function/variable name "PTT" is kept as-is in code.)

    Pairing rule:
      - For each successive R-peak interval [R_i, R_{i+1}]:
      - Look for candidate PPG systolic peaks in
        [R_i + pair_lo_ratio*RR_i, R_i + pair_hi_ratio*RR_i].
      - Compute a PTT beat only if there is exactly one candidate.
      - Discard the beat if there are zero or 2+ candidates.

    Parameters
    ----------
    save_debug_plots : bool
        If True, save distribution/outlier-inspection plots.
    debug_plot_dir : str | Path | None
        Directory to save plots/csv into.
    save_debug_csv : bool
        If True, save the beat-level raw PTT values as a csv.
    plot_prefix : str
        Prefix used for saved file names.
    """

    # ------------------------------------------------------------
    # helper
    # ------------------------------------------------------------
    def _robust_stats(x: np.ndarray) -> dict:
        """Distribution summary stats (mean/SD/CV/median/IQR/MAD/min/max/range/percentiles/skew/kurtosis) used for the PTT_* feature block."""
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]

        if x.size == 0:
            return {
                "Mean": np.nan,
                "SD": np.nan,
                "CV": np.nan,
                "Median": np.nan,
                "IQR": np.nan,
                "MAD": np.nan,
                "Min": np.nan,
                "Max": np.nan,
                "Range": np.nan,
                # "Prc05": np.nan,
                # "Prc10": np.nan,
                # "Prc25": np.nan,
                # "Prc75": np.nan,
                # "Prc90": np.nan,
                # "Prc95": np.nan,
                "Prc20" : np.nan,
                "Prc80" : np.nan,
                "Skewness": np.nan,
                "Kurtosis": np.nan,
                # "N": 0,
            }

        q25, q50, q75 = np.percentile(x, [25, 50, 75])
        iqr = q75 - q25
        mad = np.median(np.abs(x - q50))
        mean = float(np.mean(x))
        std = float(np.std(x, ddof=0))
        cv = float(std / mean) if np.isfinite(mean) and abs(mean) > 1e-12 else np.nan

        p05, p10, p90, p95 = np.percentile(x, [5, 10, 90, 95])
        p20, p80 = np.percentile(x, [20, 80])

        xc = x - mean
        m2 = np.mean(xc**2)
        m3 = np.mean(xc**3)
        m4 = np.mean(xc**4)
        if m2 > 1e-24:
            skew = float(m3 / (m2 ** 1.5))
            kurt = float(m4 / (m2 ** 2) - 3.0)
        else:
            skew, kurt = np.nan, np.nan

        return {
            "Mean": mean,
            "SD": std,
            "CV": cv,
            "Median": float(q50),
            "IQR": float(iqr),
            "MAD": float(mad),
            "Min": float(np.min(x)),
            "Max": float(np.max(x)),
            "Range": float(np.max(x) - np.min(x)),
            # "Prc05": float(p05),
            # "Prc10": float(p10),
            # "Prc25": float(q25),
            # "Prc75": float(q75),
            # "Prc90": float(p90),
            # "Prc95": float(p95),
            "Prc20": float(p20),
            "Prc80": float(p80),
            "Skewness": skew,
            "Kurtosis": kurt,
            # "N": int(x.size),
        }

    def _pair_r_to_unique_ppg_peak_by_rr_window(
        r_t: np.ndarray,
        ppk_t: np.ndarray,
        lo_ratio: float = 0.05,
        hi_ratio: float = 0.95,
        rr_ref_method: str = "median",
    ):
        """
        1) First check whether there is exactly one PPG peak inside each R-R interval.
        2) Only generate a PTT value if that single peak falls inside the rr_ref-based latency window.
        """
        r_t = np.asarray(r_t, dtype=float)
        ppk_t = np.asarray(ppk_t, dtype=float)

        meta = {
            "RRIntervals_Total": 0,
            "RRIntervals_ZeroPeakInRR": 0,
            "RRIntervals_MultiPeakInRR": 0,
            "RRIntervals_UniquePeakInRR": 0,
            "RRIntervals_OutsideWindow": 0,
            "RRIntervals_ValidPTT": 0,
            "ValidPTT_Ratio": np.nan,
        }

        debug_rows = []

        if r_t.size < 2 or ppk_t.size == 0:
            return np.array([], dtype=float), meta, pd.DataFrame(debug_rows)

        rr_all = np.diff(r_t)
        rr_all = rr_all[np.isfinite(rr_all) & (rr_all > 0)]
        if len(rr_all) == 0:
            return np.array([], dtype=float), meta, pd.DataFrame(debug_rows)

        if rr_ref_method == "median":
            rr_ref = float(np.median(rr_all))
        elif rr_ref_method == "mean":
            rr_ref = float(np.mean(rr_all))
        else:
            raise ValueError(f"Unsupported rr_ref_method: {rr_ref_method}")

        ptt_list = []
        total_rr = 0
        n_zero_rr = 0
        n_multi_rr = 0
        n_unique_rr = 0
        n_outside = 0

        for i in range(r_t.size - 1):
            r0 = r_t[i]
            r1 = r_t[i + 1]
            rr = r1 - r0

            if not np.isfinite(rr) or rr <= 0:
                continue

            total_rr += 1

            # 1) First check how many peaks fall inside the R-R interval
            left_rr = np.searchsorted(ppk_t, r0, side="right")
            right_rr = np.searchsorted(ppk_t, r1, side="left")
            cand_rr = ppk_t[left_rr:right_rr]
            n_cand_rr = len(cand_rr)

            row = {
                "beat_idx": i,
                "r0_sec": r0,
                "r1_sec": r1,
                "rr_sec": rr,
                "rr_ref_sec": rr_ref,
                "n_candidates_in_rr": n_cand_rr,
                "selected_ppg_peak_sec": np.nan,
                "win_lo_sec": np.nan,
                "win_hi_sec": np.nan,
                "ptt_ms": np.nan,
                "status": "discard",
            }

            if n_cand_rr == 0:
                n_zero_rr += 1
                row["status"] = "zero_peak_in_rr"
                debug_rows.append(row)
                continue

            if n_cand_rr > 1:
                n_multi_rr += 1
                row["status"] = "multi_peak_in_rr"
                debug_rows.append(row)
                continue

            # 2) If there is a unique peak inside the R-R interval, check the latency window
            n_unique_rr += 1
            p_t = cand_rr[0]

            lo_t = r0 + lo_ratio * rr_ref
            hi_t = r0 + hi_ratio * rr_ref

            row["selected_ppg_peak_sec"] = p_t
            row["win_lo_sec"] = lo_t
            row["win_hi_sec"] = hi_t

            if not (lo_t <= p_t <= hi_t):
                n_outside += 1
                row["status"] = "outside_window"
                debug_rows.append(row)
                continue

            ptt_ms = (p_t - r0) * 1000.0
            row["ptt_ms"] = ptt_ms
            row["status"] = "valid"
            debug_rows.append(row)

            ptt_list.append(ptt_ms)

        n_valid = len(ptt_list)
        meta["RRIntervals_Total"] = total_rr
        meta["RRIntervals_ZeroPeakInRR"] = n_zero_rr
        meta["RRIntervals_MultiPeakInRR"] = n_multi_rr
        meta["RRIntervals_UniquePeakInRR"] = n_unique_rr
        meta["RRIntervals_OutsideWindow"] = n_outside
        meta["RRIntervals_ValidPTT"] = n_valid
        meta["ValidPTT_Ratio"] = (n_valid / total_rr) if total_rr > 0 else np.nan

        return np.asarray(ptt_list, dtype=float), meta, pd.DataFrame(debug_rows)
        
    def _save_ptt_debug_plots(
        ptt_ms: np.ndarray,
        debug_df: pd.DataFrame,
        save_dir: Path,
        stem: str,
    ):
        """Save PTT distribution/QC debug plots (histogram etc.) to save_dir; only used when save_debug_plots=True."""
        save_dir.mkdir(parents=True, exist_ok=True)

        # 1) Histogram
        plt.figure(figsize=(7, 5), dpi=200)
        if len(ptt_ms) > 0:
            plt.hist(ptt_ms, bins=min(30, max(10, len(ptt_ms)//2)))
            med = np.median(ptt_ms)
            q1, q3 = np.percentile(ptt_ms, [25, 75])
            plt.axvline(med, linestyle="--", linewidth=1, label=f"Median={med:.1f} ms")
            plt.axvline(q1, linestyle=":", linewidth=1, label=f"Q1={q1:.1f}")
            plt.axvline(q3, linestyle=":", linewidth=1, label=f"Q3={q3:.1f}")
            plt.legend()
        plt.xlabel("PTT (ms)")
        plt.ylabel("Count")
        plt.title(f"{stem} | PTT Histogram")
        plt.tight_layout()
        plt.savefig(save_dir / f"{stem}_hist.png")
        plt.close()

        # 2) Boxplot
        plt.figure(figsize=(5, 5), dpi=200)
        if len(ptt_ms) > 0:
            plt.boxplot(ptt_ms, vert=True, showfliers=True)
        plt.ylabel("PTT (ms)")
        plt.title(f"{stem} | PTT Boxplot")
        plt.tight_layout()
        plt.savefig(save_dir / f"{stem}_boxplot.png")
        plt.close()

        # 3) Beat sequence plot
        plt.figure(figsize=(10, 4), dpi=200)
        if not debug_df.empty:
            valid_df = debug_df[debug_df["status"] == "valid"].copy()
            zero_df = debug_df[debug_df["status"] == "zero_peak"].copy()
            multi_df = debug_df[debug_df["status"] == "multi_peak"].copy()

            if not valid_df.empty:
                plt.plot(valid_df["beat_idx"].to_numpy(), valid_df["ptt_ms"].to_numpy(), marker="o", linewidth=1, label="Valid PTT")
            if not zero_df.empty:
                plt.scatter(zero_df["beat_idx"].to_numpy(), np.zeros(len(zero_df)) * np.nan, marker="x", label="Zero peak")
            if not multi_df.empty:
                plt.scatter(multi_df["beat_idx"].to_numpy(), np.zeros(len(multi_df)) * np.nan, marker="s", label="Multi peak")

        plt.xlabel("Beat index")
        plt.ylabel("PTT (ms)")
        plt.title(f"{stem} | Beat-wise PTT")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_dir / f"{stem}_beat_sequence.png")
        plt.close()

        # 4) Candidate count plot
        if not debug_df.empty:
            plt.figure(figsize=(10, 4), dpi=200)
            plt.plot(debug_df["beat_idx"].to_numpy(), debug_df["n_candidates"].to_numpy(), marker="o", linewidth=1)
            plt.xlabel("Beat index")
            plt.ylabel("# candidate PPG peaks")
            plt.title(f"{stem} | Candidate Count per RR Interval")
            plt.tight_layout()
            plt.savefig(save_dir / f"{stem}_candidate_count.png")
            plt.close()

    def _build_feature_df(ptt_ms: np.ndarray, meta: dict) -> pd.DataFrame:
        """Turn the beat-level PTT array into a single-row feature DataFrame via _robust_stats()."""
        ptt_for_stats = ptt_ms if len(ptt_ms) >= min_pairs else np.array([], dtype=float)

        feat = {}
        stats = _robust_stats(ptt_for_stats)
        for k, v in stats.items():
            feat[f"PTT_{k}"] = v

        # --------------------------------------------------------
        # QC/meta features are commented out because they were found
        # inconvenient to use downstream as model features. Re-enable
        # if needed.
        # --------------------------------------------------------
        # for k, v in meta.items():
        #     feat[f"PTT_{k}"] = v
        #
        # feat["PTT_PairLoRatio"] = pair_lo_ratio
        # feat["PTT_PairHiRatio"] = pair_hi_ratio
        # --------------------------------------------------------

        return pd.DataFrame([feat])

    # ------------------------------------------------------------
    # input check
    # ------------------------------------------------------------
    if ecg_dict is None or ppg_dict is None:
        return {}

    if preprocess:
        ecg = preprocess_ecg(ecg_dict, filter_type=filter_type, fix_polarity=True)
        ppg = preprocess_ppg(ppg_dict, filter_type=filter_type)
    else:
        ecg = ecg_dict
        ppg = ppg_dict

    if "timestamp" not in ecg or "timestamp" not in ppg or "ppg" not in ppg:
        return {}

    out = {}

    debug_root = Path(debug_plot_dir) if debug_plot_dir is not None else None

    # ------------------------------------------------------------
    # fused mode
    # ------------------------------------------------------------
    if use_peak_fusion:
        try:
            ecg_time_ms = ecg["timestamp"]
            ppg_time_ms = ppg["timestamp"]

            df_ppg_raw = pd.DataFrame({
                "Timestamp": pd.to_datetime(ppg_time_ms, unit="ms"),
                "PPG": np.asarray(ppg["ppg"], dtype=float),
            }).set_index("Timestamp").sort_index()

            df_ecg_base = pd.DataFrame({
                "Timestamp": pd.to_datetime(ecg_time_ms, unit="ms"),
            }).set_index("Timestamp").sort_index()

            df_synced = pd.concat([df_ecg_base, df_ppg_raw], axis=1).sort_index()
            if resample_method == "cubic":
                df_synced["PPG"] = df_synced["PPG"].interpolate(method="cubic")
            else:
                df_synced["PPG"] = df_synced["PPG"].interpolate(method="time")

            df_synced = df_synced.loc[df_ecg_base.index].dropna(subset=["PPG"])
            if df_synced.empty:
                return {}

            valid_mask = df_ecg_base.index.isin(df_synced.index)

            dt = df_synced.index.to_series().diff().median().total_seconds()
            fs = 1.0 / dt if dt and dt > 0 else np.nan
            if not np.isfinite(fs) or fs <= 0:
                return {}

            # Replace only part of the fused-mode internals below
            per_lead_info = {}
            sig_len = len(df_synced)

            for ch_name, sig in ecg.items():
                if ch_name == "timestamp":
                    continue

                ch_sig = np.asarray(sig, dtype=float)[valid_mask]

                try:
                    _, info = nk.ecg_peaks(
                        ch_sig,
                        sampling_rate=fs,
                        correct_artifacts=correct_artifacts_per_lead,
                    )
                    per_lead_info[ch_name] = info
                except Exception:
                    per_lead_info[ch_name] = None

            _, fused_info = _fuse_multilead_rpeaks_from_info(
                info_dict=per_lead_info,
                signal_length=sig_len,
                fs_ecg=fs,
                cluster_ms=fusion_cluster_ms,
                min_agree_leads=fusion_min_agree_leads,
                outlier_ms=fusion_outlier_ms,
            )

            r_idx = np.asarray(fused_info.get("ECG_R_Peaks", []), dtype=int)

            fused_artifacts = {}
            if post_fix_fused_peaks and len(r_idx) >= 3:
                r_idx, fused_artifacts = _post_fix_fused_rpeaks(
                    r_idx,
                    fs_ecg=fs,
                    method=fused_fix_method,
                    iterative=True,
                    show=False,
                )

            _, ppg_info = nk.ppg_peaks(
                df_synced["PPG"].to_numpy(),
                sampling_rate=fs,
                method=ppg_method,
                correct_artifacts=True,
            )
            ppk_idx = np.asarray(ppg_info.get("PPG_Peaks", []), dtype=int)

            ppg_artifacts = {}
            if post_fix_ppg_peaks and len(ppk_idx) >= 3:
                ppk_idx, ppg_artifacts = _post_fix_peak_indices(
                    ppk_idx,
                    fs=fs,
                    method=ppg_fix_method,
                    iterative=True,
                    show=False,
                )

            tsec = df_synced.index.astype("int64").to_numpy() / 1e9
            r_t = tsec[r_idx] if r_idx.size > 0 else np.array([], dtype=float)
            ppk_t = tsec[ppk_idx] if ppk_idx.size > 0 else np.array([], dtype=float)

            ptt_ms, meta, debug_df = _pair_r_to_unique_ppg_peak_by_rr_window(
                r_t=r_t,
                ppk_t=ppk_t,
                lo_ratio=pair_lo_ratio,
                hi_ratio=pair_hi_ratio,
                rr_ref_method=rr_ref_method,
            )

            out["fused"] = _build_feature_df(ptt_ms, meta)

            if return_debug_df:
                debug_df = debug_df.copy()
                if len(debug_df) > 0:
                    debug_df["ecg_postfix_applied"] = bool(post_fix_fused_peaks)
                    debug_df["ppg_postfix_applied"] = bool(post_fix_ppg_peaks)
                out["fused_debug_df"] = debug_df

            out["fused_artifacts"] = fused_artifacts
            out["fused_ppg_artifacts"] = ppg_artifacts

            if debug_root is not None:
                stem = f"{plot_prefix}_fused"
                if save_debug_plots:
                    _save_ptt_debug_plots(ptt_ms, debug_df, debug_root, stem)
                if save_debug_csv:
                    debug_df.to_csv(debug_root / f"{stem}_beat_level.csv", index=False)

        except Exception as e:
            print(f"Error during fused PTT processing: {e}")
            return {}

    # ------------------------------------------------------------
    # per-lead mode
    # ------------------------------------------------------------
    else:
        df_ppg = pd.DataFrame({
            "Timestamp": pd.to_datetime(ppg["timestamp"], unit="ms"),
            "PPG": np.asarray(ppg["ppg"], dtype=float),
        }).set_index("Timestamp").sort_index()

        for ch_name, sig in ecg.items():
            if ch_name == "timestamp":
                continue

            try:
                df_ecg = pd.DataFrame({
                    "Timestamp": pd.to_datetime(ecg["timestamp"], unit="ms"),
                    "ECG": np.asarray(sig, dtype=float),
                }).set_index("Timestamp").sort_index()

                df = pd.concat([df_ecg["ECG"], df_ppg["PPG"]], axis=1).sort_index()

                if resample_method == "cubic":
                    df["PPG"] = df["PPG"].interpolate(method="cubic")
                else:
                    df["PPG"] = df["PPG"].interpolate(method="time")

                df = df.dropna(subset=["ECG", "PPG"])
                if df.empty:
                    continue

                dt = df.index.to_series().diff().median().total_seconds()
                fs = 1.0 / dt if dt and dt > 0 else np.nan
                if not np.isfinite(fs) or fs <= 0:
                    continue

                _, ecg_info = nk.ecg_peaks(
                    df["ECG"].to_numpy(),
                    sampling_rate=fs,
                    correct_artifacts=correct_artifacts_per_lead,
                )
                r_idx = np.asarray(ecg_info.get("ECG_R_Peaks", []), dtype=int)

                _, ppg_info = nk.ppg_peaks(
                    df["PPG"].to_numpy(),
                    sampling_rate=fs,
                    method=ppg_method,
                    correct_artifacts=True,
                )
                ppk_idx = np.asarray(ppg_info.get("PPG_Peaks", []), dtype=int)

                ppg_artifacts = {}
                if post_fix_ppg_peaks and len(ppk_idx) >= 3:
                    ppk_idx, ppg_artifacts = _post_fix_peak_indices(
                        ppk_idx,
                        fs=fs,
                        method=ppg_fix_method,
                        iterative=True,
                        show=False,
                    )

                tsec = df.index.astype("int64").to_numpy() / 1e9
                r_t = tsec[r_idx] if r_idx.size > 0 else np.array([], dtype=float)
                ppk_t = tsec[ppk_idx] if ppk_idx.size > 0 else np.array([], dtype=float)

                ptt_ms, meta, debug_df = _pair_r_to_unique_ppg_peak_by_rr_window(
                    r_t=r_t,
                    ppk_t=ppk_t,
                    lo_ratio=pair_lo_ratio,
                    hi_ratio=pair_hi_ratio,
                    rr_ref_method=rr_ref_method,
                )

                out[ch_name] = _build_feature_df(ptt_ms, meta)
                
                if return_debug_df:
                    out[f"{ch_name}_debug_df"] = debug_df

                if debug_root is not None:
                    stem = f"{plot_prefix}_{ch_name}"
                    if save_debug_plots:
                        _save_ptt_debug_plots(ptt_ms, debug_df, debug_root, stem)
                    if save_debug_csv:
                        debug_df.to_csv(debug_root / f"{stem}_beat_level.csv", index=False)

            except Exception as e:
                print(f"Error during per-lead PTT processing ({ch_name}): {e}")
                continue

    return out

# ---------------------------------------------------------------------
# GSR/EDA Feature Extractor (robust + biomarker-friendly)
# - tonic: mean/std/min/max/range/median/iqr/slope
# - phasic: mean/std/auc(abs)/rms
# - SCR peaks: count, rate(/min), amplitude(mean/std/max), IPI(mean/std)
# - QC: nan ratio, flatline ratio, outlier ratio, raw stats
# ---------------------------------------------------------------------

def get_GSR_features(
    gsr_dict: dict,
    fs_gsr: float = FS_GSR,
    preprocess: bool = True,
    min_valid_ratio: float = 0.7,     # if too many NaNs -> return None
    min_len_sec: float = 10.0,        # too short trial -> return None
    flatline_eps: float = 1e-6,
):
    """
    Args:
        gsr_dict: {'timestamp': np.ndarray, 'gsr': np.ndarray}
        fs_gsr: sampling rate (Hz)
        preprocess: if True, apply preprocess_gsr(gsr_dict) externally defined
        min_valid_ratio: minimum fraction of finite samples required
        min_len_sec: minimum signal length (seconds)
        flatline_eps: epsilon for flatline detection on derivative

    Returns:
        dict | None
    """
        
    def _safe_float(x):
        """float(x), returning nan instead of raising on failure."""
        try:
            return float(x)
        except Exception:
            return np.nan


    def _nan_ratio(x: np.ndarray) -> float:
        """Fraction of non-finite (NaN/inf) values in x."""
        x = np.asarray(x, dtype=float)
        if x.size == 0:
            return np.nan
        return float(np.mean(~np.isfinite(x)))


    def _flatline_ratio(x: np.ndarray, fs: float, eps: float = 1e-6) -> float:
        """
        Fraction of samples where |dx/dt| is near zero -> flatline-like behavior.
        """
        x = np.asarray(x, dtype=float)
        m = np.isfinite(x)
        if m.sum() < 3:
            return np.nan
        xx = x[m]
        dx = np.diff(xx) * fs
        if dx.size == 0:
            return np.nan
        return float(np.mean(np.abs(dx) <= eps))


    def _outlier_ratio_robust(x: np.ndarray, z_th: float = 6.0) -> float:
        """
        Robust z-score based outlier fraction using MAD.
        """
        x = np.asarray(x, dtype=float)
        m = np.isfinite(x)
        if m.sum() < 5:
            return np.nan
        xx = x[m]
        med = np.median(xx)
        mad = np.median(np.abs(xx - med))
        if mad <= 0:
            return 0.0
        rz = 0.6745 * (xx - med) / mad
        return float(np.mean(np.abs(rz) > z_th))


    def _slope(x: np.ndarray, fs: float) -> float:
        """
        Linear slope vs time (units per second).
        """
        x = np.asarray(x, dtype=float)
        m = np.isfinite(x)
        if m.sum() < 3:
            return np.nan
        xx = x[m]
        t = np.arange(xx.size, dtype=float) / float(fs)
        # polyfit is fine here; robust regression is overkill for short trials
        return float(np.polyfit(t, xx, 1)[0])


    def _stats_basic(x: np.ndarray, prefix: str, out: dict):
        """Write mean/SD/min/max/range/median/IQR of x into out under f"{prefix}_*" keys (in place)."""
        x = np.asarray(x, dtype=float)
        m = np.isfinite(x)
        if m.sum() == 0:
            out[f"{prefix}_Mean"] = np.nan
            out[f"{prefix}_SD"] = np.nan
            out[f"{prefix}_Min"] = np.nan
            out[f"{prefix}_Max"] = np.nan
            out[f"{prefix}_Range"] = np.nan
            out[f"{prefix}_Median"] = np.nan
            out[f"{prefix}_IQR"] = np.nan
            return

        xx = x[m]
        out[f"{prefix}_Mean"] = float(np.mean(xx))
        out[f"{prefix}_SD"] = float(np.std(xx))
        out[f"{prefix}_Min"] = float(np.min(xx))
        out[f"{prefix}_Max"] = float(np.max(xx))
        out[f"{prefix}_Range"] = float(np.max(xx) - np.min(xx))
        out[f"{prefix}_Median"] = float(np.median(xx))
        q25, q75 = np.percentile(xx, [25, 75])
        out[f"{prefix}_IQR"] = float(q75 - q25)
        
    if gsr_dict is None or "gsr" not in gsr_dict:
        return None

    if preprocess:
        preprocessed_dict = preprocess_gsr(gsr_dict)  # must exist in your codebase
    else:
        preprocessed_dict = gsr_dict

    sig = np.asarray(preprocessed_dict.get("gsr", []), dtype=float)
    if sig.size == 0:
        return None

    # length check
    if sig.size < int(min_len_sec * fs_gsr):
        return None

    # validity check
    valid_ratio = float(np.mean(np.isfinite(sig)))
    if valid_ratio < min_valid_ratio:
        return None

    out = {}

    # -------------------------
    # QC / Raw stats
    # -------------------------
    # out["EDA_Raw_NaN_Ratio"] = _nan_ratio(sig)
    # out["EDA_Raw_Valid_Ratio"] = valid_ratio
    # out["EDA_Raw_Flatline_Ratio"] = _flatline_ratio(sig, fs_gsr, eps=flatline_eps)
    # out["EDA_Raw_Outlier_Ratio"] = _outlier_ratio_robust(sig, z_th=6.0)
    _stats_basic(sig, "EDA_Raw", out)

    # sanitize
    try:
        eda_signal = nk.signal_sanitize(sig)
    except Exception:
        return None

    # -------------------------
    # Phasic decomposition
    # -------------------------
    try:
        eda_decomp = nk.eda_phasic(eda_signal, sampling_rate=fs_gsr)
    except Exception:
        return None

    # -------------------------
    # Peak detection (on phasic)
    # -------------------------
    try:
        peaks_df, info = nk.eda_peaks(eda_decomp["EDA_Phasic"].values, sampling_rate=fs_gsr)
    except Exception:
        return None

    data = pd.concat([eda_decomp, peaks_df], axis=1)
    cols = set(data.columns)

    # -------------------------
    # Tonic features
    # -------------------------
    if "EDA_Tonic" in cols:
        tonic = data["EDA_Tonic"].to_numpy(dtype=float)
        _stats_basic(tonic, "EDA_Tonic", out)
        out["EDA_Tonic_Slope"] = _slope(tonic, fs_gsr)
    else:
        # keep keys consistent
        for k in ["Mean","SD","Min","Max","Range","Median","IQR","Slope"]:
            out[f"EDA_Tonic_{k}"] = np.nan

    # -------------------------
    # Phasic features
    # -------------------------
    if "EDA_Phasic" in cols:
        phasic = data["EDA_Phasic"].to_numpy(dtype=float)
        _stats_basic(phasic, "EDA_Phasic", out)

        m = np.isfinite(phasic)
        if m.sum() >= 3:
            ph = phasic[m]
            out["EDA_Phasic_AUC_abs"] = float(np.trapz(np.abs(ph), dx=1.0 / fs_gsr))
            out["EDA_Phasic_RMS"] = float(np.sqrt(np.mean(ph**2)))
        else:
            out["EDA_Phasic_AUC_abs"] = np.nan
            out["EDA_Phasic_RMS"] = np.nan
    else:
        for k in ["Mean","SD","Min","Max","Range","Median","IQR","AUC_abs","RMS"]:
            out[f"EDA_Phasic_{k}"] = np.nan

    # -------------------------
    # SCR peak features
    # -------------------------
    if "SCR_Peaks" in cols:
        scr_peaks = data["SCR_Peaks"].to_numpy(dtype=float)
        n_peaks = int(np.nansum(scr_peaks == 1))
        out["SCR_Peaks_N"] = n_peaks

        dur_min = (len(data) / fs_gsr) / 60.0
        out["SCR_Peaks_Rate"] = float(n_peaks / dur_min) if dur_min > 0 else np.nan

        # inter-peak interval
        peak_pos = np.where(scr_peaks == 1)[0]
        if peak_pos.size >= 2:
            ipi = np.diff(peak_pos) / fs_gsr
            out["SCR_IPI_Mean"] = float(np.mean(ipi))
            out["SCR_IPI_SD"] = float(np.std(ipi))
        else:
            out["SCR_IPI_Mean"] = np.nan
            out["SCR_IPI_SD"] = np.nan
    else:
        out["SCR_Peaks_N"] = np.nan
        out["SCR_Peaks_Rate"] = np.nan
        out["SCR_IPI_Mean"] = np.nan
        out["SCR_IPI_SD"] = np.nan

    # amplitudes at peaks
    if ("SCR_Amplitude" in cols) and ("SCR_Peaks" in cols):
        scr_amp = data["SCR_Amplitude"].to_numpy(dtype=float)
        scr_peaks = data["SCR_Peaks"].to_numpy(dtype=float)
        idx = (scr_peaks == 1) & np.isfinite(scr_amp)
        if np.any(idx):
            amps = scr_amp[idx]
            out["SCR_Peaks_Amp_Mean"] = float(np.mean(amps))
            out["SCR_Peaks_Amp_SD"] = float(np.std(amps))
            out["SCR_Peaks_Amp_Max"] = float(np.max(amps))
        else:
            out["SCR_Peaks_Amp_Mean"] = np.nan
            out["SCR_Peaks_Amp_SD"] = np.nan
            out["SCR_Peaks_Amp_Max"] = np.nan
    else:
        out["SCR_Peaks_Amp_Mean"] = np.nan
        out["SCR_Peaks_Amp_SD"] = np.nan
        out["SCR_Peaks_Amp_Max"] = np.nan

    return out
