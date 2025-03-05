import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from itertools import groupby
from scipy.signal import find_peaks
import math
from sklearn.preprocessing import MinMaxScaler
import scipy.signal
from statsmodels.tsa.stattools import acf
from sklearn.decomposition import PCA


def set_seed(seed=1201):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sliding_windows(data, window_size, stride=1):
    windows = sliding_window_view(data, window_shape=(window_size,), axis=0)[::stride]
    windows = np.moveaxis(windows, -1, 1)
    return windows


def upsampling(data, ratio, axis):
    data_upsampled = scipy.signal.resample(x=data, num=data.shape[0]*ratio, axis=axis)
    return data_upsampled


def savgol_filter(data):
	return  scipy.signal.savgol_filter(data, window_length = 31, polyorder = 3, deriv=0, delta=1.0, axis= 0, mode='interp', cval=0.0)


def normalizer(range, data):
    return MinMaxScaler(feature_range=range).fit(data)


def preprocessing(signal, scaler=None):
    signal = upsampling(signal, ratio=4, axis=0)
    original = signal
    signal = savgol_filter(signal)
    
    if scaler is None:
        scaler = normalizer((-1,1), signal)
    signal = scaler.transform(signal)
         
    return signal, original, scaler


def get_auto_correlation(data, window_length):
    """
    1. Reduce data from N-channels to 1-channel via PCA (n_components=1).
    2. Compute auto-correlation for each sliding window (centered on index i),
       moving 'stride' steps at a time.

    Parameters
    ----------
    data : array-like, shape (n_samples, n_channels)
        The multi-channel time-series data.
    window_length : int
        Number of samples in each sliding window.
    stride : int
        Step size when moving from one window to the next.

    Returns
    -------
    List[np.ndarray]
        A list of 1D arrays, each containing the ACF of the windowed signal.
    """

    pca = PCA(n_components=1)
    data_1d = pca.fit_transform(data).ravel()

    half_span = window_length // 2
    offset = half_span + (window_length % 2)

    start_index = half_span
    end_index = len(data_1d) - offset

    acf_list = []
    for i in range(start_index, end_index + 1, 50):
        window = data_1d[i - half_span : i + offset]
        window_acf = acf(window, nlags=len(window))
        acf_list.append(window_acf)

    return acf_list 


def get_rep_duration(data, num_phase):
    """
    Estimate the length of a repeating pattern in 'data' by:
      1. Computing auto-correlation windows of the (PCA-reduced) signal.
      2. For each ACF window, finding the first sign change (→ zero-cross).
      3. Collecting peak indices in the ACF beyond that zero-cross.
      4. Using the 75th–84th percentiles of all peak positions.
      5. Returning the integer mean of those percentiles.

    Parameters
    ----------
    data : array-like
        Input multi-channel time series.
    num_phase : int
        Number of phase.

    Returns
    -------
    int
        Estimated repeating pattern length (rep_duration).
    """
    window_length = int(data.shape[0]/num_phase)
    ac_list = get_auto_correlation(data, window_length=window_length)

    all_peak_positions = []

    for acf_window in ac_list:
        sign_changes = np.where(acf_window[:-1] * acf_window[1:] < 0)[0]
        if sign_changes.size == 0:
            continue

        zero_cross_idx = sign_changes[0] + 1
        truncated_acf = acf_window[zero_cross_idx:]
        peak_indices, _ = find_peaks(truncated_acf, 0.15)
        for pk in peak_indices:
            all_peak_positions.append(pk + zero_cross_idx)

    if not all_peak_positions:
        return 0

    percentiles = np.percentile(all_peak_positions, np.arange(75, 85))
    rep_duration = int(np.mean(percentiles))
    return rep_duration



def get_background_window(rep_duration, num_phase, data_length, window_size):
    background_duration = abs(rep_duration * num_phase - data_length)
    
    if background_duration == 0:
        return 0

    ratio = window_size / background_duration
    
    if ratio > 1:
        return 0
    elif ratio > 0.1:
        return 1
    return 2


def ctc_decode(seq):
    """
    Return a run-length-encoded list of (value, count) for consecutive repeats in seq.
    Example:
        run_length_encode([1,1,2,2,2,3]) -> [(1, 2), (2, 3), (3, 1)]
    """
    return [(val, sum(1 for _ in group)) for val, group in groupby(seq)]


def get_count(rle_seq, num_phases):
    """
    A heuristic function that counts how many times the sequence [1, 2, ..., num_phases]
    appears (or partially appears) in rle_seq, with a special 'diff' logic when num_phases >= 4.
    
    Args:
        rle_seq (list of (part, count)): A run-length encoded list of (value, count)
                                         as produced by run_length_encode().
        num_phases (int): Maximum phase number to consider.

    Returns:
        int: The count (res) of how many times the subsequence is detected.
    """
    diff = 1 if num_phases >= 4 else 0
    threshold = math.ceil(num_phases - diff)

    decoded = [part for part, _ in rle_seq if 1 <= part <= num_phases]
    decoded.append(-1)

    res = 0
    streak = 0
    prev = -1

    for val in decoded:
        if val < prev:
            if streak >= threshold:
                res += 1
            streak = 1
        else:
            streak += 1
        prev = val

    return res
     