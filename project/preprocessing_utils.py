"""
Contains the function dedicated to the loading, filtering and shaping of the data.
"""

from scipy.io import loadmat
from scipy.signal import butter, lfilter
import numpy as np


# load train data
def load_data_from_mat(path_to_file, eeg_data='RawEEGData', label_data="Labels"):
    mat_file = loadmat(path_to_file)
    # try except: KeyError
    raw_eeg_data = mat_file[eeg_data]
    labels = mat_file[label_data]

    return raw_eeg_data, labels


# Butterworth Bandpass Filter
# Source: https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(signal, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    n_trials, n_chans, n_time = signal.shape
    filtered = np.zeros(signal.shape)
    for i in range(n_trials):
        for j in range(n_chans):
            filtered[i][j] = lfilter(b, a, signal[i][j])
    return filtered


def prepare_data(raw_data, raw_labels, sample_rate, t_low, t_high=0):

    # trim the data
    if t_high == 0:
        X = raw_data[:, :, int(round(sample_rate*t_low)):]
    else:
        X = raw_data[:, :, int(round(sample_rate*t_low)):int(round(sample_rate*t_high))]

    y = np.squeeze(raw_labels) - 1  # remove useless dimension and change labels to 0/1
    # y = raw_labels.ravel() #  view of the original array

    return X, y