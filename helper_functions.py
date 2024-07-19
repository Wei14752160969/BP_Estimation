from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pickle
from metrics import *
import seaborn as sns
from scipy.signal import find_peaks

sns.set()
length = 1024


def prepareData(mdl, X_train, X_val, X_test, Y_train, Y_val, Y_test):


    X2_train = []

    X2_val = []

    X2_test = []

    YPs = mdl.predict(X_train)

    for i in tqdm(range(len(X_train))):
        X2_train.append(np.array(YPs[i]))

    YPs = mdl.predict(X_val)

    for i in tqdm(range(len(X_val))):
        X2_val.append(np.array(YPs[i]))

    YPs = mdl.predict(X_test)

    for i in tqdm(range(len(X_test))):
        X2_test.append(np.array(YPs[i]))

    X2_train = np.array(X2_train)

    X2_val = np.array(X2_val)

    X2_test = np.array(X2_test)

    return (X2_train, X2_val, X2_test)


def prepareDataDS(mdl, X):

    X2 = []

    YPs = mdl.predict(X)

    for i in tqdm(range(len(X)), desc='Preparing Data for DS'):
        X2.append(np.array(YPs[0][i]))

    X2 = np.array(X2)

    return X2


def prepareLabel(Y):
    """
    Prepare label for deep supervised pipeline

    Returns:
        dictionary -- dictionary containing the 5 level ground truth outputs of the network
    """

    def approximate(inp, w_len):
        """
        Downsamples using taking mean over window

        Arguments:
            inp {array} -- signal
            w_len {int} -- length of window

        Returns:
            array -- downsampled signal
        """

        op = []

        for i in range(0, len(inp), w_len):
            op.append(np.mean(inp[i:i + w_len]))

        return np.array(op)

    out = {}
    out['out'] = []
    out['level1'] = []
    out['level2'] = []
    out['level3'] = []
    out['level4'] = []

    for y in tqdm(Y, desc='Preparing Label for DS'):
        # computing approximations
        cA1 = approximate(np.array(y).reshape(length), 2)

        cA2 = approximate(np.array(y).reshape(length), 4)

        cA3 = approximate(np.array(y).reshape(length), 8)

        cA4 = approximate(np.array(y).reshape(length), 16)

        # populating the labels for different labels
        out['out'].append(np.array(y.reshape(length, 1)))
        out['level1'].append(np.array(cA1.reshape(length // 2, 1)))
        out['level2'].append(np.array(cA2.reshape(length // 4, 1)))
        out['level3'].append(np.array(cA3.reshape(length // 8, 1)))
        out['level4'].append(np.array(cA4.reshape(length // 16, 1)))

    out['out'] = np.array(out['out'])  # converting to numpy array
    out['level1'] = np.array(out['level1'])
    out['level2'] = np.array(out['level2'])
    out['level3'] = np.array(out['level3'])
    out['level4'] = np.array(out['level4'])

    return out


def detect_anomalies(data, distance_limit=60):
    # Find peaks and valleys
    peaks, _ = find_peaks(data, prominence=1)
    valleys, _ = find_peaks(-data, prominence=1)

    # Merge adjacent peaks/valleys within distance_limit
    merged_peaks = merge_adjacent(peaks, distance_limit)
    merged_valleys = merge_adjacent(valleys, distance_limit)
    value_peaks = data[merged_peaks]
    value_valleys = data[merged_valleys]
    # Calculate mean and standard deviation of the merged peaks/valleys
    mean_peak = np.mean(value_peaks)
    mean_valley = np.mean(value_valleys)
    std_peak = np.std(value_peaks)
    std_valley = np.std(value_valleys)
    peaks_norm = []
    valleys_norm = []
    # Detect anomalies
    for peak in value_peaks:
        if abs(peak - mean_peak) <= std_peak:
            peaks_norm.append(peak)
    for valley in value_valleys:
        if abs(valley - mean_valley) <= std_valley:
            valleys_norm.append(valley)
    # Calculate SBP (Systolic Blood Pressure) and DBP (Diastolic Blood Pressure)
    SBP = np.mean(peaks_norm)
    DBP = np.mean(valleys_norm)

    # Calculate MAP (Mean Arterial Pressure)
    MAP = (SBP + 2 * DBP) / 3
    return SBP, DBP, MAP


def merge_adjacent(points, distance_limit):
    merged_points = []
    i = 0
    while i < len(points):
        start = points[i]
        while i < len(points) - 1 and points[i + 1] - points[i] <= distance_limit:
            i += 1
        end = points[i]
        merged_points.append((start + end) // 2)
        i += 1
    return merged_points


