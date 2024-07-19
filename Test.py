import h5py
import os
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
from scipy.fft import fft, fftfreq
import seaborn as sns
sns.set()



candidates_test = pickle.load(open('candidates_test.p', 'rb'))
data_test = []  # initialize
for k in tqdm(range(1, 5), desc='Reading from Files'):  # iterating throug the files

    f = h5py.File('part/Part_{}.mat'.format(k), 'r')

    fs = 125  # sampling frequency
    t = 10  # length of ppg episodes
    samples_in_episode = round(fs * t)  # number of samples in an episode
    ky = 'Part_' + str(k)  # key

    for indix in tqdm(range(len(candidates_test)),
                      desc='Reading from File {}/4'.format(k)):  # iterating through the candidates

        if (candidates_test[indix][0] != k):  # this candidate is from a different file
            continue

        record_no = int(candidates_test[indix][1])  # record no of the episode
        episode_st = int(candidates_test[indix][2])  # start of that episode

        ppg = []  # ppg signal
        abp = []  # abp signal

        for j in tqdm(range(episode_st, episode_st + samples_in_episode), desc='Reading Episode Id {}'.format(indix)):
            ppg.append(f[f[ky][record_no][0]][j][0])  # ppg signal
            abp.append(f[f[ky][record_no][0]][j][1])  # abp signal
        ppg = np.array(ppg)
        abp = np.array(abp)
        data_test.append([abp, ppg])  # adding the signals
f = h5py.File(os.path.join('datatest', 'data_test.hdf5'), 'w')  # saving the data as hdf5 file
dset = f.create_dataset('data_test', data=data_test)