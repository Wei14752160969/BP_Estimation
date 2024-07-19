import h5py
import os
import pickle
import heartpy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
from heartpy.exceptions import BadSignalWarning
from PyEMD import EMD, EEMD  # PyEMD库提供了EMD和EEMD的实现
from scipy.fft import fft, fftfreq
import seaborn as sns
sns.set()


def load_data():
    """
        Loads the data from the mat files
        and processes them to extract the sbp and dbp values
    """

    fs = 125								# sampling frequency
    t = 10									# length of ppg episodes
    dt = 5									# step size of taking the next episode

    samples_in_episode = round(fs * t)		# number of samples in an episode
    d_samples = round(fs * dt)				# number of samples in a step

    try:									# create the processed_data directory
        os.makedirs('processed_data')
    except Exception as e:
        print(e)

    for k in range(1, 5):					# process for the 4 different parts of the data

        print("Processing file part {} out of 4".format(k))

        f = h5py.File(os.path.join('part', 'Part_{}.mat'.format(k)), 'r')		# loads the data

        ky = 'Part_' + str(k)														# key

        for i in tqdm(range(3000), desc='Reading Records'):					# reading the records

            signal = []												# ppg signal
            bp = []													# abp signal

            output_str = '10s,SBP,DBP\n'							# starting text for a new csv file

            for j in tqdm(range(len(f[f[ky][i][0]])), desc='Reading Samples from Record {}/3000'.format(i+1)):	# reading samples from records

                signal.append(f[f[ky][i][0]][j][0])					# ppg signal
                bp.append(f[f[ky][i][0]][j][1])						# abp signal

            # print("bp After: ", len(bp))

            for j in tqdm(range(0, len(f[f[ky][i][0]])-samples_in_episode, d_samples), desc='Processing Episodes from Record {}/3000'.format(i+1)):	# computing the sbp and dbp values
                ppg = signal[j:j+samples_in_episode]
                ppg = np.array(ppg)
                # data cleaning
                with warnings.catch_warnings(record=True) as w:
                    # Place code here that may generate BadSignalWarning
                    try:
                        # Simulate operations that may generate warnings
                        working_data, measures = heartpy.process(ppg, fs, interp_clipping=True, reject_segmentwise=True)
                        pass
                    except BadSignalWarning:
                        print(f"Warning in iteration {i}: BadSignalWarning occurred. Continuing to next iteration.")
                        continue
                    # Check if any BadSignalWarning type warnings are recorded
                    # for warning in w:
                    #     if issubclass(warning.category, BadSignalWarning):
                    #         print(f"Warning in iteration {i}: {warning.message}")
                peaks = working_data['ybeat']
                # Remove removed_feats from peaklist
                left_beats = [x for x in working_data['peaklist'] if x not in working_data['removed_beats']]
                if len(left_beats) < 2:
                    continue
                num_peaklist = len(working_data['peaklist'])
                num_removed = len(working_data['removed_beats'])
                if left_beats:
                    diff = np.diff(left_beats)
                    if len(diff) < 1:
                        continue
                    maxspace = np.max(diff)
                    minspace = np.min(diff)
                    # Maxspace and left.beats [0] and 1250-leftbeats [-1] take the maximum value
                    maxspace = max(maxspace, left_beats[0], samples_in_episode - left_beats[-1])
                else:
                    # Handle situations where left.beats is empty, such as returning None or other default values
                    continue
                if (np.mean(peaks) < 1):
                    continue
                else:
                    if ((num_removed / num_peaklist) > 0.3):
                        continue
                    else:
                        if (maxspace > 200 or minspace < 60):
                            continue
                sbp = max(bp[j:j+samples_in_episode])		# sbp value
                dbp = min(bp[j:j+samples_in_episode])    	# dbp value

                output_str += '{},{},{}\n'.format(j,sbp,dbp)	# append to the csv file

                # print("j: ", j, "sbp: ", sbp, "dbp: ", dbp)

            fp = open(os.path.join('processed_data','Part_{}_{}.csv'.format(k,i)),'w')		# create the csv file
            fp.write(output_str)															# write the csv file
            fp.close()																		# close the csv file


def observe_processed_data():
    """
        Observe the sbp and dbps of the 10s long episodes
    """

    files = next(os.walk('D:/dataset1/processed_data'))[2]

    sbps = []
    dbps = []

    for fl in tqdm(files,desc='Browsing through Files'):

        lines = open(os.path.join('processed_data',fl),'r').read().split('\n')[1:-1]

        for line in tqdm(lines,desc='Browsing through Episodes from File'):

            values = line.split(',')

            sbp = int(float(values[1]))
            dbp = int(float(values[2]))

            sbps.append(sbp)
            dbps.append(dbp)


    plt.subplot(2,1,1)
    plt.hist(sbps,bins=180)
    plt.title('SBP')

    plt.subplot(2,1,2)
    plt.hist(dbps,bins=180)
    plt.title('DBP')

    plt.show()


def downsample_data(minThresh=2500, ratio=0.25):
    """
    Downsamples the data based on the scheme proposed in the manuscript
    """


    files = next(os.walk('processed_data'))[2]		# load all csv files

    sbps_dict = {}									# dictionary to store sbp and dbp values
    dbps_dict = {}

    sbps_cnt = {}									# dictionary containing count of specific sbp and dbp values
    dbps_cnt = {}

    dbps_taken = {}									# dictionary containing count of specific sbp and dbp taken
    sbps_taken = {}

    sbps = []										# list of sbps and dbps
    dbps = []

    candidates = []									# list of candidate episodes

    lut = {}										# look up table

    for fl in tqdm(files, desc='Browsing Files'):		# iterating over the csv files

        lines = open(os.path.join('processed_data', fl), 'r').read().split('\n')[1:-1]	# fetching the episodes

        for line in tqdm(lines, desc='Reading Episodes'):		# iterating over the episodes

            values = line.split(',')

            file_no = int(fl.split('_')[1])							# id of the file
            record_no = int(fl.split('.')[0].split('_')[2])			# id of the record
            episode_st = int(values[0])								# start of the episode
            sbp = int(float(values[1]))								# sbp of that episode
            dbp = int(float(values[2]))								# dbp of that episode

            if(sbp not in sbps_dict):			# new sbp found

                sbps_dict[sbp] = []				# initialize
                sbps_cnt[sbp] = 0

            sbps_dict[sbp].append((file_no, record_no, episode_st))		# add the file, record and episode info
            sbps_cnt[sbp] += 1											# increment

            if(dbp not in dbps_dict):			# new dbp found

                dbps_dict[dbp] = []				# initialize
                dbps_cnt[dbp] = 0

            dbps_dict[dbp].append((file_no, record_no, episode_st, sbp))	# add the file, record and episode info
            dbps_cnt[dbp] += 1												# increment

    sbp_keys = list(sbps_dict)				# all the different sbp values
    dbp_keys = list(dbps_dict)				# all the different dbp values

    sbp_keys.sort()					# sorting the sbp values
    dbp_keys.sort()					# sorting the dbp values

    for dbp in tqdm(dbp_keys, desc='DBP Binning'):		# iterating through the dbp values

        cnt = min(int(dbps_cnt[dbp]*ratio), minThresh)		# how many episodes of this dbp to take

        for i in tqdm(range(cnt), desc='Picking Random Indices'):

            indix = np.random.randint(len(dbps_dict[dbp]))		# picking a random index

            candidates.append([dbps_dict[dbp][indix][0], dbps_dict[dbp][indix][1], dbps_dict[dbp][indix][2]])	# add the file, record and episode info in the candidates list

            if(dbp not in dbps_taken):					# this dbp has not been taken
                dbps_taken[dbp] = 0						# initialize

            dbps_taken[dbp] += 1						# increment

            if(dbps_dict[dbp][indix][3] not in sbps_taken):		# checking if the sbp of that episode has been taken or not
                sbps_taken[dbps_dict[dbp][indix][3]] = 0		# initialize

            sbps_taken[dbps_dict[dbp][indix][3]] += 1			# increment

            if(dbps_dict[dbp][indix][0] not in lut):			# this file is not in look up table

                lut[dbps_dict[dbp][indix][0]] = {}				# add the file in look up table

            if(dbps_dict[dbp][indix][1] not in lut[dbps_dict[dbp][indix][0]]):	# this record is not in look up table

                lut[dbps_dict[dbp][indix][0]][dbps_dict[dbp][indix][1]] = {}	# add the record in look up table

            if(dbps_dict[dbp][indix][2] not in lut[dbps_dict[dbp][indix][0]][dbps_dict[dbp][indix][1]]):	# this episode is not in look up table

                lut[dbps_dict[dbp][indix][0]][dbps_dict[dbp][indix][1]][dbps_dict[dbp][indix][2]] = 1		# add this episode in look up table

            dbps_dict[dbp].pop(indix)		# remove this episode, so that this episode is not randomly selected again

    for sbp in tqdm(sbp_keys, desc='SBP Binning'):		# iterating on the sbps

        if sbp not in sbps_taken:			# this sbp has not yet been taken
            sbps_taken[sbp] = 0				# initialize

        cnt = min(int(sbps_cnt[sbp]*ratio), minThresh) - sbps_taken[sbp]		# how many episodes of this sbp to take, removed the count already included during dbp based binning

        for i in tqdm(range(cnt), desc='Picking Random Indices'):		# iterate over how many episodes to take

            while len(sbps_dict[sbp]) > 0:					# while there are some episodes with that sbp left

                try:
                    indix = np.random.randint(len(sbps_dict[sbp]))		# picking a random episode
                except:
                    pass

                try:								# see if that episode is contained in the look up table
                    dumi = lut[sbps_dict[sbp][indix][0]][sbps_dict[sbp][indix][1]][sbps_dict[sbp][indix][2]]
                except:
                    sbps_dict[sbp].pop(indix)
                    continue

                candidates.append([sbps_dict[sbp][indix][0], sbps_dict[sbp][indix][1], sbps_dict[sbp][indix][2]])	# add new candidate

                sbps_taken[sbp] += 1								# increment

                sbps_dict[sbp].pop(indix)							# remove that episode

                break												# repeat the process

    sbps_dict = {}			# garbage collection
    dbps_dict = {}

    sbps_cnt = {}			# garbage collection
    dbps_cnt = {}

    sbps = []				# garbage collection
    dbps = []

    lut = {}				# garbage collection

    print('Total {} episodes have been selected'.format(len(candidates)))

    pickle.dump(candidates, open('candidates.p', 'wb'))		# save the candidates

    '''
        plotting the downsampled episodes
    '''

    sbp_keys = list(sbps_taken)
    dbp_keys = list(dbps_taken)

    sbp_keys.sort()
    dbp_keys.sort()

    for sbp in sbp_keys:
        sbps.append(sbps_taken[sbp])

    for dbp in dbp_keys:
        dbps.append(dbps_taken[dbp])

    plt.figure()

    plt.subplot(2, 1, 1)
    plt.bar(sbp_keys, sbps)
    plt.title('SBP')

    plt.subplot(2, 1, 2)
    plt.bar(dbp_keys, dbps)
    plt.title('DBP')

    plt.show()



def extract_episodes(candidates):
    """
        Extracts the episodes from the raw data
    """

    try:								# making the necessary directories
        os.makedirs('ppgs')
    except Exception as e:
        print(e)

    try:
        os.makedirs('abps')
    except Exception as e:
        print(e)

    for k in tqdm(range(1,5), desc='Reading from Files'):				# iterating throug the files

        f = h5py.File('part/Part_{}.mat'.format(k), 'r')

        fs = 125																# sampling frequency
        t = 10																	# length of ppg episodes
        samples_in_episode = round(fs * t)										# number of samples in an episode
        ky = 'Part_' + str(k)													# key

        for indix in tqdm(range(len(candidates)), desc='Reading from File {}/4'.format(k)):		# iterating through the candidates

            if(candidates[indix][0] != k):					# this candidate is from a different file
                continue

            record_no = int(candidates[indix][1])			# record no of the episode
            episode_st = int(candidates[indix][2])			# start of that episode

            ppg = []										# ppg signal
            abp = []										# abp signal

            for j in tqdm(range(episode_st, episode_st+samples_in_episode), desc='Reading Episode Id {}'.format(indix)):

                ppg.append(f[f[ky][record_no][0]][j][0])	# ppg signal
                abp.append(f[f[ky][record_no][0]][j][1])	# abp signal

            pickle.dump(np.array(ppg), open(os.path.join('ppgs', '{}.p'.format(indix)), 'wb'))		# saving the ppg signal
            pickle.dump(np.array(abp), open(os.path.join('abps', '{}.p'.format(indix)), 'wb'))		# saving the abp signal


def merge_episodes():
    """5056065059096509696+5
        Merges the extracted episodes
        and saves them as a hdf5 file
    """

    try:									# creates the necessary directory
        os.makedirs('data')
    except Exception as e:
        print(e)

    files = next(os.walk('abps'))[2]				# all the extracted episodes

    np.random.shuffle(files)						# random shuffling, we perform the random shuffling now
                                                    # so that we can split the data straightforwardly next step

    data = []										# initialize

    for fl in tqdm(files):

        abp = pickle.load(open(os.path.join('abps',fl),'rb'))			# abp signal
        ppg = pickle.load(open(os.path.join('ppgs',fl),'rb'))			# ppg signal

        data.append([abp, ppg])											# adding the signals



    f = h5py.File(os.path.join('data','data.hdf5'), 'w')				# saving the data as hdf5 file
    dset = f.create_dataset('data', data=data)

def process_data():
    """
        Detecting outliers, denoising, removing baseline drift, and eliminating phase differences
    """
    fl = h5py.File(os.path.join('data', 'data.hdf5'), 'r')  # load the episode data
    data = []
    for i in tqdm(range(len(fl['data']))):
        abp = np.array(fl['data'][i][0])  		# abp signal
        ppg = np.array(fl['data'][0][1])		# ppg signal
        # Detecting outliers
        ppg = heartpy.hampel_filter(ppg, filtsize=6)
        # denoising, removing baseline drift with EMD
        emd = EMD()
        imfs = emd.emd(ppg)

        # Perform FFT on each IMF
        fft_results = []
        for i, imf in enumerate(imfs):
            # Calculate FFT
            n = len(imf)
            fft_imf = fft(imf)
            fft_freq = fftfreq(n, d=1 / 125)

            # Obtain the amplitude of FFT
            fft_magnitude = np.abs(fft_imf)

            # Select the positive frequency section
            fft_magnitude = fft_magnitude[:n // 2]
            fft_freq = fft_freq[:n // 2]

            # Store results
            fft_results.append((fft_freq, fft_magnitude))

        # Remove artifacts IMFs
        clean_imfs = imfs[:2]
        # Reconstruct signal
        ppg = np.sum(clean_imfs, axis=0)

        # ppg = np.transpose(ppg)
        ppg = heartpy.filter_signal(ppg, [0.7, 8], sample_rate=125, order=4, filtertype='bandpass')
        data.append([abp, ppg])  # adding the signals
    f = h5py.File(os.path.join('data', 'data_process.hdf5'), 'w')				# saving the data as hdf5 file
    dset = f.create_dataset('data', data=data)


def eliphase():
    fl = h5py.File(os.path.join('data', 'data_process.hdf5'), 'r')  # load the episode data
    data = []
    for i in tqdm(range(len(fl['data']))):
        abp = np.array(fl['data'][i][0])  		# abp signal
        ppg = np.array(fl['data'][0][1])		# ppg signal
        # Calculate cross-correlation function g(Δt)
        cross_corr = np.correlate(ppg, abp, mode='full')
        # Find the lag with maximum correlation
        max_corr_index = np.argmax(cross_corr)
        # Check if the phase difference is greater than or equal to -226
        if abs(-1250 + max_corr_index) >= 226:
            continue
        # Adjust PPG and ABP segments based on phase difference
        if -1250 + max_corr_index > 0:
            ppg = ppg[-1250 + max_corr_index: -1250 + 1024 + max_corr_index]
            abp = abp[0:1024]
        else:
            ppg = ppg[0:1024]
            abp = abp[abs(-1250 + max_corr_index): 1024 + abs(-1250 + max_corr_index)]
        data.append([abp, ppg])  # adding the signals
    f = h5py.File(os.path.join('data', 'data_phase.hdf5'), 'w')				# saving the data as hdf5 file
    dset = f.create_dataset('data', data=data)


def main():
    load_data()
    observe_processed_data()
    downsample_data()
    candidates = pickle.load(open('candidates.p', 'rb'))
    extract_episodes(candidates)
    merge_episodes()
    process_data()
    eliphase()

if __name__ == '__main__':
    main()
