from lab1_proto import *
import lab1_tools as tools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.cluster import hierarchy
from random import randint

# Load data (utterances of digits) and example
data = np.load('Labs/Lab1/lab1_data.npz', allow_pickle=True)['data']
example = np.load('Labs/Lab1/lab1_example.npz', allow_pickle=True)['example'].item()

use_seven = True

Q4 = False
Q5 = False
Q6 = True
Q7 = False

# list_seven = [data[16], data[17], data[38], data[39]]
# data_seven_2 = np.array((data[16], data[17], data[38], data[39]))
# data = np.array(list_seven)

# # Plot the speech sample
# plt.plot(example['samples'])
# plt.title('Speech samples')
# plt.show()

if Q4:

# 4: Mel Frequency Cepstrum Coefficients

    # 4.1: Enframe
    print("Enframing the speech samples...")
    # Figuring out the frame's window length (number of samples per frame)
    # and window shift (number of overlapping samples in consecutive frames)
    # from the signal's sampling rate.
    num_samples = int(example['samplingrate']/1000) # Number of samples per 1ms
    winlen = num_samples * 20 # Window length of 20ms ~ number of samples in 20ms
    winshift = num_samples * 10 # Window shift in 10ms
    frames = enframe(example['samples'], winlen, winshift)
    if compare(frames, example['frames']): print("The result matches the example.")
    else: print("The result doesn't match the example.")

    # 4.2: Pre-emphasis filter
    print("Applying pre-emphasis filter...")
    pre_emphasis = preemp(frames, 0.97)
    if compare(pre_emphasis, example['preemph']): print("The result matches the example.")
    else: print("The result doesn't match the example.")

    # 4.3: Hamming window
    print("Applying hamming window...")
    hamming_window = windowing(pre_emphasis)
    if compare(hamming_window, example['windowed']): print("The result matches the example.")
    else: print("The result doesn't match the example.")

    # 4.4: Fast Fourier Transfrom
    print("Applying Fast Fourier Transfrom...")
    FFT = powerSpectrum(hamming_window, 512)
    print("Plotting the resulting power spectogram...")
    plt.pcolormesh(FFT)
    plt.title("Power Spectogram")
    plt.show()
    if compare(FFT, example['spec']): print("The result matches the example.")
    else: print("The result doesn't match the example.")

    # According to the Sampling Theorem, f_max is the largest frequency
    # in the signal, and 2 * f_max is the minimum sampling rate (samples
    # per second) for the signal.

    # 4.5: Mel filterbank log spectrum
    print("Applying Mel filterbank log spectrum...")
    MSPEC = logMelSpectrum(FFT, 20000)
    if compare(MSPEC, example['mspec']): print("The result matches the example.")
    else: print("The result doesn't match the example.")

    # 4.6: Cosine transform and liftering
    print("Applying cosine transform...")
    MFCC = cepstrum(MSPEC, 13)
    if compare(MFCC, example['mfcc']): print("The result matches the example.")
    else: print("The result doesn't match the example.")

    print("Applying liftering...")
    LMFCC = tools.lifter(MFCC)
    if compare(LMFCC, example['lmfcc']): print("The result matches the example.")
    else: print("The result doesn't match the example.")

    # Apply to data
    # First step
    mfcc_features = mfcc(data[0]['samples']) # liftered
    mspec_features = mspec(data[0]['samples'])

    # Compute the mspec and mfcc for all utterances and stack them
    for i in range(1, len(data)):
        mfcc_features = np.vstack((mfcc_features, mfcc(data[i]['samples']))) # liftered
        mspec_features = np.vstack((mspec_features, mspec(data[i]['samples'])))

if Q5:

    # 5: Feature Correlation
    mfcc_correlation = np.corrcoef(mfcc_features.T)
    plt.title("Correlation of liftered MFCC features")
    plt.pcolormesh(mfcc_correlation)
    plt.show()

    mspec_correlation = np.corrcoef(mspec_features.T)
    plt.title("Correlation of MSPEC features")
    plt.pcolormesh(mspec_correlation)
    plt.show()

if Q6:
# 6: Speech Segments with Clustering
# Fit a Gaussian Mixture Model to the data
    
    mfcc_features = mfcc(data[0]['samples']) # liftered
    mspec_features = mspec(data[0]['samples'])

    # Compute the mspec and mfcc for all utterances and stack them
    for i in range(1, len(data)):
        mfcc_features = np.vstack((mfcc_features, mfcc(data[i]['samples']))) # liftered
        mspec_features = np.vstack((mspec_features, mspec(data[i]['samples'])))

    num_components = [4, 8, 16, 32]
    subplots = 221
    for component in num_components:
        model = GaussianMixture(n_components=component, covariance_type='diag')
        model.fit(mfcc_features)
        score = np.exp(model.score_samples(mfcc_features))    
        print("Score with {} components is: {}".format(component, sum(score)))
        posteriors = model.predict_proba(mfcc_features)

        plt.subplot(subplots)
        plt.pcolormesh(posteriors)
        subplots += 1
        plt.title('GMM with {} components'.format(component))

    plt.tight_layout()
    plt.show()

    if use_seven:
        data = np.array(list(filter(lambda x: x['digit'] == '7', data)))

        subplots = 221
        for idx in range(len(data)):
            # plt.subplots(nrows=2, ncols=2)
            plt.subplot(subplots)
            subplots += 1
            plt.title('Time signal for digit {} uttered by {}'.format(data[idx]['digit'], data[idx]['gender']))
            plt.plot(data[idx]['samples'])

        plt.tight_layout()
        plt.show()

        subplots = 221
        for idx in range(len(data)):

            mfcc_features = mfcc(data[idx]['samples'])
            plt.subplot(subplots)
            subplots += 1
            posteriors = model.predict_proba(mfcc_features)
            plt.pcolormesh(posteriors)
            plt.title("Posterior for digit {} uttered by {}".format(data[idx]['digit'], data[idx]['gender']))

        plt.tight_layout()
        plt.show()

if Q7:
# 7: Comparing Utterances
    
    # --------------------------------------------------
    # Compare two utterances

    utterance_1 = mfcc(data[16]['samples']) # liftered
    utterance_2 = mfcc(data[17]['samples'])
    dist_DTW, dist_loc, AccD_mat, path = dtw(utterance_1, utterance_2, getEuclidean)

    plt.pcolormesh(AccD_mat)
    plt.title('Distance matrix between features')
    plt.show()
    
    plt.title("Path between utterances")
    plt.scatter(np.array(path[:])[:,0],np.array(path[:])[:,1])
    plt.show()
    
    # ------------------------------------------------
    # Compare all features

    matrix_dtw = np.zeros((len(data),len(data)))

    for i in range(matrix_dtw.shape[0]):
        for j in range(matrix_dtw.shape[1]):
            # dist_DTW, _, _, _ = dtw(mfcc(data[i]['samples']), mfcc(data[j]['samples']), getEuclidean, Comp_Dist = False)
            dist_DTW = dtw(mfcc(data[i]['samples']), mfcc(data[j]['samples']), getEuclidean, Comp_Dist = True)
            matrix_dtw[i][j] = dist_DTW
    
    plt.title("Digit utterance comparison")
    plt.pcolormesh(matrix_dtw)
    plt.show()

    clusters = hierarchy.linkage(matrix_dtw, method='complete')
  
    # plt.figure(figsize=(25, 10))
    plt.figure(figsize=(25, 10))
    hierarchy.dendrogram(clusters, labels=tidigit2labels(data), orientation='right')
    plt.title('Dendrogram for all utterances')
    plt.show()
