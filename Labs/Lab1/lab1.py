from lab1_proto import *
import lab1_tools as tools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


# Load data (utterances of digits) and example
data = np.load('lab1_data.npz', allow_pickle=True)['data']
example = np.load('./lab1_example.npz', allow_pickle=True)['example'].item()


# Plot the speech sample
plt.plot(example['samples'])
plt.title('Speech samples')
plt.show()


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


# 5: Feature Correlation
mfcc_correlation = np.corrcoef(mfcc_features.T)
plt.title("Correlation of liftered MFCC features")
plt.pcolormesh(mfcc_correlation)
plt.show()

mspec_correlation = np.corrcoef(mspec_features.T)
plt.title("Correlation of MFCC features")
plt.pcolormesh(mspec_correlation)
plt.show()


# 6: Speech Segments with Clustering
# Fit a Gaussian Mixture Model to the data
model = GaussianMixture(n_components=4, covariance_type='diag')
model.fit(mfcc_features)
score = np.exp(model.score_samples(mfcc_features))
print(sum(score))