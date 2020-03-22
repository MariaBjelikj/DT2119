from lab1_proto import *
import lab1_tools as tools
import numpy as np
import matplotlib.pyplot as plt

data = np.load('lab1_data.npz', allow_pickle=True)['data']
example = np.load('./lab1_example.npz', allow_pickle=True)['example'].item()

plt.plot(example['samples'])
plt.title('Speech samples')
plt.show()

# 4.1: Enframe
frames = enframe(example['samples'], 400, 200)
print(compare(frames, example['frames']))

# 4.2: Pre-emphasis filter
pre_emphasis = preemp(frames, 0.97)
print(compare(pre_emphasis, example['preemph']))

# 4.3: Hamming window
hamming_window = windowing(pre_emphasis)
print(compare(hamming_window, example['windowed']))

# 4.4: Fast Fourier Transfrom
fast_fourier_transform = powerSpectrum(hamming_window, 512)
print(compare(fast_fourier_transform, example['spec']))

# 4.5: Mel filterbank log spectrum
log_mel_spectrum = logMelSpectrum(fast_fourier_transform, 20000)
print(compare(log_mel_spectrum, example['mspec']))

# 4.6: Cosine transform and filtering
cosine_transfrom = cepstrum(log_mel_spectrum, 13)
print(compare(cosine_transfrom, example['mfcc']))

l_cosine_transfrom = tools.lifter(cosine_transfrom)
print(compare(l_cosine_transfrom, example['lmfcc']))


# Apply to data
