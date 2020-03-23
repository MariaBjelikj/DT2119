# DT2119, Lab 1 Feature Extraction
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ssi
from scipy import fftpack
from lab1_tools import trfbank
from scipy.fftpack.realtransforms import dct
from lab1_tools import *

# Function given by the exercise ----------------------------------

def mspec(samples, winlen=400, winshift=200, preempcoeff=0.97, nfft=512, samplingrate=20000):
    """Computes Mel Filterbank features.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        samplingrate: sampling rate of the original signal

    Returns:
        N x nfilters array with mel filterbank features (see trfbank for nfilters)
    """
    frames = enframe(samples, winlen, winshift)
    preemph = preemp(frames, preempcoeff)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)
    return logMelSpectrum(spec, samplingrate)

def mfcc(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, nceps=13, samplingrate=20000, liftercoeff=22):
    """Computes Mel Frequency Cepstrum Coefficients.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        nceps: number of cepstrum coefficients to compute
        samplingrate: sampling rate of the original signal
        liftercoeff: liftering coefficient used to equalise scale of MFCCs

    Returns:
        N x nceps array with lifetered MFCC coefficients
    """
    mspecs = mspec(samples, winlen, winshift, preempcoeff, nfft, samplingrate)
    ceps = cepstrum(mspecs, nceps)
    return lifter(ceps, liftercoeff)

# Functions to be implemented ----------------------------------

def enframe(samples, winlen, winshift):
    """
    Slices the input samples into overlapping windows.

    Args:
        winlen: window length in samples.
        winshift: shift of consecutive windows in samples
    Returns:
        numpy array [N x winlen], where N is the number of windows that fit
        in the input signal
    """
    
    frames = np.array(samples[0:winlen]) # Start array here with the first frame
                                         # and stack all the frames on top for 
                                         # each window shift
    for i in range(winshift, len(samples) - winlen, winshift): 
        frames = np.vstack([frames, samples[i:i+winlen]]) # extract and stack frames
        
    return frames
    
    
def preemp(input, p=0.97):
    """
    Pre-emphasis filter.

    Args:
        input: array of speech frames [N x M] where N is the number of frames and
               M the samples per frame
        p: preemhasis factor (defaults to the value specified in the exercise)

    Output:
        output: array of pre-emphasised speech samples
    Note (you can use the function lfilter from scipy.signal)
    """
    
    # For the definition of the filter coefficients, check the documentation at
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html
    # and slide 67 in lecture 02 (alpha = p = 0.97)
    return ssi.lfilter([1, -p], [1], input)


def windowing(input):
    """
    Applies hamming window to the input frames.

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
    Output:
        array of windoed speech samples [N x M]
    Note (you can use the function hamming from scipy.signal, include the sym=0 option
    if you want to get the same results as in the example)
    """
    
    hamming_window = ssi.hamming(input.shape[1], sym=False)
    
    # Plot the hamming window.
    #print("Plotting the hamming window shape...")
    #plt.plot(hamming_window)
    #plt.title("Hamming window")
    #plt.show()
    
    # Why we use hamming window at: 
    # https://stackoverflow.com/questions/5418951/what-is-the-hamming-window-for
    return hamming_window * input


def powerSpectrum(input, nfft):
    """
    Calculates the power spectrum of the input signal, that is the square of the modulus of the FFT

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
        nfft: length of the FFT
    Output:
        array of power spectra [N x nfft]
    Note: you can use the function fft from scipy.fftpack
    """
    
    return np.power(np.abs(fftpack.fft(input, nfft)), 2)

def logMelSpectrum(input, samplingrate):
    """
    Calculates the log output of a Mel filterbank when the input is the power spectrum

    Args:
        input: array of power spectrum coefficients [N x nfft] where N is the number of frames and
               nfft the length of each spectrum
        samplingrate: sampling rate of the original signal (used to calculate the filterbank shapes)
    Output:
        array of Mel filterbank log outputs [N x nmelfilters] where nmelfilters is the number
        of filters in the filterbank
    Note: use the trfbank function provided in lab1_tools.py to calculate the filterbank shapes and
          nmelfilters
    """
    
    #print("Plotting the filters in linear frequency scale...")
    #plt.plot(trfbank(samplingrate, input.shape[1])) # nfft = input.shape[1]
    #plt.title("Filters in linear frequency scale")
    #plt.show()
    return np.log(input.dot(trfbank(samplingrate, input.shape[1]).T))


def cepstrum(input, nceps):
    """
    Calulates Cepstral coefficients from mel spectrum applying Discrete Cosine Transform

    Args:
        input: array of log outputs of Mel scale filterbank [N x nmelfilters] where N is the
               number of frames and nmelfilters the length of the filterbank
        nceps: number of output cepstral coefficients
    Output:
        array of Cepstral coefficients [N x nceps]
    Note: you can use the function dct from scipy.fftpack.realtransforms
    """

    return dct(input)[:, :nceps]
    
def dtw(x, y, dist):
    """Dynamic Time Warping.

    Args:
        x, y: arrays of size NxD and MxD respectively, where D is the dimensionality
              and N, M are the respective lenghts of the sequences
        dist: distance function (can be used in the code as dist(x[i], y[j]))

    Outputs:
        d: global distance between the sequences (scalar) normalized to len(x)+len(y)
        LD: local distance between frames from x and y (NxM matrix)
        AD: accumulated distance between frames of x and y (NxM matrix)
        path: best path thtough AD

    Note that you only need to define the first output for this exercise.
    """

def compare(frames, example_frames):
    """ Plots sample and example and returns True if they are equal """
    
    plt.pcolormesh(frames)
    plt.show()
    
    plt.pcolormesh(example_frames)
    plt.show()

    return np.isclose(frames, example_frames).all()