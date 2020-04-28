import numpy as np
from lab2_proto import *
from lab2_tools import *
from prondict import prondict
import matplotlib.pyplot as plt

# Data contains MFCC features
data = np.load('lab2_data.npz', allow_pickle=True)['data']

# Models to test functions, trained on TIDIGITS with 13 MFCC feature vectors
models_onespkr = np.load('lab2_models_onespkr.npz') # trained on a single speaker (female)
models_all = np.load('lab2_models_all.npz') # trained on full training set

# Example for debugging
example = np.load('lab2_example.npz')["example"]

# Load one of the model files
phoneHMMs_onespkr = np.load('lab2_models_onespkr.npz', allow_pickle=True)['phoneHMMs'].item()
phoneHMMs_all = np.load('lab2_models_all.npz', allow_pickle=True)['phoneHMMs'].item()

The resulting variable phoneHMMs is a dictionary with 21 keys each corresponding to a phonetic
model

list(sorted(phoneHMMs.keys()))