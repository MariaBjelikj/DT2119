import numpy as np
from lab2_proto import *
from lab2_tools import *
from prondict import prondict
import matplotlib.pyplot as plt

data = np.load('lab2_data.npz')['data']

models_onespkr = np.load('lab2_models_onespkr.npz')
models_all = np.load('lab2_models_all.npz')
example = np.load('lab2_example.npz')["example"]
example.shape=(1,)
example = example[0]
