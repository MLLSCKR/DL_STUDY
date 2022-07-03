# mathutil

import pandas as pd
import numpy as np
import os

def relu(x):
    return np.maximum(x, 0)

def derv_relu(x):
    return np.sign(x)

def batch_mean(x):
    return np.mean(x, axis = 1).reshape(x.shape[0], 1)

def batch_std(x):
    return np.std(x, axis = 1).reshape(x.shape[0], 1)