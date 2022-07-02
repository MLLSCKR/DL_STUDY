# mathutil

import pandas as pd
import numpy as np
import os

def relu(x):
    return np.maximum(x, 0)

def derv_relu(x):
    return np.sign(x)
