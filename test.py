from bitarray import test
import numpy as np
import pandas as pd
from regex import W
from sympy import re
import mathutil

test_array = np.random.random(size = (100, 10))

a = np.std(test_array, axis = 1).reshape(100, 1)

epsilon = np.ones(shape = (test_array.shape[0], 1)) * 1e-5

print(epsilon)