import numpy as np
import pandas as pd
from regex import W
from sympy import re

test_array = np.random.randint(0, 100, size = 100)

test_df = pd.DataFrame({'col1' : test_array, 'col2' : test_array * 2})

a = 100