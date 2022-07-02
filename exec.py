"""
exec file for testing mlp
"""

from class_mlp_model import *
from class_dataset_regression import *


ad = Regression_Dataset('Regression Model1', 'regression')
am = mlp_model("Regression Model1", [10, 5], 'regression', ad)
am.__exec__(epoch_count = 10, report = 2)