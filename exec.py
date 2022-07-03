"""
exec file for testing mlp
"""

from class_mlp_model import *
from class_dataset_regression import *


ad = Regression_Dataset('Regression Model1', 'regression')
am = mlp_model("Regression Model1", [7, 5, 3], 'regression', ad)
am.__exec__(epoch_count = 10, report = 2, cnt = 5)

ad2 = Regression_Dataset('Regression Model2', 'regression')
am2 = mlp_model('Regression Model2', [7, 5, 3], 'regression', ad2, True)
am2.__exec__(epoch_count= 10, report = 2, cnt = 5)