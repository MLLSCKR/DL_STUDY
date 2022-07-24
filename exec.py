"""
exec file for testing mlp
"""

from class_mlp_model import *
from class_dataset_regression import *
from class_dataset_classification import *

# Regression Model Execution
"""
ad = Regression_Dataset('Regression Model1', 'regression')
am = mlp_model("Regression Model1", [7, 5, 3], 'regression', ad)
am.__exec__(epoch_count = 10, report = 1, cnt = 5)

ad2 = Regression_Dataset('Regression Model2', 'regression')
am2 = mlp_model('Regression Model2', [7, 5, 3], 'regression', ad2, True)
am2.__exec__(epoch_count= 10, report = 1, cnt = 5)
"""

# Select(Multi Classification) Model Execution
dataset_1 = Classifiaction_Dataset('Selection_model1', 'Select(Multi Classification)', resolution = [100, 100], input_shape = [-1])
fm1 = mlp_model("Selection_model1", [10, 5], 'Select(Multi Classification)', dataset_1, False)
fm1.__exec__(epoch_count = 10, report = 1, cnt = 5)

dataset_2 = Classifiaction_Dataset('Selection_model2', 'Select(Multi Classification)', resolution = [100, 100], input_shape = [-1])
fm2 = mlp_model("Selection_model2", [10, 5], 'Select(Multi Classification)', dataset_2, True)
fm2.__exec__(epoch_count = 10, report = 1, cnt = 5)
