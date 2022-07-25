"""
exec file for testing mlp
"""

from class_mlp_model import *
from class_dataset_regression import *
from class_dataset_classification import *
from class_dataset_adam import *
from adam_mlp_model import *

# Regression Model Execution
def Regression_exec():
    ad = Regression_Dataset('Regression Model1', 'regression')
    am = mlp_model("Regression Model1", [7, 5, 3], 'regression', ad)
    am.__exec__(epoch_count = 10, report = 1, cnt = 5)

    ad2 = Regression_Dataset('Regression Model2', 'regression')
    am2 = mlp_model('Regression Model2', [7, 5, 3], 'regression', ad2, True)
    am2.__exec__(epoch_count= 10, report = 1, cnt = 5)


# Select(Multi Classification) Model Execution
def Select_exec():
    dataset_1 = Classifiaction_Dataset('Selection_model1', 'Select(Multi Classification)', resolution = [100, 100], input_shape = [-1])
    fm1 = mlp_model("Selection_model1", [100, 50, 25, 10], 'Select(Multi Classification)', dataset_1, False)
    fm1.__str__()
    fm1.__exec__(epoch_count = 10, report = 1, cnt = 5)

    dataset_2 = Classifiaction_Dataset('Selection_model2', 'Select(Multi Classification)', resolution = [100, 100], input_shape = [-1])
    fm2 = mlp_model("Selection_model2", [100, 50, 25, 10], 'Select(Multi Classification)', dataset_2, True)
    fm2.__str__()
    fm2.__exec__(epoch_count = 10, report = 1, cnt = 5)

# Adam Algorithm Model Execution
def Adam_exec():
    dataset_1 = Adam_Dataset('Adam Selection Model1', 'Adam', resolution = [100, 100], input_shape = [-1])
    adam1 = AdamModel('Adam', [100, 10], 'Adam', dataset_1, False)
    adam1.__str__()
    adam1.__exec__()

#Regression_exec()
#Select_exec()
Adam_exec()