"""
Filename: EvaluateModel.py
Created at: 2024-05-12

Description:
This module executes the modeltesting

"""


def execute(model, X_test, y_test):
    """
    Args:
       model (NeuronalNetworkModel): Instance of NeuronalNetworkModel class
       X_test (list<?>): The featureset of the testdata
       y_test (list<?>): The labelset of the testdata
    Returns:
       The test result (metrics)
    """
    return model.evaluate(X_test, y_test)
