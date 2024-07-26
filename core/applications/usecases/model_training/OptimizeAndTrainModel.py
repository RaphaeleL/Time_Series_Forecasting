"""
Filename: OptimizeAndTrainModel.py
Created at: 2024-06-14

Description:
This module executes the modeltraining with a hyperparameter optimization

Last Changed:
- Author: Richard Höpken, Date: 2024-07-03
"""


def execute(tuner, X_train, y_train):
    """
    Args:
      tuner (HyperModel): Instance of HyperModel class
      X_train (list<?>): The featureset of the trainingdata
      y_train (list<?>): The labelset of the trainingdata
    Returns:
      The best performing model of the hyperparameter optimization
    """
    if type(tuner) == str:
        raise ValueError(tuner)
    tuner.search(X_train, y_train)
    return tuner.get_best_models()[0]
