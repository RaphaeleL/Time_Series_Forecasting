from core.domains import CVTuner

"""
Filename: CVTunerFactory.py
Created at: 2024-06-15

Description:
Factory for creating CVTuner instances

Last Changed:
- Author: Richard Höpken, Date: 2024-07-01
"""


def generate(hypermodel, oracle, n_splits, objective, directory, project_name):
    """
    args:
        hypermodel (HyperModel): The hypermodel which holds the hyperparameter configuration
        oracle (Oracle): Optimizer of type keras_tuner.Oracle
        n_splits (int): number of splits for the cross validation
        directory (str): Determines the directory in which the hyperparameters will be stored after training. This prevents redundant training if there is no configuration change
        project_name (str): Determines the file in which the hyperparameters will be stored after training. This prevents redundant training if there is no configuration change

    returns:
        cvTuner (CVTuner): The tuner for the hpo
    """
    return CVTuner.CVTuner(
        hypermodel=hypermodel,
        oracle=oracle,
        n_splits=n_splits,
        objective=objective,
        directory=directory,
        project_name=project_name,
    )
