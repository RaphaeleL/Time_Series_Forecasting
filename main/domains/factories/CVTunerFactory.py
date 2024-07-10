from domains import CVTuner

"""
Filename: CVTunerFactory.py
Created at: 2024-06-15

Description:
Factory for creating CVTuner instances

"""


def generate(hypermodel, oracle, nsplits, objective, directory, project_name):
    """
    args:
        hypermodel (HyperModel): The hypermodel which holds the hyperparameter configuration
        oracle (Oracle): Optimizer of type keras_tuner.Oracle
        nsplits (int): number of splits for the cross validation
        directory (str): Determines the directory in which the hyperparameters will be stored after training. This prevents redundant training if there is no configuration change
        project_name (str): Determines the file in which the hyperparameters will be stored after training. This prevents redundant training if there is no configuration change

    returns:
        cvTuner (CVTuner): The tuner for the hpo
    """
    return CVTuner.CVTuner(
        hypermodel=hypermodel,
        oracle=oracle,
        nsplits=nsplits,
        objective=objective,
        directory=directory,
        project_name=project_name,
    )
