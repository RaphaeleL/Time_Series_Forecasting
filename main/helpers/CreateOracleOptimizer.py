import keras_tuner


"""
Filename: CreateOracleOptimizer.py
Created at: 2024-06-03

Description:
This module resolves the hyper_optimizer the user passes as string over the command line to the actual hyper_optimizer ORacle from keras_tuner
These Optimizers define how and which hyperparameters will be tried in the hyperparameter optimization
An Oracle is a keras_tuner Class. It is basicly the Keras_tuner Hyper_Optimizer class which takes own arguments itself
Read more about keras_tuner.Oracle here: https://keras.io/api/keras_tuner/oracles/

"""

dispatcher = {
    "RandomSearch": keras_tuner.oracles.RandomSearchOracle,
    "GridSearch": keras_tuner.oracles.GridSearchOracle,
    "BayesianOptimization": keras_tuner.oracles.BayesianOptimizationOracle,
    "Hyperband": keras_tuner.oracles.HyperbandOracle,
}


def build_optimizer(optimizer_str, objective, max_trials):
    """
    The public method of CreateOracleOptimizer

    Args:
        optimizer_str (str): The optimizer in string format
        model (HyperModel): Or Tuner. Instance of HyperModel class
        objective (str): The Metric or value which is used as feedback for the hyperparameter optimization
        max_trials (int): Determines the amount of max different combinations of hyperparameters
        directory (str): Determines the directory in which the hyperparameters will be stored after training. This prevents redundant training if there is no configuration change
        project_name (str): Determines the file in which the hyperparameters will be stored after training. This prevents redundant training if there is no configuration change
    Returns:
        resolvedOptimizer (keras_tuner.<optimizer_class>): The actual class of the optimizer with keras_tuner.<optimizer_class>
    """

    if optimizer_str not in dispatcher:
        raise ValueError(
            "Optimizer "
            + optimizer_str
            + " does not exist. Please check if the name is written correctly."
        )

    return dispatcher[optimizer_str](objective, max_trials)
