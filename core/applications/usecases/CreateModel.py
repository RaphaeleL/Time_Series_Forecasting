from core.helpers import CreateOracleOptimizer
from core.domains.factories import (
    HyperModelFactory as HMFactory,
    NeuronalNetworkModelFactory as NNMFactory,
    CVTunerFactory,
)

"""
Filename: CreateModel.py
Created at: 2024-06-15

Description:
Creates an Hypermodel instance if should_optimize = True, creates a NeuronalNetworkModel if should_optimize = False

Last Changed:
- Author: Richard Höpken, Date: 2024-07-03
"""


def execute(
    should_optimize,
    optimization_method,
    n_splits,
    units,
    activation,
    dropout,
    epochs,
    batch_size,
    lamda,
    objective,
    max_trials,
    metrics,
    model_optimizer,
    loss,
    directory,
    learning_rate,
    project_name,
):
    """
    The public method of CreateModel

    Args:
        should_optimize (boolean):
        optimization_method (str): The optimizer in string format
        n_splits (int): number of splits for the cross validation
        units (int): The amount of neurons per layer
        activation (str): The activation function used by the model
        dropout (float): Determines the percentage amount of neurons which will deactivated during training (prevents overfitting)
        epochs (int): the amount of Epochs for the training
        batch_size (int): The batch size determines the amount of batches your training data will be divided into
        lambda (float): The Lambda value for your regularisation (L1 and L2).
        objective (str): The Metric or value which is used as feedback for the hyperparameter optimization
        max_trials (int): Determines the amount of max different combinations of hyperparameters
        metrics (list(keras.metrics.<metric_class>)): The resolved metrics for model performancetesting after training and testing
        model_optimizer (str): the optimizer determines how the model will be optimated during training
        loss (str): Detemines on what metric the model will optimize during training
        directory (str): Determines the directory in which the hyperparameters will be stored after training. This prevents redundant training if there is no configuration change
        learning_rate (float): the learning rate of the training
        project_name (str): Determines the file in which the hyperparameters will be stored after training. This prevents redundant training if there is no configuration change
    Returns:
        X_test (list<list<?>>): The feature data used for the test
        y_test (list<list<?>>): The label data used for the test
    """

    if should_optimize:
        model = HMFactory.generate(
            units=units,
            activation=activation,
            dropout=dropout,
            epochs=epochs,
            batch_size=batch_size,
            lamda=lamda,
            metrics=metrics,
            model_optimizer=model_optimizer,
            learning_rate=learning_rate,
            loss=loss,
        )

        return CVTunerFactory.generate(
            hypermodel=model,
            n_splits=n_splits,
            oracle=CreateOracleOptimizer.build_optimizer(
                optimization_method, objective=objective, max_trials=max_trials
            ),
            objective=objective,
            directory=directory,
            project_name=project_name,
        )

    else:
        return NNMFactory.generate(
            units=units[0], activation=activation[0], dropout=dropout[0], lamda=lamda[0]
        )
