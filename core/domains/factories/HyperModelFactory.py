from core.domains import HyperModel as hm

"""
Filename: HyperModelFactory.py
Created at: 2024-06-15

Description:
Factory for creating HyperModel instances

Last Changed:
- Author: Richard Höpken, Date: 2024-06-24
"""


def generate(
    units,
    activation,
    dropout,
    batch_size,
    epochs,
    lamda,
    metrics,
    model_optimizer,
    loss,
    learning_rate,
):
    """
    args:
        units (int): The amount of neurons per layer
        activation (str): The activation function used by the model
        dropout (float): Determines the percentage amount of neurons which will deactivated during training (prevents overfitting)
        epochs (int): the amount of Epochs for the training
        batch_size (int): The batch size determines the amount of batches your training data will be divided into
        lambda (float): The Lambda value for your regularisation (L1 and L2).
        metrics (list(keras.metrics.<metric_class>)): The resolved metrics for model performancetesting after training and testing
        model_optimizer (str): the optimizer determines how the model will be optimated during training
        loss (str): Detemines on what metric the model will optimize during training
        learning_rate (float): the learning rate of the training
    returns:
        hypermodel (Hypermodel): A new HyperModel instance
    """
    return hm.HyperModel(
        units=units,
        activation=activation,
        dropout=dropout,
        batch_size=batch_size,
        epochs=epochs,
        lamda=lamda,
        metrics=metrics,
        model_optimizer=model_optimizer,
        learning_rate=learning_rate,
        loss=loss,
    )
