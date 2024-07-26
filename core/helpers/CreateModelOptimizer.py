import keras

"""
Filename: CreateModelOptimizer.py
Created at: 2024-06-03

Description:
This module resolves the optimizer the user passes as string over the command line to the actual optimizer classes from keras_tuner
These Optimizers define how the model will be optimated during training

Last Changed:
- Author: Raphaele Salvatore Licciardo, Date: 2024-07-01
"""

dispatcher = {"Adam": keras.optimizers.Adam}


def build_optimizer(optimizer_str, learning_rate):
    """
    The public method of ResolveModelOptimizer

    Args:
        optimizer_str (str): The optimizer in string format
        learning_rate (float): the learning rate of the training
    Returns:
        resolvedOptimizer (keras.optimizers.<optimizer_class>): The actual class of the optimizer with keras.optimizers.<optimizer_class>
    """

    if optimizer_str not in dispatcher:
        raise ValueError(
            "Optimizer "
            + optimizer_str
            + " does not exist. Please check if the name is written correctly."
        )

    return dispatcher[optimizer_str](learning_rate)
