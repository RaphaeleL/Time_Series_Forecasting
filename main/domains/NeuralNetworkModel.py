from keras._tf_keras.keras.layers import Bidirectional, LSTM, Dropout, Flatten, Dense
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.regularizers import l2

"""
Filename: NeuronalNetworkModel.py
Created at: 2024-05-12

Description:
Class for initialising and changing the neural network model used by the pipeline

"""


class NeuralNetworkModel:
    """
    Class for initialising and changing the neural network model used by the pipeline
    """

    def __init__(self, units, activation, dropout, lamda):
        """
        args:
            units (int): The amount of neurons per layer
            activation (str): The activation function used by the model
            dropout (float): Determines the percentage amount of neurons which will deactivated during training (prevents overfitting)
            epochs (int): the amount of Epochs for the training
        """
        self.nn: Sequential = Sequential(
            [
                Bidirectional(
                    LSTM(units, activation=activation, return_sequences=True)
                ),
                Bidirectional(
                    LSTM(units, activation=activation, return_sequences=True)
                ),
                Dropout(dropout),
                Bidirectional(
                    LSTM(units, activation=activation, return_sequences=True)
                ),
                Bidirectional(
                    LSTM(units, activation=activation, return_sequences=True)
                ),
                Dropout(dropout),
                Bidirectional(
                    LSTM(units, activation=activation, return_sequences=True)
                ),
                Bidirectional(
                    LSTM(units, activation=activation, return_sequences=True)
                ),
                Dropout(dropout),
                Flatten(),
                Dense(50, activation=activation, kernel_regularizer=l2(lamda)),
                Dense(1),
            ]
        )
        self.nn.evaluate

    def set_model(self, model):
        """
        Method for further implementaion
        Sets new model for the pipeline (need to be of type keras.model.Sequential)

        args:
            model (Sequential): the neural network
        """
        if isinstance(model, Sequential):
            self.nn = model
        else:
            raise TypeError(
                "Invalid type for model, needs to be of type "
                "keras.models.Sequential"
                "!"
            )
