import keras_tuner as kt

from core.domains.factories import (
    NeuronalNetworkModelFactory as NNMFactory
)
from core.helpers import CreateModelOptimizer

"""
Filename: HyperModel.py
Created at: 2024-06-15

Description:
Extends the Keras_tuner HyperModel class
Class builds the different Hyperparameters and returns the neuralNetwork 

Last Changed:
- Author: Raphaele Salvatore Licciardo, Date: 2024-06-24
"""


class HyperModel(kt.HyperModel):
    """
    Extends the Keras_tuner HyperModel class
    Class builds the different Hyperparameters and returns the neuralNetwork
    """

    def __init__(
        self,
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
        """
        self.units = units
        self.activation = activation
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.lamda = lamda
        self.metrics = metrics
        self.model_optimizer = model_optimizer
        self.learning_rate = learning_rate
        self.loss = loss

    def build(self, hp):
        """
        Overwritten build function of keras_tuner HyperModel class
        args:
            hp: Hyperparameters to optimize
        returns:
            model (keras.sequential): a neural network model
        """
        units = hp.Int(
            "units",
            min_value=self.units[0],
            max_value=self.units[1],
            step=self.units[2],
        )
        activation = hp.Choice("activation", self.activation)
        dropout = hp.Float(
            "dropout", self.dropout[0], self.dropout[1], step=self.dropout[2]
        )
        lamda = hp.Float("lamda", self.lamda[0], self.lamda[1], step=self.lamda[2])
        hp_learning_rate = hp.Float(
            "learning_rate",
            self.learning_rate[0],
            self.learning_rate[1],
            step=self.learning_rate[2],
        )

        model = NNMFactory.generate(units, activation, dropout, lamda)
        neuralNetwork = model.nn

        neuralNetwork.compile(
            optimizer=CreateModelOptimizer.build_optimizer(
                self.model_optimizer, learning_rate=hp_learning_rate
            ),
            loss=self.loss,
            metrics=self.metrics,
        )

        return neuralNetwork

    def fit(self, hp, model, *args, **kwargs):
        """
        Overwritten fit function of keras_tuner HyperModel class
        *args, **kwargs are all parameters that are not modified in this overwritten version of fit,
          so they are just passed threw to the original fit function of the neural network

        args:
            hp: Hyperparameters to optimize
            model (Sequential): the neural network
            *args: Parameters before epochs and batch_size
            **kwargs: Parameters after epochs and batch_size
        returns:
            the training results
        """

        epochs = hp.Int(
            "epochs",
            min_value=self.epochs[0],
            max_value=self.epochs[1],
            step=self.epochs[2],
        )
        batch_size = hp.Int(
            "batch_size",
            min_value=self.batch_size[0],
            max_value=self.batch_size[1],
            step=self.batch_size[2],
        )

        return model.fit(
            *args,
            epochs=epochs,
            batch_size=batch_size,
            **kwargs,
        )
