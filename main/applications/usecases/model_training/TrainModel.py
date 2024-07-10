from helpers import CreateModelOptimizer

"""
Filename: TrainModel.py
Created at: 2024-06-15

Description:
This module executes the modeltraining without a hyperparameter optimization

"""


def execute(
    model,
    X_train,
    y_train,
    epochs,
    metrics,
    model_optimizer,
    loss,
    learning_rate,
    validation_split,
):
    """
    Args:
       model (NeuronalNetworkModel): Instance of NeuronalNetworkModel class
       X_train (list<?>): The featureset of the trainingdata
       y_train (list<?>): The labelset of the trainingdata
       metrics (list(keras.metrics.<metric_class>)): The resolved metrics for model performancetesting after training and testing
       model_optimizer (str): the optimizer determines how the model will be optimated during training
       loss (str): Detemines on what metric the model will optimize during training
       learning_rate (float): the learning rate of the training
       validation_split (float): the size of the validation data during model training without hpo
    Returns:
       Nothing
    """
    model.nn.compile(
        optimizer=CreateModelOptimizer.build_optimizer(
            model_optimizer, learning_rate=learning_rate
        ),
        loss=loss,
        metrics=metrics,
    )

    model.nn.fit(X_train, y_train, epochs=epochs, validation_split=validation_split)
