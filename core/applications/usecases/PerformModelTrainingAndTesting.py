from core.applications.usecases.model_training import (
    OptimizeAndTrainModel,
    EvaluateModel,
    TrainModel,
)


"""
Filename: PerformModelTrainingAndTesting.py
Created at: 2024-06-03

Description:
This module calls the training (hyperparameter optimization and training if should_optimize = True) and testing usecases

Last Changed:
- Author: Raphaele Salvatore Licciardo, Date: 2024-07-01
"""


def execute(
    should_optimize,
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    epochs,
    metrics,
    model_optimizer,
    loss,
    learning_rate,
    validation_split,
):
    """
    The public method of PerformModelTrainingAndTesting

    Args:
        should_optimize (boolean):
        model (HyperModel): Or Tuner. Instance of NeuronalNetworkModel class or instance of HyperModel class if should_optimize = True
        X_train (list<?>): The featureset of the trainingdata
        y_train (list<?>): The labelset of the trainingdata
        X_val (list<?>): The featureset of the valuationdata
        y_val (list<?>): The labelset of the valuationdata
        X_test (list<?>): The featureset of the testdata
        y_test (list<?>): The labelset of the testdata
        epochs (Tuple(int)/int): The epochs which determine how many runs the model does threw the dataset during modeltraining
        metrics (list(keras.metrics.<metric_class>)): The resolved metrics for model performancetesting after training and testing
        model_optimizer (str): the optimizer determines how the model will be optimated during training
        loss (str): Detemines on what metric the model will optimize during training
        learning_rate (float): the learning rate of the training
        validation_split (float): the size of the validation data during model training without hpo
    returns:
        Returns the loss value & metrics values for the model in test mode.
    """
    if should_optimize:
        print("\nstart of hyperparameter optimization and training")
        best_model = OptimizeAndTrainModel.execute(model, X_train, y_train)

        print("\nstart of testing")
        return EvaluateModel.execute(best_model, X_test, y_test)
    else:
        print("\nstart of training")
        TrainModel.execute(
            model,
            X_train,
            y_train,
            epochs[0],
            metrics,
            model_optimizer,
            loss,
            learning_rate[0],
            validation_split,
        )

        print("\nstart of testing")
        return EvaluateModel.execute(model.nn, X_test, y_test)
