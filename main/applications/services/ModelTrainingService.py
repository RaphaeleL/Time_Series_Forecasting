from applications.usecases import (
    CreateTestData,
    PerformModelTrainingAndTesting,
    CreateModel,
)
from configs import HyperparameterSavePathConfig as hpspc
from helpers import MetricsResolver, FeatureNameBuilder, LabelExtractor
from applications.usecases import PerformDataPreparation

"""
Filename: ModelTrainingService.py
Created at: 2024-06-01

Description:
Service Class
Calls the DataPreperation, creates the Models and calls the Training with or without Hyperparameteroptimization followed by the final test

"""


def execute_model_training(
    csv,
    inputlenght,
    predictionlenght,
    epochs,
    batchsize,
    testsize,
    hyper_optimizer,
    nsplits,
    objective,
    maxtrials,
    should_optimize,
    units,
    activation,
    dropout,
    lamda,
    metrics,
    model_optimizer,
    loss,
    learning_rate,
    validation_split,
    project_name,
):
    """
    Forwarding only, calls all the necessary functionality for model training with ot without hpo
    """

    training_data, test_data = PerformDataPreparation.execute(csv=csv, testsi=testsize)
    metrics = MetricsResolver.resolve(metrics)

    model = CreateModel.execute(
        should_optimize=should_optimize,
        optimization_method=hyper_optimizer,
        nsplits=nsplits,
        units=units,
        activation=activation,
        dropout=dropout,
        epochs=epochs,
        batch_size=batchsize,
        lamda=lamda,
        objective=objective,
        max_trials=maxtrials,
        metrics=metrics,
        model_optimizer=model_optimizer,
        directory=hpspc.DIRECTORY,
        learning_rate=learning_rate,
        project_name=project_name,
        loss=loss,
    )

    feature_names = FeatureNameBuilder.build(training_data)
    X_train, y_train = LabelExtractor.create_features_and_labels(
        training_data, inputlenght, predictionlenght, feature_names
    )
    X_test, y_test = CreateTestData.execute(
        test_data=test_data, inplen=inputlenght, predlen=predictionlenght
    )
    return PerformModelTrainingAndTesting.execute(
        should_optimize=should_optimize,
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        epochs=epochs,
        metrics=metrics,
        model_optimizer=model_optimizer,
        loss=loss,
        learning_rate=learning_rate,
        validation_split=validation_split,
    )
