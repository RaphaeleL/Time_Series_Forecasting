from core.applications.services import ModelTrainingService

"""
Filename: ModelTrainingController.py
Created at: 2024-06-01

Description:
Controller class
Forwardes to the ModelTrainingService

Last Changed:
- Author: Richard Höpken, Date: 2024-07-03
"""


def execute_model_training(
    csv,
    input_length,
    prediction_length,
    epochs,
    batch_size,
    test_size,
    hyper_parameter_optimizer,
    n_splits,
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
    print("execution")
    return ModelTrainingService.execute_model_training(
        csv,
        input_length,
        prediction_length,
        epochs,
        batch_size,
        test_size,
        hyper_parameter_optimizer,
        n_splits,
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
    )
