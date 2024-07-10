from applications.services import ModelTrainingService

"""
Filename: ModelTrainingController.py
Created at: 2024-06-01

Description:
Controller class
Forwardes to the ModelTrainingService

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
    print("execution")
    return ModelTrainingService.execute_model_training(
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
    )
