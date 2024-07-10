"""
Filename: ChoicesForHyperOptimizerConfig.py
Created at: 2024-06-15

Description:
Stores possible values for the Hyperparameter optimizer. If you want to use an optimizer which is not listed here there will be an error raised by the argument parser.
You can add more optimizers, just make sure that they are of type "keras_tuner.oracles"

"""

choices_for_optimizer = [
    "RandomSearch",
    "GridSearch",
    "BayesianOptimization",
    "Hyperband",
]
