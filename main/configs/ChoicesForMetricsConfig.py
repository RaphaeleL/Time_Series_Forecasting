"""
Filename: ChoicesForMetricsConfig.py
Created at: 2024-06-15

Description:
Stores possible metrics for the training results. If you want to use metrics which is not listed here there will be an error raised by the argument parser.
You can add more metrics, just make sure that they are supported by "keras".

"""

choices_for_metrics = [
    "MeanSquaredError",
    "RootMeanSquaredError",
    "MeanAbsoluteError",
    "MeanAbsolutePercentageError",
    "MeanSquaredLogarithmicError",
    "R2Score",
    "LogCoshError",
    "CosineSimilarity",
]
