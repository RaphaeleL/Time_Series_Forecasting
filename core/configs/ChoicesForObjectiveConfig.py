"""
Filename: ChoicesForObjectiveConfig.py
Created at: 2024-06-15

Description:
Stores possible values for the metric u want to evaluate your model on. If you want to use an evaluation metric which is not listed here there will be an error raised by the argument parser.
You can add more evaluation metrics, just make sure that they are supported by "keras"

Last Changed:
- Author: Raphaele Salvatore Licciardo, Date: 2024-07-01
"""

choices_for_objective = [
    "val_loss",
]
