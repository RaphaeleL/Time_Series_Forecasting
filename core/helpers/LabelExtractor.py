import numpy as np

"""
Filename: LabelExtractor.py
Created at: 2024-06-03

Description:
extracts the labels from the dataset

Last Changed:
- Author: Richard Höpken, Date: 2024-06-24
"""


def create_features_and_labels(data, input_length, prediction_length, feature_names):
    """
    args:
        data (dataframe): the data as pd dataframe
        input_length (int): The input length determines the size of each element in X_test
        prediction_length (int): The prediction length determines the size of each element in y_test
        feature_names (list<str>): The names of the features
    """
    X, y = [], []
    for i in range(len(data) - input_length - prediction_length):
        X.append(data.iloc[i : i + input_length][feature_names].values)
        y.append(
            data.iloc[i + input_length : i + input_length + prediction_length][
                feature_names[0]
            ].values
        )

    X = np.array(X)
    y = np.array(y)

    return X, y
