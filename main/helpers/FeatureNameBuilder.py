"""
Filename: FeatureNameBuilder.py
Created at: 2024-06-11

Description:
Extracts all features out of the dataset by their name

"""


def build(data):
    """
    args:
        data (dataframe): the data as pd dataframe
    returns:
        col (list<str>): The names of the features
    """

    return [col for col in data if col.startswith("feature_")]
