import pandas as pd
import csv

"""
Filename: CsvReader.py
Created at: 2024-05-21

Description:
Reads in csv files and checks if they are valid csv files

"""


def read_if_csv_is_valid(path):
    """
    args:
        path (str): the path to the csv file
    returns:
        data (Dataframe): the data as pd.Dataframe
    """
    try:
        data = pd.read_csv(path)
        return data
    except (csv.Error, IOError) as e:
        raise ValueError(f"Not a valid csv file: {e}")
