from core.helpers import CsvReader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import pandas as pd


"""
Filename: PerformDataPreparation.py
Created at: 2024-06-03

Description:
This module take necessary data preperation steps like reading in the csv file, scaling the data and splitting the data into train and test data

Last Changed:
- Author: Raphaele Salvatore Licciardo, Date: 2024-07-03
"""


def execute(csv, testsi):
    """
    The public method of PerformDataPreperation

    args:
        csv (str): The path to the csv file where the data is located
        testsi (float): The size of the testdata (in precent)
    returns:
        training__data (list<?>): The training data set
        test__data (list<?>): The test data set
    """
    scaler = MinMaxScaler()
    data = CsvReader.read_if_csv_is_valid(csv)
    data_scaled_array = scaler.fit_transform(data)
    data_scaled = pd.DataFrame(
        data_scaled_array, index=data.index, columns=data.columns
    )
    training_data, test_data = train_test_split(data_scaled, test_size=testsi)

    return training_data, test_data
