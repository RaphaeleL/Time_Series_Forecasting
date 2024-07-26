from core.helpers import LabelExtractor, FeatureNameBuilder

"""
Filename: CreateTestData.py
Created at: 2024-06-03

Description:
This module initiates the labelextraction for the testdata

Last Changed:
- Author: Raphaele Salvatore Licciardo, Date: 2024-07-03
"""


def execute(test_data, inplen, predlen):
    """
    The public method of CreateTestData

    Args:
        test_data (list<?>): The part of the data reserved for testing
        inplen (int): The input length determines the size of each element in X_test
        predlen(int): The prediction length determines the size of each element in y_test
    Returns:
        X_test (list<list<?>>): The feature data used for the test
        y_test (list<list<?>>): The label data used for the test
    """
    feature_names = FeatureNameBuilder.build(test_data)
    X_test, y_test = LabelExtractor.create_features_and_labels(
        test_data, inplen, predlen, feature_names
    )
    return X_test, y_test
