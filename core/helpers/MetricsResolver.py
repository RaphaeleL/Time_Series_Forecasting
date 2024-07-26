import importlib

"""
Filename: MetricsResolver.py
Created at: 2024-06-03

Description:
This module resolves the metrics the user passes as string over the command line to the actual metric classes from keras.metrics
These Metrics measure the performance of the model

Last Changed:
- Author: Raphaele Salvatore Licciardo, Date: 2024-06-24
"""


def resolve(metrics):
    """
    The public method of ResolveMetrics

    Args:
        metrics (list(str)): The metrics in string format
    Returns:
        resolvedMetrics (list(keras.metrics.<metric_class>)): The actual classes of the metrics with keras.metrics.<metric_class>
    """
    resolvedMetrics = []
    for metric in metrics:
        try:
            resolvedMetrics.append(__class_for_name(metric))
        except:
            raise ValueError(
                "Metric "
                + metric
                + " does not exist. Please check if the name is written correctly."
            )
    return resolvedMetrics


def __class_for_name(class_name):
    """
    Resolves a given string class_name to a class

    Args:
        class_name (str): Name of the class
    Returns:
        keras.metrics.<class_name>: The resolved class
    """
    module = importlib.import_module("keras.metrics")
    # get the class, will raise AttributeError if class cannot be found
    return getattr(module, class_name)
