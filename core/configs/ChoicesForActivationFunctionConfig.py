"""
Filename: ChoicesForActivationFunctionConfig.py
Created at: 2024-06-15

Description:
Stores possible values for the activation function. If you want to use a activation function which is not listed here there will be an error raised by the argument parser.
You can add more activation functions, just make sure that they are supported by "keras".

Last Changed:
- Author: Raphaele Salvatore Licciardo, Date: 2024-07-01
"""

choices_for_activation = [
    "relu",
    "tanh",
    "sigmoid",
    "softmax",
    "softplus",
    "softsign",
    "selu",
    "elu",
    "exponential",
    "leaky_relu",
    "relu6",
    "silu",
    "hard_silu",
    "gelu",
    "hard_sigmoid",
    "linear",
    "mish",
    "log_softmax",
]
