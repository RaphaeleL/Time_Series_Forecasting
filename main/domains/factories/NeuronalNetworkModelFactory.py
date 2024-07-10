from domains import NeuralNetworkModel as nwm


"""
Filename: NeuronalNetworkModelFactory.py
Created at: 2024-06-15

Description:
Factory for creating NeuronalNetworkModel instances

"""


def generate(units, activation, dropout, lamda):
    """
    args:
        units (int): The amount of neurons per layer
        activation (str): The activation function used by the model
        dropout (float): Determines the percentage amount of neurons which will deactivated during training (prevents overfitting)
        lambda (float): The Lambda value for your regularisation (L1 and L2).
    returns:
        neuronalNetworkModel (NeuronalNetworkModel): A new NeuronalNetworkModel instance
    """
    return nwm.NeuralNetworkModel(units, activation, dropout, lamda)
