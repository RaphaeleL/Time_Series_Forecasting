import argparse

from configs import (
    HyperparameterSavePathConfig as Hpspc,
    ChoicesForActivationFunctionConfig as Cfafc,
    ChoicesForHyperOptimizerConfig as Cfhoc,
    ChoicesForMetricsConfig as Cfmc,
    ChoicesForObjectiveConfig as Cfoc,
    ChoicesForModelOptimizerConfig as Cfmoc,
    ChoicesForLossConfig as Cflc,
)

"""
Filename: ArgumentParserBuilder.py
Created at: 2024-06-03

Description:
Initializes the argument parser for parsing parameters over the command line interface(cli)

"""


def build_should_optimize_parser():
    """
    Initializes the initial parser which does only read out should_optimize

    returns:
        the initial parser for should optimize
    """
    initial_parser = argparse.ArgumentParser()

    initial_parser.add_argument(
        "--should_optimize",
        action="store_true",
        help="Hyperparameter optimization if true, only training if false (without hpo)",
    )

    return initial_parser


def build_main_parser(should_optimize):
    """
    Initializes the main Parser

    args:
        should_optimize (boolean): this boolean switches certain parameters from lenght 3 to 1 and vise versa. Hyperparamter optimization needs length 3 without hpo those parameters need length 1
    returns:
        the main parser
    """

    if should_optimize:
        nargs_lenght = 3
    else:
        nargs_lenght = 1

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        type=str,
        default="exampleData/univariant_example.csv",
        help="path to csv data as string",
    )
    parser.add_argument(
        "--inputlenght",
        type=int,
        required=False,
        default=10,
        help="lenght of features as int",
    )
    parser.add_argument(
        "--predictionlenght",
        type=int,
        required=False,
        default=1,
        help="lenght of labels as int",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        nargs=nargs_lenght,
        default=[5],
        help="amount of epochs as int",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        nargs=nargs_lenght,
        default=[4],
        help="amount of batches as int",
    )
    parser.add_argument(
        "--testsize",
        type=float,
        required=False,
        default=0.2,
        help="the size of the test data cutoff",
    )
    parser.add_argument(
        "--hyper_optimizer",
        type=str,
        required=False,
        choices=Cfhoc.choices_for_optimizer,
        default="RandomSearch",
        help="the hyperparameter optimization method",
    )
    parser.add_argument(
        "--nsplits",
        type=int,
        required=False,
        default=2,
        help="number of splits for validation",
    )
    parser.add_argument(
        "--objective",
        type=str,
        required=False,
        choices=Cfoc.choices_for_objective,
        default="val_loss",
        help="the objective which will be optimized during the training",
    )
    parser.add_argument(
        "--maxtrials",
        type=int,
        required=False,
        default=100,
        help="the number of different possible hyperparameter combinations",
    )
    parser.add_argument(
        "--should_optimize",
        action="store_true",
        help="Hyperparameter optimization if true, only training if false (without hpo)",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        choices=Cfmc.choices_for_metrics,
        default=["RootMeanSquaredError"],
        help="the metrics which will be calculated during training (like root mean square error or F1Score)",
    )
    parser.add_argument(
        "--units",
        type=int,
        nargs=nargs_lenght,
        default=[10],
        help="The amount of neutrons per layer in the neural network",
    )
    parser.add_argument(
        "--activation",
        type=str,
        nargs="+",
        choices=Cfafc.choices_for_activation,
        default=["relu"],
        help="the activation function of the neural network",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        nargs=nargs_lenght,
        default=[0.1],
        help="determines how many neurons will be deactivated during training",
    )
    parser.add_argument(
        "--lamda",
        type=float,
        nargs=nargs_lenght,
        default=[0.001],
        help="The Lambda value for your regularisation (L1 and L2)",
    )
    parser.add_argument(
        "--model_optimizer",
        type=str,
        choices=Cfmoc.choices_for_model_optimizer,
        default="Adam",
        help="the optimizer determines how the model will be optimated during training",
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=Cflc.choices_for_loss,
        default="mse",
        help="detemines on what metric the model will optimize during training",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        nargs=nargs_lenght,
        default=[1e-2],
        help="the learning rate of the training",
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.2,
        help="the size of the validation data during model training without hpo",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default=Hpspc.DEFAULT_FOLDER,
        help="name for the folder the hyperparameter optimization preset will be stored in",
    )

    return parser
