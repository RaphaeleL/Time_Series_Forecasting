from adapters.controllers import ModelTrainingController
from helpers import ArgumentParserBuilder

"""
Filename: Main.py
Created at: 2024-05-17

Description:
The Main class. Initiates the Argparser and calls the ModelTrainingController class. Main is a placeholder for a future Microservice Implementation. With giving APIs to the Controller classes, main becomes redundant.

"""


def main(args):
    return ModelTrainingController.execute_model_training(
        args.csv,
        args.inputlenght,
        args.predictionlenght,
        args.epochs,
        args.batchsize,
        args.testsize,
        args.hyper_optimizer,
        args.nsplits,
        args.objective,
        args.maxtrials,
        args.should_optimize,
        args.units,
        args.activation,
        args.dropout,
        args.lamda,
        args.metrics,
        args.model_optimizer,
        args.loss,
        args.learning_rate,
        args.validation_split,
        args.project_name,
    )


if __name__ == "__main__":
    initial_parser = ArgumentParserBuilder.build_should_optimize_parser()
    initial_args, _ = initial_parser.parse_known_args()
    parser = ArgumentParserBuilder.build_main_parser(initial_args.should_optimize)
    main(parser.parse_args())
