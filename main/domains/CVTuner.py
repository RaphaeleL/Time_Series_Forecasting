import keras_tuner
import numpy as np

from sklearn import model_selection

"""
Filename: CVTuner.py
Created at: 2024-06-15

Description:
The CVTuner. Own Implementation of the "kerastuner.engine.tuner.Tuner" tuner class.

"""


class CVTuner(keras_tuner.Tuner):
    """
    The CVTuner. Own Implementation of the "kerastuner.engine.tuner.Tuner" tuner class. Read more about keras_tuner under following link:
    https://keras.io/keras_tuner/
    """

    def __init__(self, hypermodel, oracle, nsplits, objective, directory, project_name):
        """
        init for CVTuner. Calls super constructor, only nsplits is saved in this class

        args:
            hypermodel (HyperModel): The hypermodel which holds the hyperparameter configuration
            oracle (Oracle): Optimizer of type keras_tuner.Oracle
            nsplits (int): number of splits for the cross validation
            directory (str): Determines the directory in which the hyperparameters will be stored after training. This prevents redundant training if there is no configuration change
            project_name (str): Determines the file in which the hyperparameters will be stored after training. This prevents redundant training if there is no configuration change

        returns:
            cvTuner (CVTuner): The tuner for the hpo
        """
        super().__init__(
            hypermodel=hypermodel,
            oracle=oracle,
            directory=directory,
            project_name=project_name,
        )
        self.nsplits = nsplits
        self.objective = objective

    def run_trial(self, trial, x, y, batch_size=None, epochs=None):
        """
        overwritten run_trial function of Tuner. The keras_Tuner classes do not support cross validation. Therefore it is implemented in this overwritten run_trial function
        1. Cross validation with call on original run_trial function of keras_tuner.tuner, which returns a list of history objects
        2. Read out the values of the objective from the history objects.
        3. Only the objective value at the last epoch is used here
        4. Update The trial objective value by the mean of every objective value at the last epoch of each split
        5. Return the history objects the same way like the original run_trial method from keras_tuner.tuner would have returned them.

        Note: TimeSeriesSplit is repeated for every trial. TSP is not very performance hungry but for later performance improvement TSP could be outsourced of this class!

        args:
            trial (Trial): The current trial
            x: The featureset of the trial
            y: The labelset of the trial
            batchsize: Not used only there for overwriting reasons
            epochs: Not used only there for overwriting reasons

        returns:
            histories (list(history)): The history objects which holds information about parameters, epochs, and metrics.
        """

        """
      print("")
      print("Training split", idx)
      model = self.hypermodel.build(trial.hyperparameters)
      self.hypermodel.fit(trial.hyperparameters, model, x_train, y_train)
      print("")
      print("Valuation split", idx)
      val_losses.append(model.evaluate(x_val, y_val)[0])
      idx += 1
    self.oracle.update_trial(trial.trial_id, {'val_loss': np.mean(val_losses)})
    self.save_model(trial.trial_id, model)
    """

        # cross validation after expanding window approach!
        cv = model_selection.TimeSeriesSplit(n_splits=self.nsplits)

        histories_list = []
        idx = 1
        for train_indices, val_indices in cv.split(x):
            print("Current split: %s of %s splits." % (idx, self.nsplits))
            print("")
            x_train, x_val = x[train_indices], x[val_indices]
            y_train, y_val = y[train_indices], y[val_indices]
            histories_list.append(
                super().run_trial(
                    trial, x_train, y_train, validation_data=(x_val, y_val)
                )
            )  # 1
            print(
                "____________________________________________________________________________________"
            )
            print("")
            idx += 1

        validation_metrics_list = []
        for histories in histories_list:
            validation_metrics_list.append(
                {
                    key: value
                    for key, value in histories[0].history.items()
                    if key.startswith(self.objective)
                }  # 2
            )

        val_losses = []
        for metrics in validation_metrics_list:
            val_losses.append(metrics[self.objective][-1])  # 3

        self.oracle.update_trial(
            trial.trial_id, {self.objective: np.mean(val_losses)}
        )  # 4

        histories_list = np.array(histories_list).flatten()
        return histories_list.tolist()  # 5
