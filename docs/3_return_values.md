# Return Values

In case Hyperparameter Optimization is activated:

- After execution there will be a folder created which name is specified under the `HyperparameterSavePathConfig.py` file.
- In this folder you will find the project by the name you gave threw the `--project_name` variable (the default name is under `HyperparameterSavePathConfig.py` as well).
- In this folder you will find a trial.json for every trial which was executed.
- This trial.json holds information about the used hyperparameter for this trial and the results of the loss value & metrics values for the model in train mode.
- On top of that the loss value & metrics values for the model in test mode will be returned by the pipeline.

In case Hyperparameter Optimization is deactivated:
- Only the loss value & metrics values for the model in test mode will be returned by the pipeline.