# Required Input Parameters

- The following chapter describes the parameters to run the pipeline. There are no required parameters, but keep in mind that every parameter holds a default value. So not using a parameter at all is not possible and u should check the default values before ignoring a parameter. 
- Parameter are passed over the command line in the following format:

```bash
--<parametername> <parametervalue>
--csv 'datastorage/data.csv'
```

- multiple parameters can just be written successively divided by a single space.

```bash
--csv 'datastorage/data.csv' --inplen 7 --predlen 1 ...
```

- Some parameters have two possible value types.
  - some values change from a single value to a tuple of size 3 if you are using the hyperparameter optimization.

```bash
--batchsi 16 64 8   # if you are using the hyperparameter optimization
--batchsi 64          # if you are not using the hyperparameter optimization
```

- So its import to keep in mind, if you set parameter "should_optimize" you have to give the tuple variants as input.
- if you do not set "should_optimize" you have give a single input to affected parameters.

```bash
--should_optimize # not using should_optimize sets it to false
--batchsi 16 64 8 
--batchsi 64          
```

- The tuples always have following structure.
  - minimum, maximum, stepsize.
- The only exceptions are the parameters "activation" and "metrics".
  - you have to pass a list of all possible strings here.

```bash
--activation 'relu' 'tanh' ... #this is not limited to a specific length
```

___

```bash
'--csv', type=str, default='data/univariant_example.csv', help='path to csv data as string'
```

- Determines the path to the data your model will be trained and tested on.
- If the datafile is located inside the projektstructure (below the directory "Projektarbeit_RichardHoepken") you only have to input the relativ path to the file.
  - Examplepath: datastorage/data.csv.
- If the datafile is located outside the projectstructure you have to give the absolute path as input.
  - Examplepath: C:\Users\admin\Documents\data.csv.
- Make sure that the datafile is a valid csv file and that it sticks to the required format.
  - The format is described in this readme in the chapter **Data Structure**.

```bash
'--input_length', type=int, required=False, default=7, help='length of features as int'
'--prediction_length', type=int, required=False, default=1, help='length of labels as int'
```

- The input and prediction length are more or less your feature label equivalent. The Model will take i Datapoints (i = Input length) as features and will take p datapoints (p = Prediction length) as label.
- Input length i.
  -  the i datapoints will consist of all features in the given dataset.
- Prediction length p.
  - the p datapoints will only consist of the **first** feature in the dataset.
- Because the "label" for the forecasting is generated only of the first feature, you need to make sure that the first feature of the dataset is the feature you actualy want a prediction of.
  - read more about data structure in the chapter **Data Structure**.
- Example:
  - Data: {[1, Monday], [2, Tuesday], [3, Wednesday], [4, Thursday], [5, Friday], [6, Saturday], [7, Sunday]}
  - input_length = 3, prediction_length = 1
  - Result:
    - X = {([1, Monday], [2, Tuesday], [3, Wednesday]), ([2, Tuesday], [3, Wednesday], [4, Thursday]), ([3, Wednesday], [4, Thursday], [5, Friday]), ...}
    - y = {4, 5, 6, ...}

```bash
'--epochs', type=int, nargs=nargs_length, default=[5],  help='amount of epochs as int'
```

Sets the amount of epochs for the training. If you set this number for example to 20, the training will perform 20 different training epochs, for each split, before chosing new hyperparameters.

```bash
"batchsi", type=Tuple(int, int, int)/int, help="BATCH_SIZE"
```

The batch size determines the amount of batches your training data will be divided into. If you have a batch size of 128 for example your training data will be devided into 128 different batches for training.

```bash
'--batch_size', type=int, nargs=nargs_length, default=[16], help='amount of batches as int'
```

- Determines the ratio between training and test data.
- Sets the size of the testdata to given value.
- The rest of the data then becomes trainingdata automaticly.
- If you pick a test size of 0.2 (20%) for example you will have 20% test data and 80% training data.

```bash
'--test_size', type=float, required=False, default=0.2, help='the size of the test data cutoff'
```

- Determines the size of the testdata which will be used for the final evaluation of the model
- its a percentage number. 0.2 means 20% of the whole dataset will be reserver for the final test

```bash
'--hyper_parameter_optimizer', type=str, required=False, choices=Cfhoc.choices_for_optimizer, default='RandomSearch', help='the hyperparameter optimization method'
```

Determines the optimizer you want to use for your hyperparameter optimization.

```bash
'--n_splits', type=int, required=False, default=5, help='number of splits for validation'
```

- Determines the amount of validation splits for the hyperparameter optimization.
- The amount of splits automaticly sets the ratio between validation data and training data.
- More splits mean a smaller amount of validation data per split.
- 
```bash
'--objective', type=str, required=False, choices=Cfoc.choices_for_objective, default='val_loss', help='the objective which will be optimized during the training'
```

Determines the objective the hyperparameter optimization has to optimize after. if you pick "val_loss" for example, the optimizer will try to make the loss on the valuation data as small as possible.

```bash
'--maxtrials', type=int, required=False, default=100, help='the number of different possible hyperparameter combinations'
```

The total number of trials (model configurations) to test at most. If you pick 60 for example the hyperparameter optimization will try max 60 different configurations for your hyperparameters until it will pick the best configuration.

```bash
'--should_optimize', action='store_true', help='Hyperparameter optimization if true, only training if false (without hpo)'

--should_optimize #it is true now
# if you dont use it it will be false
```

- A boolean which activates/deactivates the hyperparamter optimization.
- This parameter is a flag! Which means if you want to use the HyperParameterOptimization, you only have to pass the name of the parameter, if you dont wanna use it then ignore it.

```bash
'--metrics', type=str, nargs='+', choices=Cfmc.choices_for_metrics, default=['F1Score'], help='the metrics which will be calculated during training (like root mean square error or F1Score)'
```

A list of all metrics you want to measure the performance of you model with. The individual metrics will be displayed with each training/test step.

```bash
'--units', type=int, nargs=nargs_length, default=[100], help='The amount of neutrons per layer in the neural network'
```

Amounts of neurons in the layers. If you have a specific neuron layout dont use this variable.

```bash
'--activation', type=str, nargs='+', choices=Cfafc.choices_for_activation, default=['relu'], help='the activation function of the neural network'
```

The [activation function](https://en.wikipedia.org/wiki/Activation_function) of the network.

```bash
'--dropout', type=float, nargs=nargs_length, default=[0.1], help='determines how many neurons will be deactivated during training'
```

The dropout determines how many neurons will be deactivated during training. The higher the dropout rate the more neurons will be deactivated. This prevents overfitting. If you set the dropout too low you may have high overfitting, is the dropout to high you my have high underfitting.

```bash
'--lamda', type=float, nargs=nargs_length, default=[0.001], help='The Lambda value for your regularisation (L1 and L2)'
```

The Lambda value for your regularisation (L1 and L2). 

```bash
'--model_optimizer', type=str, choices=Cfmoc.choices_for_model_optimizer,  default='Adam', help='the optimizer determines how the model will be optimated during training'
```

- The model optimizer determines how the model will be trained.
- Currently Adam is the only one possible, so you can ignore this parameter aslong as you dont want to add any optimizers.

```bash
'--loss', type=str, choices=Cflc.choices_for_loss,  default='mse', help='detemines on what metric the model will optimize during training'
```

The loss function. Same as with the model optimizer there is just one right now which is 'mse'. So you can ignore this parameter aslong as you dont want to add any loss functions.

```bash
'--learning_rate', type=float, nargs=nargs_length, default=[1e-2], help='the learning rate of the training'
```

The learning rate determines how fast the model will optimize the model parameters. A higher value means that the training will be done quicker but you maybe will get a worse result than using a lower learning rate.

```bash
'--validation_split', type=float, default=None, help='the size of the validation data during model training without hpo'
```

The validation_split is only for normal training without hyperparamterOptimization. It determines the amount of valuation Data which is used during the model training. Like with test size this is a percentage value.

```bash
'--project_name', type=str, default=Hpspc.DEFAULT_FOLDER, help='name for the folder the hyperparameter optimization preset will be stored in'
```

This will be the name of your hyperparameterOptimizationResults. You can look the results up under "Projektarbeit_RichardHoepken/hyperparameter_optimization_presets/<project_name>".
