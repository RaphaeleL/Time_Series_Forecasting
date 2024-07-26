# Model Import 

- The Pipeline gives you the option to import your own model for testing.
- Some values of the model are optimized over the hyperparameter optimization.
  - "Units", "dropout" and "activation function" are examples here.
  - If you want these parameters to be optimized in your model as well, then you need to insert these variables into your model (Detailed example at the end of this chapter).
  - If you want a special values for certain parameters in your model you can just define these yourself and ignore these parameters.

Following parameters are affecting your model directly:

```bash
"units", type=Tuple(int)/int, help="UNITS"
```

Amounts of neurons in the layers.

```bash
"activation", type=list(str)/str, help="ACTIVATION"
```

The [activation function](https://en.wikipedia.org/wiki/Activation_function) of the network.

```bash
"dropout", type=Tuple(float)/float, help="DROPOUT"
```

The dropout determines how many neurons will be deactivated during training. The higher the dropout rate the more neurons will be deactivated. This prevents overfitting.

```bash
"lamda", type=Tuple(float)/float, help="LAMBDA"
```

The lambda value for your regularisation (L1 and L2). 

---

Example how to use these parameters in combination with your model:

```bash
Sequential([
            Bidirectional(LSTM(units, activation=activation,  return_sequences=True)),
            Bidirectional(LSTM(units, activation=activation, return_sequences=True)),
            Dropout(dropout),
            Bidirectional(LSTM(units, activation=activation, return_sequences=True)),
            Bidirectional(LSTM(units, activation=activation, return_sequences=True)),
            Dropout(dropout),
            Bidirectional(LSTM(units, activation=activation, return_sequences=True)),
            Bidirectional(LSTM(units, activation=activation, return_sequences=True)),
            Dropout(dropout),
            Flatten(),
            Dense(50, activation=activation, kernel_regularizer=l2(lamda)),
            Dense(1),
        ])
```

- Replace the actual values of the model with the parameter names.
  - Replace Units with the units parameter, `l2(value)` with the lamda parameter etc.
- Copy and paste your model for the `nn` variable in the `NeuralNetworkModel` Class or use the `set_model` Method in the `NeuralNetworkModel` Class.
  - Keep in mind that your model needs to be of type `keras.Sequential`.