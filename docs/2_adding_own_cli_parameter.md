# Adding your own parameters:

- Some parameters have choices which limit the possible parameters you can give as imput.
- There are choices for the activation function, optimnizer of the hyperparameters, optimizer of the model, loss, metrics, and for the objective.
- Every of these 6 different parameters have their own configuration file where all possible parameter values are saved.
- They all begin with Choices + parameter + config (for example: `ChoicesForActivationFunctionConfig`).
- You will find them under "main/configs".
- If you want to add possible parameters to any of these choices you just need to add the string into the lists inside the configs.
- **You need to validate those parameters yourself!**
  - you have to check if every frameworks used in this pipeline supports this parameter.
  - if the parameter is spelled correctly.
  - if the parameter works with neural networks or forecasting problems.
  - etc.