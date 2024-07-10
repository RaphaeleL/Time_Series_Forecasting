# Forecasting Pipeline 

In this work, a comprehensive modular deep learning pipeline was designed and developed to be capable of comparing various prediction models. This pipeline is be applicable to different datasets to make accurate forecasts (e.g. forecasting the energy requirements of an industrial operation).

## Setup

 Install requirements.txt.

 ```bash
 pip install -r requirements.txt
 ```

## Execution

With Hyperparameter optimization:

```
python Main/main.py \
    --csv 'exampleData/univariant_example.csv'  \
    --inputlenght 10  \
    --predictionlenght 1 \
    --epochs 1 4 1  \
    --batchsize 2 2 2 \ 
    --testsize 0.2 \
    --hyper_optimizer 'RandomSearch' \
    --nsplits 2 \ 
    --objective 'val_loss' \
    --maxtrials 2 \
    --should_optimize \
    --metrics 'R2Score' \
    --units 10 20 10 \
    --activation 'relu' 'tanh' \
    --dropout 0.0 0.5 0.1 \
    --lamda 0.0005 0.001 0.0015 \
    --model_optimizer 'Adam' \
    --loss 'mse' \
    --learning_rate 1e-4 1e-3 1e-2 \
    --project_name 'project_complete_test_70trials'
```

Without Hyperparameter optimization:

```
python Main/main.py \
    --csv 'exampleData/univariant_example.csv'  \
    --inputlenght 10  \
    --predictionlenght 1  \
    --epochs 1  \
    --batchsize 2  \
    --testsize 0.2  \
    --hyper_optimizer 'RandomSearch'  \
    --nsplits 2  \
    --objective 'val_loss'  \
    --maxtrials 2  \
    --metrics 'R2Score'  \
    --units 10  \
    --activation 'relu'  \
    --dropout 0.0  \
    --lamda 0.0005  \
    --model_optimizer 'Adam'  \
    --loss 'mse'  \
    --learning_rate 1e-4  \
    --validation_split 0.2  \
    --project_name 'project_complete_test_70trials' \
```

## Contact

For further Questions please email one of them:

- Raphaele Salvatore Licciardo
- Richard Höpken
