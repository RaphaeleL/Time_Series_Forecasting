# BiFlex - Forecasting

Forecasting load and charging requirements for vehicle fleets in companies.

## Getting Started

You can find a detailed Documentation under `docs/`.

## Usage

### Setup

1. Install Python
   - Windows: https://www.python.org/
   - MacOS: `brew install python`
   - Linux: 
     - Debian/Ubuntu: `sudo apt install python3`
     - Fedora: `sudo dnf install python3`
     - Arch: `sudo pacman -S python`
2. Clone the Repository `git clone https://github.com/DataDrivenSustainabilitySolutions/BiFlex-Forecasting.git`
3. Install the Requirements `pip install -r requirements.txt`

### Execution

With Hyperparameter optimization:

```bash
python main.py \
    --csv 'data/univariant_example.csv'  \
    --input_length 10  \
    --prediction_length 1 \
    --epochs 1 4 1  \
    --batch_size 2 2 2 \ 
    --test_size 0.2 \
    --hyper_parameter_optimizer 'RandomSearch' \
    --n_splits 2 \ 
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

```bash
python main.py \
    --csv 'data/univariant_example.csv'  \
    --input_length 10  \
    --prediction_length 1  \
    --epochs 1  \
    --batch_size 2  \
    --test_size 0.2  \
    --hyper_parameter_optimizer 'RandomSearch'  \
    --n_splits 2  \
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
    --project_name 'project_complete_test_70trials'
```

## Contributing

Contributions are what make the open source community a vibrant space for learning, inspiration, and creativity. Every contribution is deeply valued and appreciated.

## Contact

For further Questions please contact one of them:

- <a href="mailto:raphaele_salvatore.licciardo@h-ka.de"><b>Raphaele Salvatore Licciardo</b> (Researcher at the ISRG)</a> 
- <a href="mailto:hori1011@h-ka.de"><b>Richard Hoepken</b> (Student)</a> 
