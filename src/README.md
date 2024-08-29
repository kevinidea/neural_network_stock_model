# Neural Network Model Training

This repository contains scripts and instructions for training neural network models on various datasets. The models can be trained on either monthly or quarterly data, with both new and restricted variable sets available.

## Prerequisites

Before running the scripts, ensure that you have set the correct working directory:

```bash
cd /zfs/projects/darc/wolee_edehaan_suzienoh-exploratory-ml/kevin/
```

## Training the Model

To train the model, submit the appropriate SLURM job script based on the dataset and variable set:

### Monthly Data

- **New Variables**:
  ```bash
  sbatch src/neural_network_month.slurm
  ```

- **Restricted Variables**:
  ```bash
  sbatch src/neural_network_month_restricted.slurm
  ```

### Quarterly Data

- **New Variables**:
  ```bash
  sbatch src/neural_network_quarter.slurm
  ```

- **Restricted Variables**:
  ```bash
  sbatch src/neural_network_quarter_restricted.slurm
  ```

### Modifying SLURM Script Parameters

Each SLURM script contains several configurable parameters:

```bash
#SBATCH -J NN_month                     # Job name
#SBATCH -p long                         # Partition (queue) to submit to
#SBATCH -G 0                            # Number of GPUs (Max: 4), only use GPU in gpu partition
#SBATCH -c 26                           # Number of CPU cores
#SBATCH --mem=160G                      # Memory allocation
#SBATCH -t 7-                           # Maximum job time (7 days)
#SBATCH -o '/zfs/projects/darc/wolee_edehaan_suzienoh-exploratory-ml/kevin/output/log/monthly_new_vars_%j.out'  # Output log path
#SBATCH --mail-type=ALL                 # Email notifications
#SBATCH --mail-user=kevin128@stanford.edu  # Your email
```

## Python Environment

The Python virtual environment for this project is located at:

```
/zfs/projects/darc/wolee_edehaan_suzienoh-exploratory-ml/kevin/venv
```

To activate the environment, use the following command:

```bash
source /zfs/projects/darc/wolee_edehaan_suzienoh-exploratory-ml/kevin/venv/bin/activate
```

## Main Script

The primary Python scripts used for training are `main.py` and `main_restricted.py`. These scripts accept several command-line arguments to configure the training process:

```python
parser = argparse.ArgumentParser(description='Building Neural Network Model')

parser.add_argument('--infile_path', type=str, required=True,
                    help='The input file path')
parser.add_argument('--period', type=str, default='month',
                    help='Period: either month or quarter')
parser.add_argument('--prediction_parent_path', type=str, required=True,
                    help='The parent path for prediction files.')
parser.add_argument('--num_samples', type=int, default=32,
                    help='Number of trials to run with Ray Tune')
parser.add_argument('--max_num_epochs', type=int, default=20,
                    help='Maximum number of epochs')
parser.add_argument('--num_cpus', type=int, default=32,
                    help='Number of CPUs to use for Ray Tune')
parser.add_argument('--cpus_per_trial', type=int, default=1,
                    help='Number of CPUs per trial for Ray Tune')
parser.add_argument('--num_gpus', type=int, default=0,
                    help='Number of GPUs to use for Ray Tune and model training')
parser.add_argument('--gpus_per_trial', type=int, default=0,
                    help='Number of GPUs per trial for Ray Tune')
parser.add_argument('--patience', type=int, default=2,
                    help='Early stopping patience (epochs without improvement)')
parser.add_argument('--prediction_years', nargs='*', type=int, required=False, default=[],
                    help='List of prediction years (e.g., 2000 2008)')
```

## Supporting Modules

The project is structured with several supporting modules to handle data preprocessing, transformation, modeling, and post-processing:

- **Data Preprocessing**: 
  ```bash
  preprocess_data_nn.py
  ```
- **Data Transformation**:
  ```bash
  transform_data_nn.py
  ```
- **Modeling**:
  ```bash
  model_data_nn.py
  ```
- **Post-processing Predictions**:
  ```bash
  postprocess_predictions.py
  ```

## Prediction Results
- **Result Summary**:
  ```bash
  prediction_result_summary.ipynb
  ```
- **Detailed Results**:
  All the predictions are saved by default in the directory below. It should contain at least four subdirectories corresponding to the four datasets: `monthly_new_vars`, `monthly_new_restricted`, `quarterly_new_vars`, and `quarterly_new_restricted`.
  ```bash
  /zfs/projects/darc/wolee_edehaan_suzienoh-exploratory-ml/kevin/output/prediction
  ```
  Note that each subdirectory should contain prediction data generated from 1985 to 2020 and a `result.csv` file.

## Log

All the logs, by default, are saved in the directory below. It should contain at least four subdirectories corresponding to the four datasets: `monthly_new_vars`, `monthly_new_restricted`, `quarterly_new_vars`, and `quarterly_new_restricted`.

```
/zfs/projects/darc/wolee_edehaan_suzienoh-exploratory-ml/kevin/output/log
```

## Ray Tune Detailed Results

All hyperparameter tuning results from Ray Tune are saved in the following directory:

```
/zfs/projects/darc/wolee_edehaan_suzienoh-exploratory-ml/kevin/ray_results
```
