import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import csv
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.init as init
import random
import torch.optim as optim
import logging
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import ray
# from our own modules
from preprocess_data_nn import PreprocessData
from transform_data_nn import TransformData
from create_dataset_nn import XandYDataset
from model_data_nn import ModelData


### Initial setup
# Working directory
project_dir = '/zfs/projects/darc/wolee_edehaan_suzienoh-exploratory-ml'
os.chdir(project_dir)

# Logging
logging.basicConfig(level=logging.WARNING)
file_name = os.path.basename(__file__)
logger = logging.getLogger(file_name)
logger.setLevel(level=logging.DEBUG)

# Declare global variables
global continuous_vars
global binary_vars
global embed_vars
global header

# List of continuous variables
continuous_vars = ['absacc', 'acc', 'aeavol', 'age', 'agr', 'baspread', 'beta', 
            'betasq', 'bm', 'bm_ia', 'cash', 'cashdebt', 'cashpr','cfp', 
            'cfp_ia', 'chatoia', 'chcsho', 'chempia', 'chfeps', 'chinv', 
            'chmom', 'chnanalyst', 'chpmia', 'chtx', 'cinvest', 'currat', 
            'depr', 'disp', 'dolvol', 'dy', 'ear', 'egr', 'ep', 'fgr5yr', 
            'gma', 'grcapx', 'grltnoa', 'herf', 'hire', 'idiovol', 'ill', 
            'indmom', 'invest', 'lev', 'lgr', 'maxret', 'mom12m', 'mom1m', 
            'mom36m', 'mom6m', 'ms', 'mve', 'mve_ia', 'nanalyst', 'nincr', 
            'operprof', 'orgcap', 'pchcapx_ia', 'pchcurrat', 'pchdepr', 
            'pchgm_pchsale', 'pchquick', 'pchsale_pchinvt', 'pchsale_pchrect', 
            'pchsale_pchxsga', 'pchsaleinv', 'pctacc', 'pricedelay', 'ps', 
            'quick', 'rd_mve', 'rd_sale', 'realestate', 'retvol', 'roaq', 
            'roavol', 'roeq', 'roic', 'rsup', 'salecash', 'saleinv', 
            'salerec', 'secured', 'sfe', 'sgr', 'sp', 'std_dolvol', 
            'std_turn', 'stdacc', 'stdcf', 'sue', 'tang', 'tb', 'turn', 
            'zerotrade']


# List of binary variables
binary_vars = ['convind', 'divi', 'divo', 'ipo', 'rd', 'securedind', 'sin']

# List of embedding variables
embed_vars = ['permno']

# Headers
header = ['permno','pyear']

def main():
    
    ### Load and preprocess the data ###
    
    # Create a preprocess data instance
    infile_path = 'Info Processing and Mutual Funds/masterv14.csv'
    period = 'month'
    preprocessor = PreprocessData(infile_path, period)
    
    # Load and preprocess the data
    df = preprocessor.load_and_preprocess_data()
    logger.info(f'preprocessed df: {df.shape}')
    
    # Apply secondary preprocessing
    df = preprocessor.apply_secondary_preprocessing()
    logger.info(f'secondary preprocessed df: {df.shape}')
    logger.info(f'df sample: {df.head()}')

    ### Transform the data ###
    
    # Get train_data, test_data, retrain_data, and prediction_data
    logger.info(f'\n\nTransform data\n')
    transformer = TransformData(
        train_year_start=1980, 
        prediction_year=1988, 
        df=df,
        period=period,
        continuous_vars=continuous_vars, 
        binary_vars=binary_vars, 
        embed_vars=embed_vars, 
        header=header,
        year_col='pyear'
    )
    
    train_data = transformer.get_train_data()
    test_data = transformer.get_test_data()
    retrain_data = transformer.get_retrain_data()
    prediction_data = transformer.get_prediction_data()
    
    # Build a pipeline
    transformer.build_pipeline(lower_percentile=5, upper_percentile=95)
    logger.debug(f'Pipeline built: {transformer.pipeline}')
    
    # Generate X and y with train_data and test_data
    logger.info(f'\n\nGenerate X and y with train_data and test_data\n')
    features = transformer.independent_vars
    target = transformer.target
    pipeline = transformer.pipeline
    
    x_train_tf, y_train_tf, x_test_tf, y_test_tf = transformer.transform_data(
        train_data=train_data, 
        test_data=test_data,
        features=features,
        target=target,
        pipeline=pipeline,
    )
    
    logger.info(f'''
        x_train_tf: {x_train_tf.shape}
        y_train_tf: {y_train_tf.shape}\n
        x_test_tf: {x_test_tf.shape}
        y_test_tf: {y_test_tf.shape}\n
    '''
    )
    
    # Generate X and y with retrain_data and prediction_data
    logger.info(f'\n\nGenerate X and y with retrain_data and prediction_data\n')
    x_retrain_tf, y_retrain_tf, x_prediction_tf, y_prediction_tf = transformer.transform_data(
        train_data=retrain_data, 
        test_data=prediction_data,
        features=features,
        target=target,
        pipeline=pipeline,
    )
    
    logger.info(f'''
        x_retrain_tf: {x_retrain_tf.shape}
        y_retrain_tf: {y_retrain_tf.shape}\n
        x_prediction_tf: {x_prediction_tf.shape}
        y_prediction_tf: {y_prediction_tf.shape}\n
    '''
    )
    
    
    ### Create dataset ###
    
    continuous_len = transformer.continuous_len
    
    # Train dataset
    logger.info(f'Create train_dataset')
    train_dataset = XandYDataset(
        X_continuous_vars=x_train_tf[:, :continuous_len], 
        X_embedding_vars=x_train_tf[:, continuous_len:], 
        y=y_train_tf
    )
    logger.info(f'train_dataset first example: {train_dataset[0]}')
    
    # Test dataset
    logger.info(f'Create test_dataset')
    test_dataset = XandYDataset(
        X_continuous_vars=x_test_tf[:, :continuous_len], 
        X_embedding_vars=x_test_tf[:, continuous_len:], 
        y=y_test_tf
    )
    logger.info(f'test_dataset first example: {test_dataset[0]}')
    
    # Retrain dataset
    logger.info(f'Create retrain_dataset')
    retrain_dataset = XandYDataset(
        X_continuous_vars=x_retrain_tf[:, :continuous_len], 
        X_embedding_vars=x_retrain_tf[:, continuous_len:], 
        y=y_retrain_tf
    )
    logger.info(f'retrain_dataset first example: {retrain_dataset[0]}')
    
    # Prediction dataset
    logger.info(f'Create prediction_dataset')
    prediction_dataset = XandYDataset(
        X_continuous_vars=x_prediction_tf[:, :continuous_len], 
        X_embedding_vars=x_prediction_tf[:, continuous_len:], 
        y=y_prediction_tf
    )
    logger.info(f'prediction_dataset first example: {prediction_dataset[0]}')
    
    ### Create dataloader ###
    
    torch.manual_seed(42)
    batch_size = 32
    
    # Train and test dataloader
    logger.info(f'train and test dataloader')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    logger.info(f'train_loader first example: {next(iter(train_loader))}\n')
    logger.info(f'test_loader first example: {next(iter(test_loader))}\n')
    
    # Retrain and prediction dataloader
    logger.info(f'retrain and prediction dataloader')
    retrain_loader = DataLoader(retrain_dataset, batch_size=batch_size, shuffle=True)
    prediction_loader = DataLoader(prediction_dataset, batch_size=batch_size, shuffle=False)
    logger.info(f'retrain_loader first example: {next(iter(retrain_loader))}\n')
    logger.info(f'prediction_loader first example: {next(iter(prediction_loader))}\n')
    
    ### Model the data and tune hyperparameters ###
    
    logger.info(f'Hyperparameters tuning with Ray Tune')
    logger.info(f'Training data years: {transformer.train_years}')
    logger.info(f'Testing data year: {transformer.test_year}')
    ray_results_path = "/zfs/projects/darc/wolee_edehaan_suzienoh-exploratory-ml/kevin/ray_results"
    num_samples=40
    max_num_epochs=30
    num_cpus = 10
    num_gpus=0
    # Initialize Ray
    ray.init(num_cpus=num_cpus, num_gpus=num_gpus)
    
    # Hyperparameter tuning with Ray Tune
    data_modeler = ModelData(ray_results_path=ray_results_path, verbose=3)
    best_trial = data_modeler.get_best_trial(
        train_loader=train_loader, 
        test_loader=test_loader, 
        num_samples=num_samples, 
        max_num_epochs=max_num_epochs, 
        gpus_per_trial=num_gpus,
    )
    logger.info(f'Ray Tune results have been saved to: {ray_results_path}')
    
    ### Retrain the model with optimized hyperparameter using retrain_data ###
    
    logger.info(f'''/n/nRetrain a new model with data in years: {transformer.retrain_years}\n
        Using the optimized hyperparameters: {best_trial.config}/n''')
    trained_model = data_modeler.train_fnn(
        config=best_trial.config, 
        train_loader=retrain_loader, 
        test_loader=prediction_loader, 
        ray_tuning=False,
    )
    
    ### Prediction ###
    
    # Make predictions
    logger.info(f'Making prediction for data in year: {transformer.test_year}')
    predictions = data_modeler.predict(trained_model, prediction_loader)
    logger.info(f'Prediction data shape: {predictions.shape}')
    prediction_data['pred'] = predictions
    
    # Calculate final prediction performance
    mae = mean_absolute_error(prediction_data[transformer.target], prediction['pred'])
    logger.info(f'Mean Absolute Error: {mae}')
    logger.info(f"Prediction Stats: {prediction_data[[transformer.target, 'pred']].describe()}")

if __name__ == '__main__':
    main()
    