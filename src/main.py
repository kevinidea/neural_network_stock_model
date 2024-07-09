import os
from sklearn.metrics import root_mean_squared_error
import torch
from torch.utils.data import DataLoader
import logging
import ray
import sys
import argparse

### Initial setup

# Working directory
project_dir = '/zfs/projects/darc/wolee_edehaan_suzienoh-exploratory-ml'
os.chdir(project_dir)

# Append the right path to sys.path to import our own modules
src_dir = os.path.join(project_dir, 'kevin/src')
sys.path.append(src_dir)

# Import our own modules
from preprocess_data_nn import PreprocessData
from transform_data_nn import TransformData
from create_dataset_nn import XandYDataset
from model_data_nn import ModelData

# Speed up sklearn
from sklearnex import patch_sklearn
patch_sklearn()

# Logging
logging.basicConfig(level=logging.WARNING)
file_name = os.path.basename(__file__)
logger = logging.getLogger(file_name)
logger.setLevel(level=logging.INFO)

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
    
    ### Parsing arguments ###
    
    parser = argparse.ArgumentParser(description='Building Neural Network Model')
    
    parser.add_argument('--infile_path', type=str, default='Info Processing and Mutual Funds/masterv14.csv',
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

    args = parser.parse_args()
    
    ### Load and preprocess the data ###
    
    # Create a preprocess data instance
    preprocessor = PreprocessData(args.infile_path, args.period)
    
    # Load and preprocess the data
    df = preprocessor.load_and_preprocess_data()
    logger.info(f'preprocessed df: {df.shape}')
    
    # Apply secondary preprocessing
    df = preprocessor.apply_secondary_preprocessing()
    logger.info(f'secondary preprocessed df: {df.shape}')
    logger.info(f'df sample: {df.head()}')

    ### Transform the data ###
    
    # Get all predictio data years
    train_year_start = df['pyear'].drop_duplicates().min()
    prediction_years = list(df['pyear'].drop_duplicates().sort_values())[5:]
    logger.info(f'Train year start: {train_year_start}')
    logger.info(f'Prediction data years: {prediction_years}')
    
    # train_year_start = 1980
    # prediction_year = 1988
    
    # Loop through all the prediction years and build optimized model for each year
    logger.info(f'\n\nLoop through all the prediction years and build optimized model for each year\n')
    
    for prediction_year in prediction_years:
        # Get train_data, test_data, retrain_data, and prediction_data
        logger.info(f'\n\nTransform data\n')
        transformer = TransformData(
            train_year_start=train_year_start,
            prediction_year=prediction_year,
            df=df,
            period=args.period,
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

        # Generate X and y with train_data, test_data, retrain and prediction data
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

        logger.info(f'\n\nCreate dataset\n')

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

        logger.info(f'\n\nCreate dataloader\n')

        ModelData.set_seed(42)
        # Batch size has a big impact to performance, smaller seems to yield lower loss
        batch_size = 256

        # Train and test dataloader
        logger.info(f'Create train and test dataloader')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        logger.debug(f'train_loader first example: {next(iter(train_loader))}\n')
        logger.debug(f'test_loader first example: {next(iter(test_loader))}\n')

        # Retrain and prediction dataloader
        logger.info(f'Create retrain and prediction dataloader')
        retrain_loader = DataLoader(retrain_dataset, batch_size=batch_size, shuffle=True)
        prediction_loader = DataLoader(prediction_dataset, batch_size=batch_size, shuffle=False)
        logger.debug(f'retrain_loader first example: {next(iter(retrain_loader))}\n')
        logger.debug(f'prediction_loader first example: {next(iter(prediction_loader))}\n')

        ### Model the data and tune hyperparameters ###

        logger.info(f'\n\nHyperparameters tuning with Ray Tune\n')
        logger.info(f'Training data years: {transformer.train_years}')
        logger.info(f'Testing data year: {transformer.test_year}')
        ray_results_path = "/zfs/projects/darc/wolee_edehaan_suzienoh-exploratory-ml/kevin/ray_results"
        num_samples = args.num_samples
        max_num_epochs = args.max_num_epochs
        num_cpus = args.num_cpus
        cpus_per_trial = args.cpus_per_trial
        num_gpus = args.num_gpus
        gpus_per_trial = args.gpus_per_trial
        continuous_dim = transformer.continuous_len
        num_embeddings = train_data['permno'].nunique()
        # Important to set the device because it will be frequently used
        device = torch.device("cuda" if num_gpus > 0 else "cpu")

        logger.info(
            f'''\n
            ray_results_path: {ray_results_path}
            num_samples: {num_samples}
            max_num_epochs: {max_num_epochs}
            num_cpus: {num_cpus}
            cpus_per_trial: {cpus_per_trial}
            num_gpus: {num_gpus}
            gpus_per_trial: {gpus_per_trial}
            continuous_dim: {continuous_dim}
            num_embeddings: {num_embeddings}
            device: {device}
            '''
        )

        # Initialize Ray
        ray.init(
            num_cpus=num_cpus, 
            num_gpus=num_gpus,
            runtime_env={"working_dir": src_dir},
        )

        # Hyperparameter tuning with Ray Tune
        data_modeler = ModelData(ray_results_path=ray_results_path, verbose=0)

        best_trial = data_modeler.get_best_trial(
            train_loader=train_loader,
            test_loader=test_loader,
            continuous_dim=continuous_dim,
            num_embeddings=num_embeddings,
            device=device, # CPUs seem to be faster than GPUs because of more parellel processing
            num_samples=num_samples,
            max_num_epochs=max_num_epochs,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            cpus_per_trial=cpus_per_trial,
            gpus_per_trial=gpus_per_trial,
        )
        logger.info(f'Ray Tune results have been saved to: {ray_results_path}')
        logger.info(f'Best trial directory: {best_trial.local_path}')

        ### Retrain the model with optimized hyperparameter using retrain_data ###

        best_config = best_trial.config
        # Overide the num_embeddings with retrain_data
        best_config['num_embeddings'] = retrain_data['permno'].nunique()

        logger.info(f'''\n\nRetrain a new model with data in years: {transformer.retrain_years}\n
            Using the optimized hyperparameters: {best_config}\n''')
        trained_model = data_modeler.train_fnn(
            config=best_config, 
            train_loader=retrain_loader, 
            test_loader=prediction_loader,
            device=device,
            ray_tuning=False,
        )

        ray.shutdown()

        ### Prediction ###

        # Make predictions
        logger.info(f'Making prediction for data in year: {prediction_year}')
        predictions = data_modeler.predict(trained_model, prediction_loader, device)
        logger.info(f'Prediction data shape: {predictions.shape}')
        prediction_data['pred'] = predictions

        # Calculate final prediction performance
        rmse = root_mean_squared_error(prediction_data[transformer.target], prediction_data['pred'])
        logger.info(f'Root Mean Squared Error for Prediction in {prediction_year}: {rmse}')
        logger.info(f"Prediction Stats: {prediction_data[[transformer.target, 'pred']].describe()}")

        # Save predictions
        prediction_path = f'{args.prediction_parent_path}/{args.period}ly_prediction_{prediction_year}.csv'
        prediction_data.to_csv(prediction_path, index=False)

if __name__ == '__main__':
    main()
    