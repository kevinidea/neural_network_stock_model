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
import torch.nn.init as init
import random
import torch.optim as optim
import logging
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import ray


### Initial setup
# Working directory
os.chdir('/zfs/projects/darc/wolee_edehaan_suzienoh-exploratory-ml')

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


### Load and preprocess the data

class PreprocessData():
    
    def __init__(self, infile_path, period, continuous_vars, binary_vars, embed_vars, header):
        self.infile_path = infile_path
        self.period = self._validate_period(period)
        self.continuous_vars = continuous_vars
        self.binary_vars = binary_vars
        self.embed_vars = embed_vars
        self.header = header
        self.independent_vars = self.continuous_vars + self.binary_vars + self.embed_vars + ['date']
        # To store some outputs
        self.pipeline = None
        self.df = None
        self.target = self._set_target(period)
        
    def _validate_period(self, period):
        if period not in ['quarter', 'month']:
            raise ValueError("period must be 'quarter' or 'month'")
        return period
    
    def _set_target(self, period):
        if period == 'quarter':
            return 'retq'
        elif period == 'month':
            return 'ret'
    
    def load_and_preprocess_data(self):

        """
        Loads and preprocesses the input data.

        Returns:
        DataFrame: Preprocessed pandas DataFrame.
        """
        logger.info('\n\nLoading and preprocessing data...\n')
        try:
            # Load data
            df = pd.read_csv(self.infile_path)
            df.columns = [e.lower() for e in df.columns]

            # Convert 'date' to datetime
            df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')

            # Extract year
            df['pyear'] = df['date'].dt.year

            # Remove months if quarterly, otherwise keep all months
            if self.period == 'quarter':
                df = df[df['date'].dt.month.isin([1, 4, 7, 10])]

            # Sort data
            df.sort_values(['date', 'permno'], inplace=True)

            # Format date
            df['date'] = df['date'].dt.strftime('%Y-%m')

            # Drop unnecessary column if exists
            if 'fpedats' in df.columns:
                del df['fpedats']

            # Print first few rows for debugging
            logger.debug(df[['date', 'permno']].head())
            logger.debug('-' * 50)
            
            # Update self.df
            self.df = df
            
            return self.df

        except FileNotFoundError:
            logger.error(f"Error: The file {self.infile_path} does not exist.")
        except pd.errors.ParserError:
            logger.error("Error: The file could not be parsed.")
        except Exception as e:
            logger.error(f"An error occurred: {e}")

    class _CustomWinsorizer(BaseEstimator, TransformerMixin):

        """
        A custom transformer for Winsorizing numeric data.

        Attributes:
        lower_percentile (int): The lower percentile for clipping data.
        upper_percentile (int): The upper percentile for clipping data.
        """

        def __init__(self, lower_percentile, upper_percentile):
            self.lower_percentile = lower_percentile
            self.upper_percentile = upper_percentile

        def fit(self, X, y=None):
            self.lower_bound_ = np.percentile(X, self.lower_percentile)
            self.upper_bound_ = np.percentile(X, self.upper_percentile)
            return self

        def transform(self, X):
            X_clipped = np.clip(X, self.lower_bound_, self.upper_bound_)

            return X_clipped

    class _TimePeriodMeanTransformer(BaseEstimator, TransformerMixin):

        """
        A custom transformer for imputing missing data based on time period means.

        Attributes:
        date_column (str): The column name representing dates.
        numeric_columns (list): List of numeric column names for which means are calculated.
        period (str): The time period for grouping data, either 'quarter' or 'month'.
        """

        def __init__(self, date_column, numeric_columns, period='quarter'):
            self.date_column = date_column
            self.numeric_columns = numeric_columns
            self.period = period

        def fit(self, X, y=None):
            X[self.date_column] = pd.to_datetime(X[self.date_column])
            if self.period == 'quarter':
                X['Period'] = X[self.date_column].dt.quarter
            elif self.period == 'month':
                X['Period'] = X[self.date_column].dt.month
            else:
                raise ValueError("period must be 'quarter' or 'month'")

           # Calculate and store the means of each numeric column for each time period
            self.period_means_ = X.groupby('Period')[self.numeric_columns].mean()
            return self

        def transform(self, X):
            X[self.date_column] = pd.to_datetime(X[self.date_column])
            if self.period == 'quarter':
                X['Period'] = X[self.date_column].dt.quarter
            elif self.period == 'month':
                X['Period'] = X[self.date_column].dt.month

            for col in self.numeric_columns:
                X[col] = X.apply(lambda row: row[col] if not pd.isna(row[col]) 
                                 else self.period_means_.loc[row['Period'], col], axis=1)
            # return X.drop(['Period'], axis=1)
            return X

    def build_pipeline(self, lower_percentile, upper_percentile):

        """
        Builds a preprocessing pipeline for both numeric and categorical data.

        Args:
        lower_percentile (float): Lower percentile for winsorization.
        upper_percentile (float): Upper percentile for winsorization.

        Returns:
        Pipeline: A composed preprocessing pipeline.
        """

        numeric_pipeline = Pipeline([
            # ('fill_na', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)),
            ('winsorizer', self._CustomWinsorizer(lower_percentile=lower_percentile, upper_percentile=upper_percentile)),
            ('scaler', StandardScaler()),
            ('impute_con', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0))
        ])

        categorical_pipeline = Pipeline([
            ('impute_cat', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)),
        ])

        embed_pipeline = Pipeline([
            ('impute_embed', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)),
        ])

        preprocessing = ColumnTransformer(
            transformers=[
                ('num', numeric_pipeline, self.continuous_vars),
                ('cat', categorical_pipeline, self.binary_vars),
                ('embed', embed_pipeline, self.embed_vars),
            ], remainder='passthrough')

        pipeline = Pipeline([
            ('Time_period_mean_imputation', self._TimePeriodMeanTransformer('date', self.continuous_vars, self.period)),
            ('Preprocessing', preprocessing),
        ])
        
        self.pipeline = pipeline
        
        return self.pipeline
    
    def apply_secondary_preprocessing(self):
        logger.info('\n\nApplying secondary data preprocessing..\n')
        # Drop null values in the target column and get years 2020 or prior
        try:
            df = self.df.dropna(subset=[self.target])
            df = df[df['pyear'] <= 2020]
            df.reset_index(drop=True, inplace=True)

            # Update the self.df
            self.df = df
            
        except Exception as e:
            logger.error(f'Secondary Proprocessing was NOT applied because of an error: {e}')
        
        return self.df
    
def main():
    # Create a preprocess data instance
    infile_path = 'Info Processing and Mutual Funds/masterv14.csv'
    period = 'month'
    preprocess_data = PreprocessData(infile_path, period, continuous_vars, binary_vars, embed_vars, header)
    
    # Load and preprocess the data
    df = preprocess_data.load_and_preprocess_data()
    logger.debug(df.shape)
    logger.debug(preprocess_data.df.shape)
    
    # Apply secondary preprocessing
    df = preprocess_data.apply_secondary_preprocessing()
    logger.debug(df.shape)
    logger.debug(preprocess_data.df.shape)
    logger.info(preprocess_data.df.head())
    
    # Build a pipeline
    preprocess_data.build_pipeline(lower_percentile=5, upper_percentile=95)
    logger.debug(preprocess_data.pipeline)

if __name__ == "__main__":
    main()
