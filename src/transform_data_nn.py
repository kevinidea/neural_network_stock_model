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
from preprocess_data_nn import PreprocessData

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

class TransformData():
    
    def __init__(self, train_year_start, prediction_year, df, period, continuous_vars, binary_vars, embed_vars, header, year_col='pyear'):
        # Train, test, retrain and prediction data
        self.train_year_start = train_year_start # example: 1980
        self.prediction_year = prediction_year # example: 1988
        self.test_year = self.prediction_year - 1 # example: 1987
        self.train_years = list(range(self.train_year_start, self.test_year, 1)) # example: 1980-1986
        self.retrain_years = list(range(self.train_year_start, self.prediction_year, 1)) # example: 1980-1987
        self.df = df
        self.year_col = year_col
        self.train_data = None # example: 1980-1986
        self.test_data = None # example: 1987
        self.retrain_data = None # example: 1980-1987
        self.prediction_data = None # example: 1988
        
        # Variables
        self.period = self._validate_period(period)
        self.continuous_vars = continuous_vars
        self.binary_vars = binary_vars
        self.embed_vars = embed_vars
        self.header = header
        
        # To store some outputs
        self.pipeline = None
        # Make sure 'date' variable is at the end because the downstream transformation requires it
        self.independent_vars = self.continuous_vars + self.binary_vars + self.embed_vars + ['date']
        self.target = self._set_target(period)
        # length of continuous variables and binary variables
        self.continuous_len = len(self.continuous_vars) + len(self.binary_vars)
        
    def get_train_data(self):
        train_data = self.df.loc[(self.df[self.year_col].isin(self.train_years))]
        self.train_data = train_data
        # Examine the actual data
        actual_years = sorted(self.train_data[self.year_col].unique())
        logger.info(f'Train data years: {actual_years}')
        logger.info(f'Train_data: {self.train_data.shape}\n')
        
        return self.train_data
    
    def get_test_data(self):
        test_data = self.df.loc[self.df[self.year_col] == self.test_year]
        self.test_data = test_data
        # Examine the actual data
        actual_years = sorted(self.test_data[self.year_col].unique())
        logger.info(f'Test data years: {actual_years}')
        logger.info(f'Test_data: {self.test_data.shape}\n')

        return self.test_data
    
    def get_retrain_data(self):
        retrain_data = self.df.loc[(self.df[self.year_col].isin(self.retrain_years))]
        self.retrain_data = retrain_data
        # Examine the actual data
        actual_years = sorted(self.retrain_data[self.year_col].unique())
        logger.info(f'Retrain data years: {actual_years}')
        logger.info(f'Retrain_data: {self.retrain_data.shape}\n')
        
        return self.retrain_data
    
    def get_prediction_data(self):
        prediction_data = self.df.loc[self.df[self.year_col] == self.prediction_year]
        self.prediction_data = prediction_data
        # Examine the actual data
        actual_years = sorted(self.prediction_data[self.year_col].unique())
        logger.info(f'Retrain data years: {actual_years}')
        logger.info(f'Prediction_data: {self.prediction_data.shape}\n')
        
        return self.prediction_data
    
    def _validate_period(self, period):
        if period not in ['quarter', 'month']:
            raise ValueError("period must be 'quarter' or 'month'")
        return period
    
    def _set_target(self, period):
        if period == 'quarter':
            return 'retq'
        elif period == 'month':
            return 'ret'
    
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
    
    def transform_data(self, train_data, test_data, features, target, pipeline):
        # Train data
        x_train = train_data.loc[:, features]
        y_train = train_data.loc[:, target]

        # Fit the pipeline to the train data
        pipeline.fit(x_train)
        x_train_tf = pipeline.transform(x_train)
        x_train_tf = x_train_tf[:, :-2]

        # Test data
        x_test = test_data.loc[:, features]
        y_test = test_data.loc[:, target]

        # Fit the pipeline to the test data
        x_test_tf = pipeline.transform(x_test)
        x_test_tf = x_test_tf[:, :-2]

        # Transform data into numpy array as type float32
        x_train_tf = x_train_tf.astype(np.float32)
        y_train_tf = y_train.to_numpy(np.float32)
        x_test_tf = x_test_tf.astype(np.float32)
        y_test_tf = y_test.to_numpy(np.float32)

        # # Transform them to tensor floats
        x_train_tf = torch.tensor(x_train_tf).float()
        y_train_tf = torch.tensor(y_train_tf).float()
        x_test_tf = torch.tensor(x_test_tf).float()
        y_test_tf = torch.tensor(y_test_tf).float()

        return x_train_tf, y_train_tf, x_test_tf, y_test_tf
        
        
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

if __name__ == '__main__':
    main()
        