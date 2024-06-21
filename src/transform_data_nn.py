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

class TransformData():
    
    def __init__(self, train_year_start, prediction_year, df, year_col='pyear'):
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
        
    def get_train_data(self):
        train_data = self.df.loc[(self.df[self.year_col].isin(self.train_years))]
        self.train_data = train_data
        # Examine the actual data
        actual_years = sorted(self.train_data[self.year_col].unique())
        logger.info(f'Train data years: {actual_years}')
        
        return self.train_data
    
    def get_test_data(self):
        test_data = self.df.loc[self.df[self.year_col] == self.test_year]
        self.test_data = test_data
        # Examine the actual data
        actual_years = sorted(self.test_data[self.year_col].unique())
        logger.info(f'Test data years: {actual_years}')

        return self.test_data
    
    def get_retrain_data(self):
        retrain_data = self.df.loc[(self.df[self.year_col].isin(self.retrain_years))]
        self.retrain_data = retrain_data
        # Examine the actual data
        actual_years = sorted(self.retrain_data[self.year_col].unique())
        logger.info(f'Retrain data years: {actual_years}')
        
        return self.retrain_data
    
    def get_prediction_data(self):
        prediction_data = self.df.loc[self.df[self.year_col] == self.prediction_year]
        self.prediction_data = prediction_data
        # Examine the actual data
        actual_years = sorted(self.prediction_data[self.year_col].unique())
        logger.info(f'Retrain data years: {actual_years}')
        
        return self.prediction_data
        
        
        
        