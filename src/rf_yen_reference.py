# Import libraries

import argparse
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
from sklearn.metrics import mean_squared_error
import math
import csv



"""
Sample code to run this script: 

python3 rf_yen.py --infile_path './pf_vars_240620.csv' --outfile_path './results' --cum 1 --lower_percentile 5 --upper_percentile 95 --period 'month'

python3 rf_yen.py --infile_path './pf_vars_240620.csv' --outfile_path './results' --cum 1 --lower_percentile 5 --upper_percentile 95 --period 'quarter'

python3 rf_yen.py --infile_path './pf_vars_240620.csv' --outfile_path './results' --cum 0 --lower_percentile 5 --upper_percentile 95 --period 'quarter'
"""


#########################################
########### Processing Data  ############
#########################################

# Declare global variables
global con_list
global dum_list
global deps
global header

# List of continuous variables
con_list = ['absacc','acc','aeavol','age','agr','baspread','beta','betasq',
            'bm','bm_ia','cash','cashdebt','cashpr','cfp','cfp_ia','chatoia',
            'chcsho','chempia','chfeps','chinv','chmom','chnanalyst','chpmia',
            'chtx','cinvest','currat','depr','disp','dolvol','dy','ear','egr',
            'ep','fgr5yr','gma','grcapx','grltnoa','herf','hire','idiovol','ill',
            'indmom','invest','lev','lgr','maxret','mom12m','mom1m','mom36m',
            'mom6m','ms','mve','mve_ia','nanalyst','nincr','operprof','orgcap',
            'pchcapx_ia','pchcurrat','pchdepr','pchgm_pchsale','pchquick',
            'pchsale_pchinvt','pchsale_pchrect','pchsale_pchxsga','pchsaleinv',
            'pctacc','pricedelay','ps','quick','rd_mve','rd_sale','realestate',
            'retvol','roic','rsup','salecash','saleinv','salerec','secured',
            'sfe','sgr','sp','std_dolvol','std_turn','stdacc','stdcf','sue',
            'tang','tb','turn','zerotrade','credrat','dp_macro','ep_macro',
            'bm_macro','ntis_macro','tbl_macro','tms_macro','dfy_macro',
            'svar_macro','accruals','sentiment_8k','sentiment_10kq',
            'complexity_8k','complexity_10kq','pead','cf_ret',
            'manager_sentiment_index','bw','hjtz','mcs','pm','ato','ptg_surp',
            'meanrec','cowc_gr1a','fnl_gr1a','kz_index','o_score','ocf_at_chg1',
            'opex_at','z_score','roaa','roea','roavola']


# List of dummy variables
dum_list = ['convind','divi','divo','ipo','rd','securedind','sin',
            'credrat_dwn','sue_top','sue_bottom','exp_ea','accruals_top',
            'accruals_bottom','sentiment_8k_top','sentiment_8k_bottom',
            'sentiment_10kq_top','sentiment_10kq_bottom','complexity_8k_top',
            'complexity_8k_bottom','complexity_10kq_top','complexity_10kq_bottom',
            'pead_top','pead_bottom','cf_ret_top','cf_ret_bottom'] # Categorical variable binary 

# List of dependent variable
deps = con_list + dum_list +['date']

# Headers
header = ['permno','pyear']


def load_and_preprocess_data(file_path, period):
    
    """
    Loads and preprocesses the input data.

    Args:
    file_path (str): The path to the CSV file to be loaded.

    Returns:
    DataFrame: Preprocessed pandas DataFrame.
    """
    
    # Load data
    df = pd.read_csv(file_path)
    df.columns = [e.lower() for e in df.columns]
    
    df['date'] = df['pdate'].copy()
    df['date'] = pd.to_datetime(df['date'])
    # df['date'] = df['date'].dt.strftime('%m-%d-%Y')

    # Extract year
    df['pyear'] = df['date'].dt.year
    # Remove months if quarterly, otherwise, monthly, keep all months
    if period == 'quarter':
        df = df[df['date'].dt.month.isin([1,4,7,10])]

    # df.sort_values(['permno','date'], inplace=True)
    df.sort_values(['date', 'permno'], inplace=True)
    df['date'] = df['date'].dt.strftime('%Y-%m')
    
    print(df[['date', 'permno']].head())
    print('-' * 50)
        
    return df


#########################################
########## Building Pipeline  ###########
#########################################

class CustomWinsorizer(BaseEstimator, TransformerMixin):
    
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


class timePeriodMeanTransformer(BaseEstimator, TransformerMixin):
    
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


def build_pipeline(con_list, dum_list, lower_percentile, upper_percentile, period):
    
    """
    Builds a preprocessing pipeline for both numeric and categorical data.

    Args:
    con_list (list): List of continuous variable names.
    dum_list (list): List of dummy (categorical) variable names.
    lower_percentile (float): Lower percentile for winsorization.
    upper_percentile (float): Upper percentile for winsorization.
    period (string): Period for getting mean values (month vs quarter)

    Returns:
    Pipeline: A composed preprocessing pipeline.
    """
    
    numeric_pipeline = Pipeline([
        ('winsorizer', CustomWinsorizer(lower_percentile=lower_percentile, upper_percentile=upper_percentile)),
        ('scaler', StandardScaler()),
        ('impute_con', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0))
    ])

    categorical_pipeline = Pipeline([
        ('impute_cat', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)),
    ])

    preprocessing = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, con_list),
            ('cat', categorical_pipeline, dum_list)
        ], remainder='passthrough')

    pipeline = Pipeline([
        ('Time_period_mean_imputation', timePeriodMeanTransformer('date', con_list, period)),
        ('Preprocessing', preprocessing),
    ])
    
    return pipeline


#########################################
######### Custome Time Split  ###########
#########################################

def custom_yearly_split(pyear_info, train_val_period, n_splits):
    
    """
    Splits time series data into yearly folds for cross-validation based on the specified training and validation period.

    This function generates indices for training and validation sets for each fold, dividing the data into yearly intervals
    based on the provided training and validation period. It also logs detailed information about each fold, including the
    training and validation years, and optionally saves this information to a CSV file.

    Parameters:
    - pyear_info (pandas.Series): A series containing year information for each sample in the dataset.
    - train_val_period (list of int): A list of years representing the total period to be divided into training and validation sets.
    - n_splits (int): The number of folds/splits for cross-validation.

    Yields:
    - tuple of np.ndarray: For each fold, yields a tuple containing two arrays: indices for the training set and indices for the validation set.

    Returns:
    - list of dict: A list containing dictionaries with detailed information about each fold, including the training and validation years.

    Note:
    - The function assumes that `pyear_info` contains year values that match those in `train_val_period`.
    - The CSV file logging fold information is saved to './results3/w_c_q_cts_fold_log_val-all.csv'. Uncomment related lines in the function to enable this feature.
    """
    
    # Define the total number of years in the training/validation period for CV
    total_years = len(train_val_period)
    # Calculate the training increment by rounding down
    yr_increment = total_years // (n_splits + 1)

    # # Initialize an empty list to store all the fold information dictionaries
    # all_fold_info = []

    # Loop through each fold 
    for split in range(1, n_splits + 1):
        
        # Define train size and years
        train_size = split * yr_increment
        train_years = train_val_period[:train_size]
        
        # Define validation years
        # For the last fold, extend the validation years to include all remaining years
        val_years = train_val_period[train_size:train_size + yr_increment] if split < n_splits else train_val_period[train_size:]
        
        train_mask = pyear_info.isin(train_years)
        val_mask = pyear_info.isin(val_years)

        # Convert boolean masks to indices
        train_index = np.where(train_mask)[0]
        val_index = np.where(val_mask)[0]

        # Log fold info
        # fold_info = {'Script': 'cts no val', 'CV period': train_val_period, 'Fold': split, 'Training Years': train_years, 'Validation Years': val_years}
        # all_fold_info.append(fold_info)

        print(f'Training and Validation period for CV: {train_val_period}, Fold {split}: Training Years: {train_years}, Validation Years: {val_years}\n')
        yield train_index, val_index
       
    # Save all_fold_info to a CSV file 
#     fold_file_path = f'./results3/w_c_q_cts_fold_log_no_val-all.csv'
#     with open(fold_file_path, 'a', newline='') as csvfile:
#         fieldnames = all_fold_info[0].keys()
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

#         # Check if the file is empty (has no content) and write the header only if it's empty
#         if csvfile.tell() == 0:
#             writer.writeheader()

#         # Write all the fold information without the header
#         writer.writerows(all_fold_info)

#     return all_fold_info


#########################################
### Traiing, Validating, and Testing  ###
#########################################

def train_and_evaluate(df, pipeline, cum, ts, period):
    
    """
    Trains a Random Forest model on a given dataset, evaluates its performance over time,
    and computes feature importances. It utilizes a custom yearly split for cross-validation
    and evaluates the model on both validation and test sets.

    The function supports analyzing the data on a quarterly or monthly basis, adjusting
    the target variable accordingly. It also allows for cumulative data consideration up to
    a specified year for training, with a dynamic training-validation-test split based on the
    year of the data.

    Args:
        df (pd.DataFrame): The dataframe containing the dataset for training and evaluation.
                           It must include columns for 'pyear', the period year, and target variables
                           'retq' for quarterly or 'ret' for monthly returns, among other features.
        pipeline (Pipeline): A preprocessing pipeline that includes scaling, encoding, and/or
                             dimensionality reduction steps to be applied to the features before training.
        cum (int): A binary flag indicating whether to use cumulative data for training (1) or
                   a fixed historical window (0).    
        ts (int): A binary flag indicating whether to use the scikit learn time series split (1) or 
                  custom timle split by year (0).
        period (str): A string indicating the period for analysis ('quarter' or 'month') which
                      determines the target variable.

    Returns:
        tuple: Contains four elements:
            - DataFrame with the yearly and rank-wise portfolio returns.
            - DataFrame with the stock predictions, including their predicted probabilities
              and rank within their respective year.
            - DataFrame summarizing the best parameters found during the RandomizedSearchCV
              process for each training period.
            - DataFrame with feature importances calculated based on permutation importance
              for each training period.

    Raises:
        ValueError: If the 'period' argument is not 'quarter' or 'month'.
        ValueError: If the 'ts' argument is not '1' or '0'.
    """

    # Define the target variable
    if period == 'quarter':
        target = 'retq'
    elif period == 'month':
        target = 'ret'
    else:
        raise ValueError("period must be 'quarter' or 'month'")
        
    print(f'\nTime series split : {ts}\n')
    
    # Set year range of the sample
    years = list(df['pyear'].drop_duplicates().sort_values())[5:-1]
    
    # Create an empty dataframe to store the results
    result = pd.DataFrame()
    df_best_params = pd.DataFrame()
    df_feature_importances = pd.DataFrame()
    stocks = pd.DataFrame()

    # Create an empty dictionary and lists to store best parameters and feature importances
    best_param_list = []
    best_param_dict = {}
    feature_list = []
    
    # Iterate through years
    for yr in tqdm(years):
    
        # Data preparation for training and validation
        if cum == 1:
            x_train = df[(df['pyear']<=yr)]
            y_train = df[(df['pyear']<=yr)]

        else:
            x_train = df[(df['pyear']<=yr)&(df['pyear']>=yr-7)]
            y_train = df[(df['pyear']<=yr)&(df['pyear']>=yr-7)]
            
        # Training
        training_years = sorted(x_train.pyear.unique())
        print(f'Training Years: {training_years}\n') 

        x_train = x_train.loc[:, deps]
        pyear_index = df.loc[x_train.index, 'pyear']
        y_train = y_train.loc[:, target] 
         
        # Fit the pipeline to the training data
        pipeline.fit(x_train)
        x_train = pipeline.transform(x_train)
        x_train = x_train[:,:-2]
        print(f'New Shape: {x_train.shape}\n')

        # Testing
        x_test = df.loc[(df['pyear']==yr+1), deps]
        print(f'Test Year: {yr+1}\n')
        
        x_test = pipeline.transform(x_test)
        x_test = x_test[:,:-2]      
        y_test = df.loc[(df['pyear']==yr+1), target]
  
        # Subset df
        rf_strategy = df.loc[(df['pyear']==yr+1),['permno', 'pyear',target,'date','mve_m']]
        
        # Build the Random Forest model using a random search
        rf = RandomForestRegressor(n_jobs=30, random_state=42)

        # Define the type of time series cross-validation
        if ts == 1:
            tscv = TimeSeriesSplit(n_splits=3)
        elif ts == 0:
            tscv = custom_yearly_split(pyear_info=pyear_index, train_val_period=training_years, n_splits=3)
        else:
            raise ValueError('ts must be 1 (for sklearn time series split) or 0 (for custom yearly time split)')
        
        param_distributions = {
            'max_depth': list(range(10, 200)),        # Max depth of the trees
            'max_features': ['log2',  'sqrt'],        # Number of features to consider for best split
            'min_samples_leaf': [3, 5],               # Min number of samples required to be a leaf node
            'min_samples_split': [3, 5],              # Min number of samples required to split an internal node
            'n_estimators': list(range(100, 1000)),   # Number of trees in the forest

        }
            
        # Create the RandomizedSearchCV instance
        random_search = RandomizedSearchCV(estimator=rf, 
                                           param_distributions=param_distributions, 
                                           n_iter=10,  # Number of parameter settings to try
                                           scoring='neg_mean_squared_error', 
                                           cv=tscv, 
                                           error_score='raise',
                                           random_state=42)  # Ensure reproducibility


        # Fit the model
        random_search.fit(x_train, y_train)
            
        best_parameters = random_search.best_params_
        print(f'    Best Parameters: {best_parameters}')
                
        best_score = - random_search.best_score_
        rmse_train = np.sqrt(best_score)
        print(f'    Best Train RMSE Score: {rmse_train}')
        
        # Get the training year range
        min_year = min(training_years)
        max_year = max(training_years)
        training_year_range = f'{min_year}-{max_year}'
        
        # Access the best Random Forest model 
        best_rf_model = random_search.best_estimator_
        # Retrieve the feature importances
        feature_importances = best_rf_model.feature_importances_

        # Get feature names
        feature_names = deps
        # Combine feature importances with feature names
        features_and_importances = zip(feature_names, feature_importances)
        sorted_features_and_importances = sorted(features_and_importances, key=lambda x: x[1], reverse=True)
        # Create a df 
        df_feature_importances = pd.DataFrame(sorted_features_and_importances)
        df_feature_importances['training_years'] = training_year_range
        # Append the feature imporatnces to the feature importance list
        feature_list.append(df_feature_importances)
        print(f'Feature importances recorded.\n')

        # Evaluate on test set
        y_pred_test = best_rf_model.predict(x_test)
        
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        print(f'    Test RMSE Score: {rmse_test}')
        
        # Store best params and MSE scores into a dictionary and append it to a list
        best_param_dict = {'train_years': training_year_range, 'best_params': best_parameters, 'train_rmse_score': rmse_train, 'test_rmse_score': rmse_test}
        best_param_list.append(best_param_dict)
        print(f'Best parameters recorded.\n')
       
        # Evalaution
        rf_strategy['prob']=y_pred_test
        rf_strategy.sort_values('prob', inplace=True)
        rf_strategy['rank'] = rf_strategy.groupby(['date'])['prob'].transform(lambda x: pd.qcut(x.values, 10, labels=False, duplicates='drop'))
        rf_strategy['port_size'] = rf_strategy.groupby(['date','rank'])['mve_m'].transform(sum) 
        rf_strategy['port_ret'] = rf_strategy[target] * rf_strategy['mve_m']/rf_strategy['port_size'] 
        
        year_vret = rf_strategy.groupby(['date','rank'])['port_ret'].sum()
        year_vret = year_vret.reset_index()
        
        # Append the results into the result df
        result = pd.concat([result, year_vret]).reset_index(drop=True)
        stocks = pd.concat([stocks, rf_strategy]).reset_index(drop=True)
    
    # Concatenate the best param list
    df_best_params = pd.DataFrame(best_param_list)
    
    # Concatenate the feature importance list
    df_feature_importances = pd.concat(feature_list)
          
    return result, df_best_params, df_feature_importances, stocks


#########################################
########## Creating Rankings  ###########
#########################################

def rank_stock_prob(df):
    
    """
    Ranks stocks within each date and rank by 'prob' value, adding a 'rank_ind' column for individual stock ranks.

    Stocks are ranked in descending order by their 'prob' value within each 'date' and 'rank' group. Ties are
    broken by order of appearance. The DataFrame is then sorted by 'date', 'decile', and 'rank_ind'.

    Parameters:
    - df (pandas.DataFrame): DataFrame with 'date', 'decile', and 'prob' columns.

    Returns:
    - df (pandas.DataFrame): Modified DataFrame with an added 'rank_ind' column, sorted accordingly.

    The function modifies the DataFrame in-place but also returns it for chaining or further use.
    """
    
    df['rank_ind'] = df.groupby(['date', 'rank'])['prob'].rank(method='first', ascending = False)
    df['rank_ind'] = df['rank_ind'].astype(int)
    df.sort_values(by=['date', 'rank', 'rank_ind'], inplace=True)
    
    return df


#########################################
########## Parsing Arguments ############
#########################################

def parse_arguments():
    
    """
    Parses command line arguments.

    Returns:
    argparse.Namespace: Parsed arguments.
    """
    
    parser = argparse.ArgumentParser(description='Train and evaluate a model on the provided dataset.')
    parser.add_argument('--infile_path', type=str, required=True, help='Path to the CSV data file')
    parser.add_argument('--outfile_path', type=str, required=True, help='Path for the output files')
    parser.add_argument('--cum', type=int, choices=[0, 1], default=1, help='Flag to use cumulative training (1) or not (0)')
    parser.add_argument('--ts', type=int, choices=[0, 1], default=1, help='Flag to use scikit-learn time series split (1) or custom yearly time split (0)')
    parser.add_argument('--lower_percentile', type=float, default=5, help='Lower percentile for winsorization')
    parser.add_argument('--upper_percentile', type=float, default=95, help='Upper percentile for winsorization')
    parser.add_argument('--period', type=str, choices=['quarter', 'month'], default='quarter', help='Period for mean imputation, either "quarter" or "month"')
    return parser.parse_args()


#########################################
################## Main #################
#########################################

def main():
    
    """
    Main function to execute the data loading, training, and evaluation processes.
    """
    
    args = parse_arguments()
    start = time.time()
    
    if args.period == 'quarter':
        target = 'retq'
    elif args.period == 'month':
        target = 'ret'
    else:
        raise ValueError("period must be 'quarter' or 'month'")
    
    # Load and preprocess data
    print('\nLoading and preprocessing data...\n')
    df = load_and_preprocess_data(args.infile_path, args.period)
   
    # Drop null values in the target column and get years 2020 or prior
    df1 = df.dropna(subset=[target])
    df1 = df1[df1['pyear'] <= 2020]
    df1.reset_index(drop=True, inplace=True)
 
    print('Training in progress...\n')
    # Build a training pipeline
    pipeline = build_pipeline(con_list, dum_list, args.lower_percentile, args.upper_percentile, args.period)
    
    # Train and evaluate models
    df_result, df_best, df_feature_importances, df_stocks = train_and_evaluate(df1, pipeline, args.cum, args.ts, args.period)
    
    # Generate the output file path for results and best params
    upper_percentile_int = int(args.upper_percentile)
    result_path = f'{args.outfile_path}_{upper_percentile_int}.csv'
    df_result.to_csv(result_path, index=False)
    print(f'Results saved to {result_path}.\n')
          
    # Save stocks in portofolios
    stock_file_path = f'{args.outfile_path}_{upper_percentile_int}_stocks.csv'
    df_stocks_ranked = rank_stock_prob(df_stocks)
    df_stocks_ranked.to_csv(stock_file_path, index=False) 
    
    # Save best parameters
    param_path = f'{args.outfile_path}_{upper_percentile_int}_best_params.csv'
    df_best.to_csv(param_path, index=False)  
    print(f'Best params saved to {param_path}.\n')  
        
    # Save feature imporatances
    feature_importance_file_path = f'{args.outfile_path}_{upper_percentile_int}_feature_importance.csv'
    df_feature_importances.to_csv(feature_importance_file_path, index=False)
    print(f'Feature importances saved to {feature_importance_file_path}.\n')
 
    end = time.time()
    print(f'\nDone! The total elapsed time is {(end - start) / 60:.2f} minutes.\n')


#########################################
################ Starter ################
#########################################

if __name__ == "__main__":
    main()
