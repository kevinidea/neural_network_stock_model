import pandas as pd
import os
import glob
import argparse
import logging

### Initial setup

# Working directory
project_dir = '/zfs/projects/darc/wolee_edehaan_suzienoh-exploratory-ml'
os.chdir(project_dir)

# Logging
logging.basicConfig(level=logging.WARNING)
file_name = os.path.basename(__file__)
logger = logging.getLogger(file_name)
logger.setLevel(level=logging.INFO)

### Post processing the prediction files

# Functions that do the post-processing
def get_prediction_file_paths(directory):
    # Use glob to get all csv files in the directory
    csv_files = glob.glob(os.path.join(directory, '*prediction*.csv'))
    csv_files = sorted(csv_files, reverse=False)
    return csv_files

def postprocess_predictions(prediction_df, period, prediction_col='pred'):
    
    # Target name based on period
    if period == 'quarter':
        target = 'retq'
    elif period == 'month':
        target = 'ret'
    else:
        raise ValueError("period must be 'quarter' or 'month'")
        
    prediction_df['prob']=prediction_df[prediction_col]
    prediction_df.sort_values('prob', inplace=True)
    prediction_df['rank'] = prediction_df.groupby(['date'])['prob'].transform(lambda x: pd.qcut(x.values, 10, labels=False, duplicates='drop'))
    prediction_df['port_size'] = prediction_df.groupby(['date','rank'])['mve_m'].transform('sum')
    prediction_df['port_ret'] = prediction_df[target] * prediction_df['mve_m']/prediction_df['port_size'] 

    year_vret = prediction_df.groupby(['date','rank'])['port_ret'].sum()
    year_vret = year_vret.reset_index()
    
    return year_vret

def create_result(prediction_parent_path, period, result_file_name=None):
    # Get the prediction data paths
    prediction_data_paths = get_prediction_file_paths(prediction_parent_path)
    
    # Postprocess the prediction and append all the results together
    results = pd.DataFrame()
    for df_path in prediction_data_paths:
        logger.info(f'Postprocessing data from path: {df_path}')
        df = pd.read_csv(df_path)
        year_vret = postprocess_predictions(df, period=period)
        results = pd.concat([results, year_vret]).reset_index(drop=True)
    
    # Sort the results
    logger.info(f'Sort the result by date and rank both ascendingly')
    sorted_results = results.sort_values(by=['date', 'rank'],  ascending=[True, True]).reset_index(drop=True)
    
    # Save the sorted results to the same parent directory if file name is given
    if result_file_name:
        logger.info(f'Saved the result file: {result_file_name}\nto the directory: {prediction_parent_path}')
        sorted_results.to_csv(f'{prediction_parent_path}/{result_file_name}', index=False)
        
    return sorted_results

def main():
    
    logger.info(f'Post processing prediction files')
    parser = argparse.ArgumentParser(description='Postprocess prediction files')
    parser.add_argument('--prediction_parent_path', type=str, required=True,
                        help='The parent path for prediction files.')
    parser.add_argument('--period', type=str, required=True,
                        help='Period is either month or quarter')
    parser.add_argument('--result_file_name', type=str, default='result.csv',
                    help='The name of the result file including extension')
    
    args = parser.parse_args()
    create_result(args.prediction_parent_path, args.period, args.result_file_name)
    
if __name__ == '__main__':
    main()