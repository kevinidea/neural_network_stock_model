import os
import pandas as pd
import logging


### Initial setup

# Working directory
project_dir = '/zfs/projects/darc/wolee_edehaan_suzienoh-exploratory-ml'
os.chdir(project_dir)

# Logging
logging.basicConfig(level=logging.WARNING)
file_name = os.path.basename(__file__)
logger = logging.getLogger(file_name)
logger.setLevel(level=logging.DEBUG)


### Load and preprocess the data

class PreprocessData():
    
    def __init__(self, infile_path, period):
        self.infile_path = infile_path
        self.period = self._validate_period(period)
        # To store some outputs
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
    preprocessor = PreprocessData(infile_path, period)
    
    # Load and preprocess the data
    df = preprocessor.load_and_preprocess_data()
    logger.debug(df.shape)
    logger.info(preprocessor.df.shape)
    
    # Apply secondary preprocessing
    df = preprocessor.apply_secondary_preprocessing()
    logger.debug(df.shape)
    logger.info(preprocessor.df.shape)
    logger.info(preprocessor.df.head())

if __name__ == "__main__":
    main()
