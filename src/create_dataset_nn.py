import os
import torch
from torch.utils.data import Dataset
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
 
class XandYDataset(Dataset):
    """
    Take the continuous variables, embedding variables and y to create a dataset object

    Args:
    X_continuous_vars (Tensor): Continuous variables including binary variables
    X_embedding_vars (Tensor): Embedding variables (typically categorical) where we want to train an embedding layer downstream
    y (Tensor): Dependent variable

    Returns:
    Pytorch Dataset object
    """
    def __init__(self, X_continuous_vars, X_embedding_vars, y):
        self.X_continuous_vars = X_continuous_vars
        if isinstance(X_embedding_vars, torch.Tensor):
            self.X_embedding_vars = X_embedding_vars.numpy()
        else:
            self.X_embedding_vars = X_embedding_vars
        self.y = y

        # Ensure the categorical variables are strings
        self.X_embedding_vars = self.X_embedding_vars.astype(str)

        # Create a mapping dictionary for each unique categorical variable to integer
        self.embedding_var_mappings = self._create_mappings(self.X_embedding_vars)

    def _create_mappings(self, X_embedding_vars):
        mappings = {}
        for i in range(X_embedding_vars.shape[1]):
            unique_values = set(X_embedding_vars[:, i])
            mappings[i] = {val: idx for idx, val in enumerate(unique_values)}
        return mappings

    def __len__(self):
        return len(self.X_continuous_vars)

    def __getitem__(self, idx):
        X_continuous = self.X_continuous_vars[idx]
        X_embedding = self.X_embedding_vars[idx]

        # Convert each categorical variable to its corresponding integer index
        X_embedding_int = torch.tensor([self.embedding_var_mappings[i][val] for i, val in enumerate(X_embedding)], dtype=torch.long)

        return X_continuous, X_embedding_int, self.y[idx]
