import os
import numpy as np
import torch
from torch import nn
import random
import torch.optim as optim
import logging
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import ray


### Initial setup
# Working directory
project_dir = '/zfs/projects/darc/wolee_edehaan_suzienoh-exploratory-ml'
os.chdir(project_dir)

# Logging
logging.basicConfig(level=logging.WARNING)
file_name = os.path.basename(__file__)
logger = logging.getLogger(file_name)
logger.setLevel(level=logging.DEBUG)


### Model data
class ModelData():
    
    def __init__(self, ray_results_path, verbose=3):
        self.ray_results_path = ray_results_path
        self.verbose = verbose
        self.best_trial = None
        
    class FlexibleNeuralNetwork(nn.Module):
        def __init__(self, continuous_dim, hidden_dim, output_dim, num_layers, num_embeddings, embedding_dim, dropout_rate=0.5):
            # FlexibleNeuralNetwork is nested class
            super(ModelData.FlexibleNeuralNetwork, self).__init__()
            self.embedding = nn.Embedding(num_embeddings, embedding_dim)

            self.layers = nn.ModuleList()

            # Input layer (adjust input_dim to account for embedding_dim)
            self.layers.append(nn.Linear(continuous_dim + embedding_dim, hidden_dim))

            # Hidden layers
            for _ in range(num_layers - 1):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                self.layers.append(nn.Dropout(dropout_rate))

            # Output layer
            self.layers.append(nn.Linear(hidden_dim, output_dim))

            # Activation function, note that nn.ReLU() is not appropriate because of outputing non-negative number only
            self.activation = nn.LeakyReLU(negative_slope=4) # nn.Tanh()

            # Apply Xavier initialization to the layers
            self._initialize_weights()

        def _initialize_weights(self):
            for layer in self.layers:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

        def forward(self, x_continuous, x_categorical):
            embedded = self.embedding(x_categorical)
            embedded = embedded.view(embedded.size(0), -1)  # Flatten the embedding
            x = torch.cat((x_continuous, embedded), dim=1)

            for layer in self.layers:
                if isinstance(layer, nn.Linear):
                    x = self.activation(layer(x))
                else:
                    x = layer(x)  # This applies dropout
            return x
    
    @staticmethod
    def set_seed(seed):
        """ Set the seed for reproducibility """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    @staticmethod
    def train(model, train_loader, loss_function, optimizer, num_epochs=10):
        for epoch in range(num_epochs):
            running_loss = 0.0
            for x_continuous, x_embedding_vars, targets in train_loader:
                x_continuous, x_embedding_vars, targets = \
                    x_continuous.to(device), x_embedding_vars.to(device), targets.to(device)

                optimizer.zero_grad()
                # outputs is squeezed from shape [batch_size, 1] to [batch_size]
                outputs = model(x_continuous, x_embedding_vars).squeeze()
                loss = loss_function(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')
    
    @staticmethod
    def evaluate(model, test_loader, loss_function, device):
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x_continuous, x_embedding_vars, targets in test_loader:
                x_continuous, x_embedding_vars, targets = x_continuous.to(device), x_embedding_vars.to(device), targets.to(device)
                outputs = model(x_continuous, x_embedding_vars).squeeze()
                loss = loss_function(outputs, targets)
                total_loss += loss.item()

        average_loss = total_loss / len(test_loader)
        return average_loss

    @staticmethod
    def predict(model, data_loader):
        model.eval()
        predictions = []
        with torch.no_grad():
            for batch in data_loader:
                # If the loader provides three values, unpack and ignore the third (targets)
                if len(batch) == 3:
                    x_continuous, x_embedding_vars, _ = batch
                else:
                    x_continuous, x_embedding_vars = batch

                x_continuous, x_embedding_vars = x_continuous.to(device), x_embedding_vars.to(device)
                # outputs is squeezed from shape [batch_size, 1] to [batch_size]
                outputs = model(x_continuous, x_embedding_vars).squeeze()
                predictions.append(outputs.cpu().numpy())

        return np.concatenate(predictions)

    @staticmethod
    def train_fnn(config, train_loader, test_loader, ray_tuning=True):
        device = torch.device("cuda" if config["num_gpus"] > 0 else "cpu")
        continuous_dim = config["continuous_dim"]
        hidden_dim = config["hidden_dim"]
        output_dim = 1
        num_layers = config["num_layers"]
        num_embeddings = config["num_embeddings"]
        embedding_dim = config["embedding_dim"]
        dropout_rate = config["dropout_rate"]
        lr = config["lr"]
        weight_decay = config["weight_decay"]
        num_epochs = config["num_epochs"]

        # Use ModelData.FlexibleNeuralNetwork to reference the nested class
        model = ModelData.FlexibleNeuralNetwork(continuous_dim, hidden_dim, output_dim, num_layers, num_embeddings, embedding_dim, dropout_rate)
        model.to(device)

        loss_function = nn.L1Loss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        try:
            for epoch in range(num_epochs):
                model.train()
                running_loss = 0.0
                for x_continuous, x_embedding_vars, targets in train_loader:
                    x_continuous, x_embedding_vars, targets = x_continuous.to(device), x_embedding_vars.to(device), targets.to(device)

                    optimizer.zero_grad()
                    outputs = model(x_continuous, x_embedding_vars).squeeze()
                    loss = loss_function(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                avg_train_loss = running_loss / len(train_loader)
                avg_test_loss = ModelData.evaluate(model, test_loader, loss_function, device)
                metrics = {
                    'avg_train_loss': avg_train_loss,
                    'avg_test_loss': avg_test_loss,
                }
                if ray_tuning:
                    ray.train.report(metrics=metrics)
                else:
                    logger.info(f'Epoch {epoch + 1}/{num_epochs}, metrics: {metrics}')

        except Exception as e:
            metrics = {
                'avg_train_loss': float('inf'),
                'avg_test_loss': float('inf'),
            }
            logger.error(f"Training failed with exception: {e}")
            if ray_tuning:
                ray.train.report(metrics=metrics)
            else:
                logger.error(f"Training failed with exception: {e}")
        
        if ray_tuning:
            # If ray tuning is enabled, return the metrics dictionary 
            return metrics
        else:
            # Otherwise, return a trained model
            return model
    
    def get_best_trial(
        self, train_loader, test_loader, continuous_dim, num_embeddings, 
        num_samples=10, max_num_epochs=20, num_cpus=2, num_gpus=0, cpus_per_trial=1, gpus_per_trial=0
    ):
        config = {
            "continuous_dim": continuous_dim,
            "hidden_dim": tune.choice([i for i in range(5, 200, 10)]),
            "num_layers": tune.choice([1, 2, 3, 4, 5]),
            "num_embeddings": num_embeddings,
            "embedding_dim": tune.choice([i for i in range(1, 50, 5)]),
            "dropout_rate": tune.uniform(0.01, 0.7),
            "lr": tune.loguniform(1e-6, 1e-2),
            "weight_decay": tune.loguniform(1e-6, 1e-3),
            "num_epochs": max_num_epochs,
            "num_gpus": num_gpus,
        }

        scheduler = ASHAScheduler(
            metric="avg_test_loss",
            mode="min",
            max_t=max_num_epochs,
            grace_period=1,
            reduction_factor=2)

        reporter = CLIReporter(
            metric_columns=["average_train_loss", "avg_test_loss", "training_iteration"])

        result = tune.run(
            tune.with_parameters(ModelData.train_fnn, train_loader=train_loader, test_loader=test_loader),
            resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
            config=config,
            num_samples=num_samples,
            scheduler=scheduler,
            progress_reporter=reporter,
            storage_path=self.ray_results_path,
            verbose=self.verbose,
        )

        best_trial = result.get_best_trial("avg_test_loss", "min", "last")
        logger.info(f"Best trial config: {best_trial.config}")
        logger.info(f"Best trial training loss: {best_trial.last_result['avg_train_loss']}")
        logger.info(f"Best trial testing loss: {best_trial.last_result['avg_test_loss']}")
        
        self.best_trial = best_trial
        
        return self.best_trial
