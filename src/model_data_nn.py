import os
import numpy as np
import torch
from torch import nn
import random
import torch.optim as optim
# To disable annoying duplicate messages from Ray
os.environ['RAY_DEDUP_LOGS'] = '0'
import logging
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import ray
from torch.utils.data import DataLoader


### Initial setup
# Working directory
project_dir = '/zfs/projects/darc/wolee_edehaan_suzienoh-exploratory-ml'
os.chdir(project_dir)

# Logging
logging.basicConfig(level=logging.WARNING)
file_name = os.path.basename(__file__)
logger = logging.getLogger(file_name)
logger.setLevel(level=logging.INFO)


### Model data
class ModelData():
    
    def __init__(self, ray_results_path, verbose=3):
        self.ray_results_path = ray_results_path
        self.verbose = verbose
        self.best_trial = None
        
    class FlexibleNeuralNetwork(nn.Module):
        def __init__(self, continuous_dim, hidden_dim, output_dim, num_layers, num_embeddings, embedding_dim, dropout_rate=0.5):
            super(ModelData.FlexibleNeuralNetwork, self).__init__()

            # Embedding layer for one categorical variable
            self.num_embeddings = num_embeddings
            # Handle new category during test or prediction phase
            # Note that index starts at 0 so num_embeddings position is 1 unit outside of the range
            self.unknown_index = num_embeddings
            self.embedding = nn.Embedding(num_embeddings + 1, embedding_dim) # + 1 for the unknown index

            # First layer
            input_dim = continuous_dim + embedding_dim
            self.first_layer = nn.Linear(input_dim, hidden_dim)
            self.first_batch_norm = nn.BatchNorm1d(hidden_dim)
            self.first_activation = nn.ReLU()
            self.first_dropout = nn.Dropout(dropout_rate)

            # Dynamic middle layers
            self.middle_layers = nn.ModuleList()
            self.middle_batch_norms = nn.ModuleList()
            self.middle_activations = nn.ModuleList()
            self.middle_dropouts = nn.ModuleList()
            for i in range(num_layers - 1):
                self.middle_layers.append(nn.Linear(hidden_dim, hidden_dim))
                self.middle_batch_norms.append(nn.BatchNorm1d(hidden_dim))
                self.middle_activations.append(nn.ReLU())
                self.middle_dropouts.append(nn.Dropout(dropout_rate))

            # Output layer
            self.output_layer = nn.Linear(hidden_dim, output_dim)

            # Xavier initialization
            self._initialize_weights()

        def _initialize_weights(self):
            # Embedding layer
            nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)
            # First linear layer
            nn.init.kaiming_normal_(self.first_layer.weight, nonlinearity='relu')
            if self.first_layer.bias is not None:
                nn.init.constant_(self.first_layer.bias, 0.0)
            
            # Middle layers
            for layer in self.middle_layers:
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)
            
            # Output layer
            nn.init.normal_(self.output_layer.weight, mean=0.0, std=0.01)
            if self.output_layer.bias is not None:
                nn.init.constant_(self.output_layer.bias, 0.0)

        def forward(self, x_continuous, x_categorical):
            # Handle new categories by mapping them to the unknown index
            x_categorical = torch.where(
                x_categorical >= self.num_embeddings, 
                torch.full_like(x_categorical, self.unknown_index),
                x_categorical
            )
            # Embedding lookup
            embedded = self.embedding(x_categorical)
            # Flatten the embedding
            embedded = embedded.view(embedded.size(0), -1)

            # Concatenate continuous features and embeddings
            x = torch.cat((x_continuous, embedded), dim=1)

            # Forward pass through the first layer
            x = self.first_layer(x)
            x = self.first_batch_norm(x)
            x = self.first_activation(x)
            x = self.first_dropout(x)

            # Forward pass through dynamic middle layers
            for i in range(len(self.middle_layers)):
                x = self.middle_layers[i](x)
                x = self.middle_batch_norms[i](x)
                x = self.middle_activations[i](x)
                x = self.middle_dropouts[i](x)

            # Output layer
            x = self.output_layer(x)
            
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
    def train(model, train_loader, loss_function, optimizer, device, num_epochs=10):
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
                x_continuous, x_embedding_vars, targets = \
                    x_continuous.to(device), x_embedding_vars.to(device), targets.to(device)
                outputs = model(x_continuous, x_embedding_vars).squeeze()
                loss = loss_function(outputs, targets)
                total_loss += loss.item()

        average_loss = total_loss / len(test_loader)
        return average_loss

    @staticmethod
    def predict(model, dataset, device, batch_size=256):
        model.eval()
        predictions = []
        # Create DataLoader
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
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
    def train_fnn(config, train_dataset, test_dataset, device, ray_tuning=True, patience=10):
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
        batch_size = config["batch_size"]

        # Update DataLoader creation with dynamic batch size
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        logger.debug(f'Created data_loaders from datasets')
        
        # Early stopping patience
        patience = patience
        best_loss = float('inf')
        epochs_without_improvement = 0

        # Use ModelData.FlexibleNeuralNetwork to reference the nested class
        model = ModelData.FlexibleNeuralNetwork(
            continuous_dim, hidden_dim, output_dim, num_layers, num_embeddings, embedding_dim, dropout_rate
        )
        # Wrap the model with DataParallel
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

        loss_function = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        try:
            for epoch in range(num_epochs):
                model.train()
                running_loss = 0.0
                for x_continuous, x_embedding_vars, targets in train_loader:
                    x_continuous, x_embedding_vars, targets = \
                        x_continuous.to(device), x_embedding_vars.to(device), targets.to(device)

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
                                
                # Early stopping mechanism
                if avg_test_loss < best_loss:
                    best_loss = avg_test_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= patience:
                    if not ray_tuning: # only display this message when not using Ray Tune
                        logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

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
        self, train_dataset, test_dataset, continuous_dim, num_embeddings, device, 
        num_samples=10, max_num_epochs=20, num_cpus=2, num_gpus=0, cpus_per_trial=1, gpus_per_trial=0, patience=10
    ):
        config = {
            "continuous_dim": continuous_dim,
            "hidden_dim": tune.choice([i for i in range(5, 200, 10)]),
            "num_layers": tune.choice([1, 2, 3, 4, 5]),
            "num_embeddings": num_embeddings,
            "embedding_dim": tune.choice([i for i in range(1, 11, 1)]),
            "dropout_rate": tune.choice([round(i * 0.01, 2) for i in range(1, 56)]), # uniform (0.01, 0.55)
            "lr": tune.loguniform(1e-6, 1e-2),
            "weight_decay": tune.loguniform(1e-6, 1e-3),
            "num_epochs": max_num_epochs,
            "num_gpus": num_gpus,
            "batch_size": tune.choice([8, 16, 32, 64, 128, 256]),
        }

        scheduler = ASHAScheduler(
            metric="avg_test_loss",
            mode="min",
            max_t=max_num_epochs,
            grace_period=1,
            reduction_factor=2
        )

        reporter = CLIReporter(
            metric_columns=["average_train_loss", "avg_test_loss", "training_iteration"])

        result = tune.run(
            # May use a large patience number because Ray Tune has early stopping schedule already
            tune.with_parameters(
                ModelData.train_fnn, 
                train_dataset=train_dataset, 
                test_dataset=test_dataset, 
                device=device, 
                ray_tuning=True, 
                patience=patience,
            ),
            resources_per_trial={
                "cpu": cpus_per_trial, 
                "gpu": gpus_per_trial
            },
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
