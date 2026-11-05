import random
from dataclasses import dataclass
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

from ucimlrepo import fetch_ucirepo

def set_seed(seed=42):
    # Python built-in random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch (CPU & GPU)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Falls du Multi-GPU nutzt

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class Dataset:
  name: str
  num_feature: int
  train_tensor: torch.tensor
  test_tensor: torch.tensor
  train_dataset: torch.utils.data.Dataset
  test_dataset: torch.utils.data.Dataset
  train_dataloader: torch.utils.data.DataLoader
  test_dataloader: torch.utils.data.DataLoader



def get_cancer_data():
    DATASETS = []
 
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

    X = breast_cancer_wisconsin_diagnostic.data.features

    target_cols = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]

    for col in target_cols:
        X[col] = pd.factorize(X[col])[0]

    train = torch.tensor(X.values, dtype=torch.float)
    test = torch.tensor(X.values, dtype=torch.float)

    train = train + torch.randn_like(train) * 1e-4
    test = test + torch.randn_like(test) * 1e-4

    mu = train.mean(0, keepdim=True)
    sd = train.std(0, keepdim=True).clamp_min(1e-6)
    train = (train - mu) / sd
    test = (test - mu) / sd

    train_dataset = TensorDataset(train)
    test_dataset = TensorDataset(test)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    num_feature = train.shape[1]

    DATASETS.append(Dataset(name='Cancer', num_feature=num_feature,
                            train_tensor=train, test_tensor=test, train_dataset=train_dataset,
                            test_dataset=test_dataset, train_dataloader=train_dataloader,
                            test_dataloader=test_dataloader))

    return DATASETS


import statistics

def print_model_statistics(models):
    """
    Computes and prints the mean, median, and variance of model metrics 
    grouped by model type across different seeds.
    """
    # Group metrics by model name
    model_data = {}
    for entry in models:
        model_name = entry["model"]
        if model_name not in model_data:
            model_data[model_name] = {
                "train_likelihood": [],
                "test_likelihood": [],
                "structure_learned [t]": []
            }
        
        for metric in model_data[model_name]:
            if metric in entry:
                model_data[model_name][metric].append(entry[metric])

    print("\n" + "="*60)
    print(f"{'STATISTICAL SUMMARY (Mean / Median / Variance)':^60}")
    print("="*60)

    for model_name, metrics in model_data.items():
        print(f"\nModel: {model_name}")
        print("-" * 40)
        for metric, values in metrics.items():
            if not values:
                continue
            
            mean_val = statistics.mean(values)
            median_val = statistics.median(values)
            # Variance requires at least 2 data points; use 0.0 if only 1 value exists
            var_val = statistics.variance(values) if len(values) > 1 else 0.0
            
            print(f"  • {metric:<22}: Mean = {mean_val:.4f}  |  Median = {median_val:.4f}  |  Variance = {var_val:.4f}")
    
    print("="*60)