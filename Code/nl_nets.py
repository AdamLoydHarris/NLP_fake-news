import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# PyTorch libraries
import torch
from torch import nn
import torch.optim as optim
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from IPython.display import display

# data and labels at tensors 
# convert data into dataloaders
# optimiser
# cost function 
# network architeture
# hyperparams
# forward, backward

def shuffle_and_split_data(X, y, seed=5):
    """
    Helper function to shuffle and split incoming data

    Args:
    X: torch.tensor
        Input data
    y: torch.tensor
        Corresponding target variables
    seed: int
        Set seed for reproducibility

    Returns:
    X_test: torch.tensor
        Test data [20% of X]
    y_test: torch.tensor
        Labels corresponding to above mentioned test data
    X_train: torch.tensor
        Train data [80% of X]
    y_train: torch.tensor
        Labels corresponding to above mentioned train data
    """
    torch.manual_seed(seed)
    # Number of samples
    N = X.shape[0]

    shuffled_indices = torch.randperm(N)  # Get indices to shuffle data, could use torch.randperm
    print(shuffled_indices) 
    X = X[shuffled_indices]
    y = y[shuffled_indices]

    # Split data into train/test
    test_size = int(N*0.2)    # Assign test datset size using 20% of samples
    X_test = X[:test_size]
    y_test = y[:test_size]
    X_train = X[test_size:]
    y_train = y[test_size:]

    return X_test, y_test, X_train, y_train


class emotionNet(nn.Module):
    
    """
    initialise neural net to classify fake news articles based purely on 10-feature emotion scores
    """

    def __init__(self, input_size, layer_sizes, output_size):
        """
        set up network params 
        """
        super(emotionNet, self).__init__()
        self.input_size = input_size # Save the input size 
        self.network = nn.Sequential() # Initialize layers 
        in_num = input_size # Initialize the temporary input feature to each layer\


    def forward(self, x):
        pass

    
