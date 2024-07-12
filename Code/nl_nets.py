import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# PyTorch libraries
import torch
from torch import nn
import torch.optim as optim
from torchvision.transforms import ToTensor



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
        in_num = input_size # Initialize the temporary input feature to each layer

    def forward(self, x):
        pass

    