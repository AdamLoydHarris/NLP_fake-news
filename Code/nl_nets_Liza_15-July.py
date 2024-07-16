import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# PyTorch libraries
import torch
import nlp_nets as nlp
from torch import nn
import torch.optim as optim
from os.path import join as opj
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm.auto import tqdm



# data and labels at tensors 
# convert data into dataloaders
# optimiser
# cost function 
# network architeture
# hyperparams
# forward, backward



################ ################ ################ ################ ################
# coding session 15th July
################ ################ ################ ################ ################


# @title Set random seed

# @markdown Executing `set_seed(seed=seed)` you are setting the seed

# For DL its critical to set the random seed so that students can have a
# baseline to compare their results to expected results.
# Read more here: https://pytorch.org/docs/stable/notes/randomness.html

# Call `set_seed` function in the exercises to ensure reproducibility.

SEED = 5
SEED = nlp.set_seed(SEED)
DEVICE = nlp.set_device()

# specify PATH
folder_dir = opj('/', 'Users', 'doctordu', 'Documents', 'Github', 'Neuromatch_NLP', 'dataset')

# specify FILE NAME
arr_fake = np.load(opj(folder_dir, 'fake_emotion_array.npy'))
arr_true = np.load(opj(folder_dir, 'true_emotion_array.npy'))
arr_labels = np.load(opj(folder_dir, 'labels_all.npy'))

data_array = np.concatenate((arr_fake, arr_true))

data_torch = torch.from_numpy(data_array).to(DEVICE)
labels_torch = torch.from_numpy(arr_labels).to(DEVICE)

g_seed = torch.Generator()
#g_seed.manual_seed(SEED)

batch_size = 6000
test_data = TensorDataset(data_torch, labels_torch)
train_size = int(0.8 * len(data_torch))
test_size = len(data_torch) - train_size

train_dataset, test_dataset = random_split(test_data, [train_size, test_size])

test_loader = DataLoader(test_dataset, batch_size=batch_size,
                         shuffle=False, num_workers=0,
                         worker_init_fn=nlp.seed_worker,
                         generator=g_seed)

train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True,
                          shuffle=True, num_workers=0,
                          worker_init_fn=nlp.seed_worker,
                          generator=g_seed)

##############################################
# Define the network
##############################################



nlp.set_seed(SEED)
net = nlp.Net('Sigmoid()', 10, [50], 2).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3)
num_epochs = 1000

_, _ = nlp.train_test_classification(net, criterion, optimizer, train_loader,
                                 test_loader, num_epochs=num_epochs,
                                 training_plot=True, device=DEVICE)
