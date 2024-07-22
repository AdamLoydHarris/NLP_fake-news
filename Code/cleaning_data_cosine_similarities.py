#!pip install torchvision --quiet

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# PyTorch libraries
import torch
#import nlp_nets as nlp
from torch import nn
import torch.optim as optim
from os.path import join as opj
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm.auto import tqdm
from torch.utils.data import TensorDataset
from torch.nn.functional import cosine_similarity
import torch.nn.functional as F

SEED = 5
set_seed(SEED)
DEVICE = set_device()
g_seed = torch.Generator()

# specify PATH
folder_dir = opj('/', 'Users', 'elizavetaparfenova', 'PycharmProjects', 'Neuromatch', 'dataset')

# load word embeddings (full dataset)
embed_data_array_train = torch.from_numpy(np.load(opj(folder_dir, 'train_embeddings_200.npy')))
embed_data_array_test = torch.from_numpy(np.load(opj(folder_dir, 'test_embeddings_200.npy')))
embed_data_array_val = torch.from_numpy(np.load(opj(folder_dir, 'validation_embeddings_200.npy')))

arr_labels_train = torch.from_numpy(np.load(opj(folder_dir, 'train_labels.npy')))
arr_labels_test = torch.from_numpy(np.load(opj(folder_dir, 'test_labels.npy')))
arr_labels_val = torch.from_numpy(np.load(opj(folder_dir, 'validation_labels.npy')))

# Move datasets to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embed_data_array_train = embed_data_array_train.to(device)
embed_data_array_test = embed_data_array_test.to(device)
embed_data_array_val = embed_data_array_val.to(device)
arr_labels_train = arr_labels_train.to(device)
arr_labels_test = arr_labels_test.to(device)
arr_labels_val = arr_labels_val.to(device)

dataset_train_separate = TensorDataset(embed_data_array_train, arr_labels_train)
dataset_test_separate = TensorDataset(embed_data_array_test, arr_labels_test)
dataset_val_separate = TensorDataset(embed_data_array_val, arr_labels_val)

# Combine datasets
all_embeddings = torch.cat((embed_data_array_train, embed_data_array_test, embed_data_array_val), dim=0)
all_labels = torch.cat((arr_labels_train, arr_labels_test, arr_labels_val), dim=0)

###################################################################
# run it in batches to avoid fails
###################################################################
def compute_cosine_similarities(embeddings):
    # returns the size of the first dimension
    num_embeddings = embeddings.size(0)
    # create matrix to store similarities
    similarities = torch.zeros((num_embeddings, num_embeddings), dtype=torch.float32)

    for i in range(num_embeddings):
        # containing the cosine similarities between embeddings[i] and each
        # of the embeddings in embeddings
        similarities[i] = cosine_similarity(embeddings[i].unsqueeze(0), embeddings, dim=1)
    return similarities

# run the cosine similarities defined function
cosine_similarities = compute_cosine_similarities(all_embeddings)

###################################################################
# plot
###################################################################

plt.figure(figsize=(10, 8))
plt.imshow(cosine_similarities.cpu().numpy(), cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.title('Cosine Similarities Matrix')
plt.show()

# Plot cosine similarities
plt.figure(figsize=(10, 6))
plt.hist(cosine_similarities.numpy().flatten(), bins=100, alpha=0.75, color='blue')
plt.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold ({threshold})')
plt.title('Distribution of Cosine Similarities')
plt.xlabel('Cosine Similarity')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()

###################################################################
# treshold
###################################################################


threshold = 0.3  # Define your threshold value

# Extract upper triangle indices (excluding diagonal) from the cosine similarity matrix
upper_triangle_indices = torch.triu_indices(cosine_similarities.size(0), cosine_similarities.size(1), offset=1)
upper_triangle_similarities = cosine_similarities[upper_triangle_indices[0], upper_triangle_indices[1]]

# Find indices where similarity is above the threshold
above_threshold_indices = upper_triangle_similarities > threshold

# Number of data points above threshold
num_above_threshold = above_threshold_indices.sum().item()
print(f"Number of data points above threshold: {num_above_threshold}")

###################################################################
# make a new tensor which marks 0 if it is below or equal to 0.3 treshold
###################################################################
# Create a tensor where each entry is 1 if the max similarity with any other embedding is above the threshold
first_column = cosine_similarities[:, 0]
first_column_np = first_column.numpy()
# then divide it into three original arrays by the length

len_train = len(arr_labels_train)
len_test = len(arr_labels_test)
len_val = len(arr_labels_val)

# Split the array
arr_train = first_column_np[:len_train]
arr_test = first_column_np[len_train:len_train + len_test]
arr_val = first_column_np[len_train + len_test:len_train + len_test + len_val]
# treshold it to 0.3 by 0 and 1 values

def array_treshold(arr):
    treshold_labels = []
    for i in arr:
        if i <= threshold:
            treshold_labels.append(0)
        else:
            treshold_labels.append(1)
    return treshold_labels

train_labels = array_treshold(arr_train)
test_labels = array_treshold(arr_test)
val_labels = array_treshold(arr_val)

# save it as three arrays for each dataset
np.save(opj(folder_dir, 'train_cosine_similarity.npy'), train_labels)
np.save(opj(folder_dir, 'test_cosine_similarity.npy'), test_labels)
np.save(opj(folder_dir, 'val_cosine_similarity.npy'), val_labels)