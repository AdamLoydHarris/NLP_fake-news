import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# PyTorch libraries
import torch
from torch import nn
import torch.optim as optim
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, TensorDataset, random_split
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

################ ################ ################ ################ ################
# coding session 15th July
################ ################ ################ ################ ################


# @title Set random seed

# @markdown Executing `set_seed(seed=seed)` you are setting the seed

# For DL its critical to set the random seed so that students can have a
# baseline to compare their results to expected results.
# Read more here: https://pytorch.org/docs/stable/notes/randomness.html

# Call `set_seed` function in the exercises to ensure reproducibility.

def set_seed(seed=None, seed_torch=True):
  """
  Function that controls randomness. NumPy and random modules must be imported.

  Args:
    seed : Integer
      A non-negative integer that defines the random state. Default is `None`.
    seed_torch : Boolean
      If `True` sets the random seed for pytorch tensors, so pytorch module
      must be imported. Default is `True`.

  Returns:
    Nothing.
  """
  if seed is None:
    seed = np.random.choice(2 ** 32)
  random.seed(seed)
  np.random.seed(seed)
  if seed_torch:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

  print(f'Random seed {seed} has been set.')


# In case that `DataLoader` is used
def seed_worker(worker_id):
  """
  DataLoader will reseed workers following randomness in
  multi-process data loading algorithm.

  Args:
    worker_id: integer
      ID of subprocess to seed. 0 means that
      the data will be loaded in the main process
      Refer: https://pytorch.org/docs/stable/data.html#data-loading-randomness for more details

  Returns:
    Nothing
  """

  # @title Set device (GPU or CPU). Execute `set_device()`
  # especially if torch modules used.

  # Inform the user if the notebook uses GPU or CPU.
  # NOTE: This is mostly a GPU free tutorial.

  def set_device():
      """
      Set the device. CUDA if available, CPU otherwise

      Args:
        None

      Returns:
        Nothing
      """
      device = "cuda" if torch.cuda.is_available() else "cpu"
      if device != "cuda":
          print("GPU is not enabled in this notebook. \n"
                "If you want to enable it, in the menu under `Runtime` -> \n"
                "`Hardware accelerator.` and select `GPU` from the dropdown menu")
      else:
          print("GPU is enabled in this notebook. \n"
                "If you want to disable it, in the menu under `Runtime` -> \n"
                "`Hardware accelerator.` and select `None` from the dropdown menu")

      return device

  worker_seed = torch.initial_seed() % 2**32
  np.random.seed(worker_seed)
  random.seed(worker_seed)

  SEED = 2021
  set_seed(seed=SEED)
  DEVICE = set_device()

from os.path import join as opj

# specify PATH
folder_dir = opj('/', 'Users', 'elizavetaparfenova', 'PycharmProjects', 'Neuromatch_NLP', 'dataset')

# specify FILE NAME
arr_fake = np.load(opj(folder_dir, 'fake_emotion_array.npy'))
arr_true = np.load(opj(folder_dir, 'true_emotion_array.npy'))
arr_labels = np.load(opj(folder_dir, 'labels_all.npy'))

data_array = np.concatenate((arr_fake, arr_true))

data_torch = torch.from_numpy(data_array).to(DEVICE)
labels_torch = torch.from_numpy(arr_labels).to(DEVICE)

g_seed = torch.Generator()
g_seed.manual_seed(SEED)

batch_size = 6000
test_data = TensorDataset(data_torch, labels_torch)
train_size = int(0.8 * len(data_torch))
test_size = len(data_torch) - train_size

train_dataset, test_dataset = random_split(test_data, [train_size, test_size])

test_loader = DataLoader(test_dataset, batch_size=batch_size,
                         shuffle=False, num_workers=0,
                         worker_init_fn=seed_worker,
                         generator=g_seed)

train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True,
                          shuffle=True, num_workers=0,
                          worker_init_fn=seed_worker,
                          generator=g_seed)

##############################################
# Define the network
##############################################

class Net(nn.Module):
  """
  Initialize MLP Network
  """

  def __init__(self, actv, input_feature_num, hidden_unit_nums, output_feature_num):
    """
    Initialize MLP Network parameters

    Args:
      actv: string
        Activation function
      input_feature_num: int
        Number of input features
      hidden_unit_nums: list
        Number of units per hidden layer, list of integers
      output_feature_num: int
        Number of output features

    Returns:
      Nothing
    """
    super(Net, self).__init__()
    self.input_feature_num = input_feature_num # Save the input size for reshaping later
    self.mlp = nn.Sequential() # Initialize layers of MLP

    in_num = input_feature_num # Initialize the temporary input feature to each layer
    for i in range(len(hidden_unit_nums)): # Loop over layers and create each one

      out_num = hidden_unit_nums[i] # Assign the current layer hidden unit from list
      layer = nn.Linear(in_num, out_num) # Use nn.Linear to define the layer
      in_num = out_num # Assign next layer input using current layer output
      self.mlp.add_module('Linear_%d'%i, layer) # Append layer to the model with a name

      actv_layer = eval('nn.%s'%actv) # Assign activation function (eval allows us to instantiate object from string)
      self.mlp.add_module('Activation_%d'%i, actv_layer) # Append activation to the model with a name

    out_layer = nn.Linear(in_num, output_feature_num) # Create final layer
    self.mlp.add_module('Output_Linear', out_layer) # Append the final layer

  def forward(self, x):
    """
    Simulate forward pass of MLP Network

    Args:
      x: torch.tensor
        Input data

    Returns:
      logits: Instance of MLP
        Forward pass of MLP
    """
    # Reshape inputs to (batch_size, input_feature_num)
    # Just in case the input vector is not 2D, like an image!
    x = x.view(-1, self.input_feature_num)

    logits = self.mlp(x) # Forward pass of MLP
    return logits

def train_test_classification(net, criterion, optimizer, train_loader,
                              test_loader, num_epochs=1, verbose=True,
                              training_plot=False, device=DEVICE):
  """
  Accumulate training loss/Evaluate performance

  Args:
    net: instance of Net class
      Describes the model with ReLU activation, batch size 128
    criterion: torch.nn type
      Criterion combines LogSoftmax and NLLLoss in one single class.
    optimizer: torch.optim type
      Implements Adam algorithm.
    train_loader: torch.utils.data type
      Combines the train dataset and sampler, and provides an iterable over the given dataset.
    test_loader: torch.utils.data type
      Combines the test dataset and sampler, and provides an iterable over the given dataset.
    num_epochs: int
      Number of epochs [default: 1]
    verbose: boolean
      If True, print statistics
    training_plot=False
      If True, display training plot
    device: string
      CUDA/GPU if available, CPU otherwise

  Returns:
    Nothing
  """
  net.train()
  training_losses = []
  for epoch in tqdm(range(num_epochs)):  # Loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
      # Get the inputs; data is a list of [inputs, labels]
      inputs, labels = data
      inputs = inputs.to(device).float()
      labels = labels.to(device).long()

      # Zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      outputs = net(inputs)

      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      # Print statistics
      if verbose:
        training_losses += [loss.item()]

  net.eval()

  def test(data_loader):
    """
    Function to gauge network performance

    Args:
      data_loader: torch.utils.data type
      Combines the test dataset and sampler, and provides an iterable over the given dataset.

    Returns:
      acc: float
        Performance of the network
      total: int
        Number of datapoints in the dataloader
    """
    correct = 0
    total = 0
    for data in data_loader:
      inputs, labels = data
      inputs = inputs.to(device).float()
      labels = labels.to(device).long()

      outputs = net(inputs)
      _, predicted = torch.max(outputs, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    return total, acc

  train_total, train_acc = test(train_loader)
  test_total, test_acc = test(test_loader)

  if verbose:
    print(f"Accuracy on the {train_total} training samples: {train_acc:0.2f}")
    print(f"Accuracy on the {test_total} testing samples: {test_acc:0.2f}")

  if training_plot:
    plt.plot(training_losses)
    plt.xlabel('Batch')
    plt.ylabel('Training loss')
    plt.show()

  return train_acc, test_acc



set_seed(SEED)
net = Net('Sigmoid()', 10, [50], 2).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3)
num_epochs = 1000

_, _ = train_test_classification(net, criterion, optimizer, train_loader,
                                 test_loader, num_epochs=num_epochs,
                                 training_plot=True, device=DEVICE)
