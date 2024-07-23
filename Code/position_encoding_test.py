# %%
import numpy as np
import matplotlib.pyplot as plt
import torch

#%%
d_pos_vec = 100  #  embedding dimension
n_position = 20  # the max sequence length
hidden_units = 100 # Dimension of embedding
vocab_size = 10 # Maximum sentence length
# Matrix of [[1, ..., 99], [1, ..., 99], ...]
i = np.tile(np.expand_dims(range(hidden_units), 0), [vocab_size, 1])
# Matrix of [[1, ..., 1], [2, ..., 2], ...]
pos = np.tile(np.expand_dims(range(vocab_size), 1), [1, hidden_units])




position_enc = np.array([
    [pos / np.power(10000, 2*i/d_pos_vec) for i in range(d_pos_vec)]
    if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])
position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
matrix =  torch.from_numpy(position_enc).type(torch.FloatTensor)
print(matrix.shape)
im = plt.imshow(matrix, cmap='hot', aspect='auto')
# %%
