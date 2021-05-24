import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib

import matplotlib.pyplot as plt

from model import GCN
from utils import device, MSELoss_L2, generate_dataset, generate_relational_matrix, plot_loss
from layer import GraphConvolutionLayer

# hyperparams
num_epochs = 10
batch_size = 100
learning_rate = 0.0001
momentum = 0.6
log_interval = int(5000 / batch_size)

n_body = 2

input_size = 5
hidden_size = 1000
output_size = 2 # multi-variable regression problem

train_loader = generate_dataset(
	n_body = n_body,
	dataset_size = 100000,
	batch_size = batch_size,
	shuffle = True
)

examples = iter(train_loader)
samples, labels = examples.next()
print(samples.shape)
A = generate_relational_matrix(samples, normalize = True).to(device)
conv1 = GraphConvolutionLayer(input_size, 100)
conv1_drop = nn.Dropout2d(p=0.5)
conv2 = GraphConvolutionLayer(100, 300)
conv2_drop = nn.Dropout2d(p=0.4)
conv3 = GraphConvolutionLayer(300, 500)
conv3_drop = nn.Dropout2d(p=0.3)
linear1 = nn.Linear(500, 300)
linear2 = nn.Linear(300, 50)
linear3 = nn.Linear(50, output_size)

x = F.relu(conv1_drop(conv1(samples, A)))
print(x.shape)
x = F.relu(conv2_drop(conv2(x, A)))
print(x.shape)
x = F.relu(conv3_drop(conv3(x, A)))
print(x.shape)
x = F.relu(linear1(x))
print(x.shape)
x = F.relu(linear2(x))
print(x.shape)
x = linear3(x)
print(x.shape)
exit(0)