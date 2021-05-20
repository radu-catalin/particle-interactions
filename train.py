import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from model import GCN
from utils import generate_dataset, generate_relational_matrix, plot_loss

# device config
# device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparams
num_epochs = 5
batch_size = 4
learning_rate = 0.001
momentum = 0.9
log_interval = int(1000 / batch_size)

n_body = 3

input_size = 5
hidden_size = 50
output_size = 2 # multi-variable regression problem

train_loader = generate_dataset(
	n_body = n_body,
	dataset_size = 1000,
	batch_size = batch_size,
	shuffle = True
)

# examples = iter(train_loader)
# examples.next()
# samples, labels = examples.next()
# print(samples.shape, labels.shape)

# model
model = GCN(input_size, hidden_size, output_size).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(
	model.parameters(),
	lr = learning_rate,
	momentum = momentum
)

pdist = nn.PairwiseDistance(p = 2)

n_total_steps = len(train_loader)
losses_train = []
# training loop
for epoch in range(num_epochs):
	for i, (data, targets) in enumerate(train_loader):
		data = data.to(device)
		targets = targets

		A = generate_relational_matrix(data, normalize = True).to(device)

		outputs = model(data, A)

		# print(outputs)
		# exit(0)
		# to do:
		#  - make a loss function that applies MSELoss for L2(x,y)
		#  - update
		#  - print for each epoch/log_interval
		# loss = pdist(outputs, targets)
		# loss = F.nll_loss(outputs, targets)
		# loss.backward()

		# # update
		optimizer.step()
		optimizer.zero_grad()

		# if (i + 1) % log_interval == 0:
		# 	print(
		# 		f'epoch: {epoch + 1} / {num_epochs}, step {i + 1} / {n_total_steps}, loss = {loss.item():.4f}'
		# 	)

print('Finished training!')