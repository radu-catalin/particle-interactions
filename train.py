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

# hyperparams
num_epochs = 10
batch_size = 100
learning_rate = 0.0001
momentum = 0.6
log_interval = int(5000 / batch_size)

n_body = 2

input_size = 5
hidden_size = 500
output_size = 2 # multi-variable regression problem

train_loader = generate_dataset(
	n_body = n_body,
	dataset_size = 100000,
	batch_size = batch_size,
	shuffle = True
)

# examples = iter(train_loader)
# samples, labels = examples.next()
# print(labels)
# print(samples)
# exit(0)
# model
model = GCN(input_size, hidden_size, output_size).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(
	model.parameters(),
	lr = learning_rate,
	momentum = momentum
)

n_total_steps = len(train_loader)
losses_train = []
# training loop
for epoch in range(num_epochs):
	for i, (data, targets) in enumerate(train_loader):
		data = data.to(device)
		targets = targets.to(device)

		A = generate_relational_matrix(data, normalize = True).to(device)

		outputs = model(data, A).to(device)

		if True in torch.isnan(outputs):
			print('NaN detected')
			exit(0)

		# to do:
		# - improve the loss function
		# - improve the network
		# -
		loss = MSELoss_L2(outputs, targets)
		loss.backward()

		# update
		optimizer.step()
		optimizer.zero_grad()

		if (i + 1) % log_interval == 0:
			losses_train.append(loss.detach().cpu().numpy())
			print(
				f'epoch: {epoch + 1} / {num_epochs}, step {i + 1} / {n_total_steps}, loss = {loss.item():.4f}'
			)

print('Finished training!')
plot_loss(losses_train, 'train loss')
plt.show()