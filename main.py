import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from physics import gen

# device config
device = torch.device('cpu')

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

def generate_dataset(dataset_size: int, batch_size: int, shuffle: str) -> DataLoader:
	iterations = int(dataset_size / 100)
	np_dataset = []
	np_targets = []
	for i in range(iterations):
		dataset = gen(n_body, True)
		for j in range(len(dataset) - 1):
			np_dataset.append(np.array(dataset[j]))
			np_targets.append(np.array(dataset[j + 1, :, 0:2]))
	np_dataset = np.array(np_dataset)
	np_targets = np.array(np_targets)

	dataset_frames, dataset_targets = map(
		torch.Tensor, (np_dataset, np_targets)
	)

	dataset_frames = TensorDataset(dataset_frames, dataset_targets)

	dataset_loader = DataLoader(
		dataset = dataset_frames,
		batch_size = batch_size,
		shuffle = True
	)

	return dataset_loader

def plot_loss(loss, label, color='red') -> None:
	plt.plot(loss, label=label, color=color)
	plt.legend()

train_loader = generate_dataset(dataset_size = 1000, batch_size = batch_size, shuffle = True)

# examples = iter(train_loader)
# examples.next()
# samples, labels = examples.next()

# print(samples.shape, labels.shape)

class GraphConvolutionLayer(nn.Module):
	def __init__(self, input_size: int, output_size: int) -> None:
		super(GraphConvolutionLayer, self).__init__()
		self.input = input_size
		self.output = output_size
		self.w = Parameter(torch.FloatTensor(input_size, output_size))
		self.reset_parameters()

	def reset_parameters(self):
		stdv = 1. / math.sqrt(self.w.size(1))
		self.w.data.uniform_(-stdv, stdv)

	def forward(self, x, A):
		x = torch.matmul(A, x)
		x = torch.matmul(x, self.w)
		return x

class GCN(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(GCN, self).__init__()

		self.conv1 = GraphConvolutionLayer(input_size, hidden_size)
		self.conv2 = GraphConvolutionLayer(hidden_size, output_size)

	def forward(self, x, A):
		x = F.relu(self.conv1(x, A))
		# x = F.dropout(x, 0.5, training=self.training)
		x = self.conv2(x, A)
		x = F.relu(x)
		return x

def generate_relational_matrix(x):
	A = np.zeros((x.shape[0], n_body, n_body))
	for i in range(x.shape[0]):
		for j in range(n_body):
			for k  in range(n_body):
				A[i][j][k] = math.sqrt((x[i][j][1] - x[i][j][0]) ** 2 + (x[i][k][1] - x[i][k][0]) ** 2)
	A = torch.Tensor(A)

	return A


# model
model = GCN(input_size, hidden_size, output_size)

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
	for i, (data, labels) in enumerate(train_loader):
		data = data.to(device)
		A = generate_relational_matrix(data)

		outputs = model(data, A)

		# to do:
		#  - make a loss function that applies MSELoss for L2(x,y)
		#  - update
		#  - print for each epoch/log_interval
		loss = pdist(outputs, labels)

		loss.backward()

		# update
		optimizer.step()
		optimizer.zero_grad()

		if (i + 1) % log_interval == 0:
			print(
				f'epoch: {epoch + 1} / {num_epochs}, step {i + 1} / {n_total_steps}, loss = {loss.item():.4f}'
			)

print('Finished training!')