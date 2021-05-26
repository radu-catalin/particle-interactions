import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib
from physics import make_video

import matplotlib.pyplot as plt

from model import GCN, GCN_var1
from utils import device, generate_dataset, generate_relational_matrix, plot_loss

# hyperparams
num_epochs = 10
batch_size = 200
learning_rate = 0.001
momentum = 0.9
log_interval = int(1000 / batch_size)

n_body = 3

input_size = 5
hidden_size = 100
output_size = 2	# multi-variable regression problem

train_loader = generate_dataset(
	n_body=n_body,
	dataset_size=600000,
	batch_size=batch_size,
	shuffle=True

)

test_loader = generate_dataset(
	n_body=n_body,
	dataset_size=50000,
	batch_size=batch_size,
	shuffle=False
)

model = GCN(input_size, hidden_size, output_size).to(device)

optimizer = torch.optim.SGD(
	model.parameters(),
	lr=learning_rate,
	momentum=momentum
)

criterion = nn.L1Loss()
n_total_steps = len(train_loader)
losses_train = []
losses_test = []
dots = []
# training loop
for epoch in range(num_epochs):
	for i, (data, targets) in enumerate(train_loader):
		data = data.to(device)
		targets = targets.to(device)

		A = generate_relational_matrix(data, normalize=True).to(device)

		outputs = model(data, A).to(device)

		if True in torch.isnan(outputs):
			print('NaN detected')
			exit(0)

		loss = criterion(outputs, targets)
		loss.backward()

		# update
		optimizer.step()
		optimizer.zero_grad()

		if (i + 1) % log_interval == 0:
			# print(model.w1)
			losses_train.append(loss.detach().cpu().numpy())
			print(
				f'epoch: {epoch + 1} / {num_epochs}, step {i + 1} / {n_total_steps}, loss = {loss.item():.4f}'
			)

	model.eval()
	test_loss = 0
	with torch.no_grad():
		num_iter = 0
		for i, (data, target) in enumerate(test_loader):
			data, target = data.to(device), target.to(device)
			# obtain the prediction by a forward pass
			A = generate_relational_matrix(data, normalize=True).to(device)
			output = model(data, A)
			# calculate the loss for the current batch and add it across the entire dataset
			test_loss += criterion(output, target)	# sum up batch loss
			num_iter += 1

			if i == len(test_loader) - 1 and epoch >= num_epochs - 4 :
				for i in range(output.shape[0]):
					dots.append(output[i,:,:].detach().cpu().numpy())
				make_video(dots, f'video{num_epochs - epoch}.mp4')
	test_loss /= num_iter
	print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
	losses_test.append(test_loss.detach().cpu().numpy())
print('Finished training!')

plot_loss(losses_train, 'train loss')
plt.figure(2)
plot_loss(losses_test, 'test_loss', color='blue')
plt.show()

# plot_loss(accuracy_test,'test_accuracy')
