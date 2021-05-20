import math
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

from physics import gen

# device config
device = torch.device('cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def MSELoss_L2(predicted, target):
	predicted_sum = np.array([], dtype=np.float32)

	for i in range(target.shape[0]):
		predicted_sum = np.append(predicted_sum, [0])
		for j in range(target.shape[1]):
			predicted_sum[i] += math.sqrt((predicted[i][j][0] - target[i][j][0]) ** 2 + (predicted[i][j][1] - target[i][j][1]) ** 2)
		predicted_sum[i] = predicted_sum[i].mean()

	predicted_sum = torch.tensor(predicted_sum, requires_grad = True)

	return (predicted_sum).mean()

def generate_dataset(n_body: int, dataset_size: int, batch_size: int, shuffle: str) -> DataLoader:
	iterations = int(dataset_size / 100)
	np_dataset = []
	np_targets = []
	for i in range(iterations):
		dataset = gen(n_body, True)
		for j in range(len(dataset) - 1):
			np_dataset.append(np.array(dataset[j]))
			np_targets.append(np.array(dataset[j + 1, :, 0:2]))
	np_dataset = np.array(np_dataset, dtype=np.float32)
	np_targets = np.array(np_targets, dtype=np.float32)

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

def generate_relational_matrix(x: torch.Tensor, normalize: bool = False) -> torch.Tensor:
	A = np.zeros((x.shape[0], x.shape[1], x.shape[1]), dtype=np.float32)
	for i in range(x.shape[0]): # i = 0, batch_size - 1
		for j in range(x.shape[1]): # j = 0, n_body - 1
			for k  in range(x.shape[1]): # k = 0, n_body - 1
				# distance between particle j and particle k
				A[i][j][k] = math.sqrt((x[i][j][1] - x[i][j][0]) ** 2 + (x[i][k][1] - x[i][k][0]) ** 2)
	A = torch.Tensor(A)
	if normalize:
		lines_sum = A.sum(dim=1)
		columns_sum = A.sum(dim=2)
		maximum = A.max()
		minimum = A.min()
		A_normalized = np.zeros((x.shape[0], x.shape[1], x.shape[1]), dtype=np.float32)

		for i in range(x.shape[0]):
			for j in range(x.shape[1]):
				for k in range(x.shape[1]):
					A_normalized[i][j][k] = (A[i][j][k] - minimum) / (minimum - maximum)

		A = torch.Tensor(A_normalized)

	# print(A)
	# exit()
	return A

def plot_loss(loss, label, color='red') -> None:
	plt.plot(loss, label=label, color=color)
	plt.legend()