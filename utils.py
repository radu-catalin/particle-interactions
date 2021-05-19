import math
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

from physics import gen

def generate_dataset(n_body: int, dataset_size: int, batch_size: int, shuffle: str) -> DataLoader:
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

def generate_relational_matrix(x):
	A = np.zeros((x.shape[0], n_body, n_body))
	for i in range(x.shape[0]): # i = 0, batch_size - 1
		for j in range(x.shape[1]): # j = 0, n_body - 1
			for k  in range(x.shape[1]): # k = 0, n_body - 1
				# distance between particle j and particle k
				A[i][j][k] = math.sqrt((x[i][j][1] - x[i][j][0]) ** 2 + (x[i][k][1] - x[i][k][0]) ** 2)
	A = torch.Tensor(A)

	return A

def plot_loss(loss, label, color='red') -> None:
	plt.plot(loss, label=label, color=color)
	plt.legend()