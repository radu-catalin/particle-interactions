import torch
import torch.nn as nn
import torch.nn.functional as F

from layer import GraphConvolutionLayer

class GCN(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(GCN, self).__init__()

		self.linear = nn.Linear(input_size, hidden_size)
		self.conv = GraphConvolutionLayer(hidden_size, output_size)

	def forward(self, x, A):
		x = self.linear(x)
		x = self.conv(x, A)
		x = F.relu(x)
		return x