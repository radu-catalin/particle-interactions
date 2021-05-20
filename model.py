import torch
import torch.nn as nn
import torch.nn.functional as F

from layer import GraphConvolutionLayer

class GCN(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(GCN, self).__init__()

		self.linear1 = nn.Linear(input_size, hidden_size)
		self.linear2 = nn.Linear(hidden_size, 30)
		self.conv1 = GraphConvolutionLayer(30, 50)
		self.conv2 = GraphConvolutionLayer(50, output_size)

	def forward(self, x, A):
		x = self.linear1(x)
		x = self.linear2(x)
		x = self.conv1(x, A)
		x = F.relu(self.conv2(x, A))
		x = torch.sigmoid(x)
		return x