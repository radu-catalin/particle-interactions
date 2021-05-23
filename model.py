import torch
import torch.nn as nn
import torch.nn.functional as F

from layer import GraphConvolutionLayer

class GCN(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(GCN, self).__init__()

		self.linear1 = nn.Linear(input_size, hidden_size)
		self.linear2 = nn.Linear(hidden_size, 50)
		self.linear3 = nn.Linear(50, 200)
		self.conv1 = GraphConvolutionLayer(200, 500)
		self.conv2 = GraphConvolutionLayer(500, output_size)

	def forward(self, x, A):
		x = F.relu(self.linear1(x))
		x = F.relu(self.linear2(x))
		x = self.linear3(x)
		x = self.conv1(x, A)
		x = F.relu(self.conv2(x, A))
		# x = torch.sigmoid(x)
		return x