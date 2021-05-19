import torch
import torch.nn as nn
import torch.nn.functional as F

from layer import GraphConvolutionLayer

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