import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class GraphConvolutionLayer(nn.Module):
	def __init__(self, input_size: int, output_size: int) -> None:
		super(GraphConvolutionLayer, self).__init__()
		self.input = input_size
		self.output = output_size
		self.w = Parameter(torch.FloatTensor(input_size, output_size))

	def forward(self, x, A):
		x = torch.matmul(A, x)
		x = torch.matmul(x, self.w)
		return x