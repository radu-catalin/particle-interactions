import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class GraphConvolutionLayer(nn.Module):
	def __init__(self, input_size: int, output_size: int) -> None:
		super(GraphConvolutionLayer, self).__init__()
		self.input = input_size
		self.output = output_size
		self.w = Parameter(torch.randn([input_size, output_size], requires_grad = True))
		self.reset_parameters()

	def reset_parameters(self):
		stdv = 1. / math.sqrt(self.w.size(1))
		self.w.data.uniform_(-stdv, stdv)

	def forward(self, x, A):
		x = torch.matmul(A, x)
		x = torch.matmul(x, self.w)
		return x

	def zero_grad(self):
		if self.w.grad is not None:
			self.w.grad.zero_()