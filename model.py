import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from layer import GraphConvolutionLayer

# class GCN(nn.Module):
# 	def __init__(self, input_size, hidden_size, output_size):
# 		super(GCN, self).__init__()

# 		self.conv1 = GraphConvolutionLayer(input_size, hidden_size)
# 		self.conv1_drop = nn.Dropout2d(p=0.5)
# 		self.conv2 = GraphConvolutionLayer(hidden_size, 300)
# 		self.conv2_drop = nn.Dropout2d(p=0.4)
# 		self.conv3 = GraphConvolutionLayer(300, 500)
# 		self.conv3_drop = nn.Dropout2d(p=0.3)

# 		self.linear1 = nn.Linear(500, 300)
# 		self.linear2 = nn.Linear(300, 50)
# 		self.linear3 = nn.Linear(50, output_size)

# 	def forward(self, x, A):
# 		x = F.relu(self.conv1_drop(self.conv1(x, A)))
# 		x = F.relu(self.conv2_drop(self.conv2(x, A)))
# 		x = F.relu(self.conv3_drop(self.conv3(x, A)))
# 		x = F.relu(self.linear1(x))
# 		x = F.relu(self.linear2(x))
# 		x = self.linear3(x)
# 		return x

# 	def zero_grad(self):
# 		self.conv1.zero_grad()
# 		self.conv2.zero_grad()
# 		self.conv3.zero_grad()

class GCN(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(GCN, self).__init__()

		self.conv1 = GraphConvolutionLayer(input_size, hidden_size)
		self.conv2 = GraphConvolutionLayer(hidden_size, 300)

		self.linear1 = nn.Linear(300, hidden_size)
		self.linear2 = nn.Linear(hidden_size, 50)
		self.linear3 = nn.Linear(50, output_size)

	def forward(self, x, A):
		x = F.relu(self.conv1(x, A))
		x = F.relu(self.conv2(x, A))
		x = F.relu(self.linear1(x))
		x = F.relu(self.linear2(x))
		x = self.linear3(x)
		# x = torch.sigmoid(x)
		return x

class GCN_var1(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(GCN_var1, self).__init__()
		self.w1 = Parameter(torch.randn([input_size, hidden_size], requires_grad = True))
		self.w2 = Parameter(torch.randn([hidden_size, 50], requires_grad = True))
		self.w3 = Parameter(torch.randn([50, output_size], requires_grad = True))
		self.reset_parameters()
		self.params = [self.w1, self.w2, self.w3]

	def reset_parameters(self):
		stdv = 1. / math.sqrt(self.w1.size(1))
		self.w1.data.uniform_(-stdv, stdv)
		stdv = 1. / math.sqrt(self.w2.size(1))
		self.w2.data.uniform_(-stdv, stdv)
		stdv = 1. / math.sqrt(self.w3.size(1))
		self.w3.data.uniform_(-stdv, stdv)

	def forward(self, x, A):
		x = F.relu(torch.matmul(A,torch.matmul(x,self.w1)))
		x = F.relu(torch.matmul(A,torch.matmul(x,self.w2)))
		x = torch.matmul(A,torch.matmul(x,self.w3))
		return x

	def zero_grad(self):
		for param in self.params:
			if param.grad is not None:
				param.grad.zero_()