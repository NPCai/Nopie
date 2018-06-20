import string
import random

import torch
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

class RNNEncoder(nn.Module):
	
	def __init__(self):


class RNNDecoder(nn.Module):
	def __init__(self, hidden_size, output_size):
		super(RNNDecoder, self).__init__()
		self.hidden_size = hidden_size

		self.embedding = nn.Embedding
		self.gru = nn.GRU(hidden_size, hidden_size)
		self.out = nn.Linear(hidden_size, output_size)
		self.softmax = nn.LogSoftmax(dim=1)

	def forward(self, input, hidden):
		output = self.embedding(input).view(1,1,-1) # Reshape
		output = F.relu(output)
		output, hidden = self.gru(output,hidden)
		output = self.softmax(Self.out(output[0]))
		return output, hidden

	def initHidden(self):
		return torch.zeros(1,1, self.hidden_size, device = device)

