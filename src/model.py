import string
import random

import torch
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

VOCAB_SIZE = 1e6

class RNNEncoder(nn.Module):

	def __init__(self):
		super().__init__()
		self.hidden_size = 500
		self.embedding = nn.Embedding(VOCAB_SIZE, hidden_size)
		self.gru = nn.GRU(hidden_size, hidden_size, 1) # 1 hidden layer

	def forward(self, input, hidden): # Meant to be called in a for loop over the input sequence
		# TODO(jacob)
		pass

class RNNDecoder(nn.Module):
	def __init__(self, hidden_size, output_size):
		super(RNNDecoder, self).__init__()
		self.hidden_size = hidden_sized

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
