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
		self.n_layers = 1
		self.hidden_size = 500
		self.embedding = nn.Embedding(VOCAB_SIZE, hidden_size)
		self.gru = nn.GRU(hidden_size, hidden_size, n_layers) # 1 hidden layer

	def forward(self, input_var, hidden): 
		embed = self.embedding(input_var)
		seq_len = len(word_inputs)
		embedded = self.embedding(word_inputs).view(seq_len, 1, -1)
		output, hidden = self.gru(embedded, hidden)
		return output, hidden

	def initHidden(self):
		hidden = torch.zeroes(self.n_layers, 1, self.hidden_size)
		if USE_CUDA: hidden = hidden.cuda()
		return hidden
		

class RNNDecoder(nn.Module):
	def __init__(self, hidden_size, output_size):
		super(RNNDecoder, self).__init__()
		self.hidden_size = hidden_size

		self.embedding = nn.Embedding
		self.gru = nn.GRU(hidden_size, hidden_size)
		self.out = nn.Linear(initHidden_size, output_size)
		self.softmax = nn.LogSoftmax(dim=1)

	def forward(self, input, hidden):
		output = self.embedding(input).view(1,1,-1) # Reshape
		output = F.relu(output)
		output, hidden = self.gru(output,hidden)
		output = self.softmax(self.out(output[0]))
		return output, hidden

	def initHidden(self):
		hidden = torch.zeroes(self.n_layers, 1, self.hidden_size)
		if USE_CUDA: hidden = hidden.cuda()
		return hidden
