import torch
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

class RNNEncoder(nn.Module):

	def __init__(self):
		super().__init__()
		self.input_size = 300
		self.hidden_size = 500
		self.gru = nn.GRU(hidden_size, hidden_size, 1) # 1 hidden layer

	def forward(self, input, hidden): # Meant to be called in a for loop over the input sequence
		


class RNNDecoder(nn.Module):

