import torch.nn as nn
import torch.nn.functional as F

class RNNEncoder(nn.Module):
	def __init__(self, hidden_size=100): 
		super().__init__()
		self.gru = nn.GRU(hidden_size, hidden_size)
	def forward(self, sentence): # Takes all input at once, sentence is a tensor
		# (seq_len, batch len, input_size), [1] is the hidden state
		return self.gru(sentence.view(len(sentence), 1, -1))[1] 

class RNNDecoder(nn.Module):
	def __init__(self, hidden_size=100, output_size = 100):
		super().__init__()
		self.gru = nn.GRU(hidden_size, hidden_size)
		self.linear = nn.Linear(hidden_size, output_size)
	def forward(self, word, hidden): # Takes one input at a time
		hidden_state = self.gru(word.view(1, 1, -1), hidden)[1]
		probs = F.softmax(self.linear(hidden)) # i.e. the probs at the t'th step for beam search
		return probs, hidden_state