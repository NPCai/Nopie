import torch.nn as nn
import torch.nn.functional as F
import utils

class RNNEncoder(nn.Module):
	def __init__(self, embedding_size=100, hidden_size=100): # TODO(jacob) use bigger word vectors/increase hidden size
		super().__init__()
		self.gru = nn.GRU(embedding_size, hidden_size)
	def forward(self, sentence): # Takes all input at once, sentence is a tensor
		# (seq_len, batch len, input_size), [1] is the hidden state
		return self.gru(sentence.view(len(sentence), 1, -1))[1] 

class RNNDecoder(nn.Module):
	def __init__(self, embedding_size=100, hidden_size=100, vocab_size = utils.getVocabSize()):
		super().__init__()
		self.gru = nn.GRU(embedding_size, hidden_size)
		self.linear = nn.Linear(hidden_size, vocab_size)
	def forward(self, word, hidden): # Takes one input at a time
		new_hidden = self.gru(word.view(1, 1, -1), hidden)[1]
		probs = F.softmax(self.linear(new_hidden)) # i.e. the probs at the t'th step for beam search
		return probs, new_hidden