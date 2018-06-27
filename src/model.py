import torch.nn as nn
import torch.nn.functional as F
import utils

class RNNEncoder(nn.Module):
	def __init__(self, embedding_size=100, hidden_size=512):
		super().__init__()
		self.gru = nn.GRU(embedding_size, hidden_size)
	def forward(self, sentence): # Takes all input at once, sentence is a tensor
		# (seq_len, batch len, input_size), [1] is the hidden state
		return self.gru(sentence.view(len(sentence), 1, -1))[1] 

class RNNDecoder(nn.Module):
	def __init__(self, embedding_size=100, hidden_size=512, vocab_size = utils.getVocabSize()):
		super().__init__()
		self.gru = nn.GRU(embedding_size, hidden_size)
		self.linear = nn.Linear(hidden_size, vocab_size)
	def forward(self, word, hidden): # Takes one input at a time
		new_hidden = self.gru(word.view(1, 1, -1), hidden)[1]
		probs = F.softmax(self.linear(new_hidden).view(1,-1)) # NOTE: softmax expects 2-dim input or else everything breaks
		return probs, new_hidden

class RNNAttentionDecoder(nn.Module): # TODO: add coverage penalty to train after implementing
	def __init__(self, hidden_size = 512, vocab_size = utils.getVocabSize(), dropout_p = 0.1):
		super().__init__()
		self.gru = nn.GRU(hidden_size, hidden_size)
		self.attn = nn.Linear(hidden_size*2,)
		self.attn_combine
		self.dropout = nn.Dropout(self.dropout_p)

	def forward(self, word, hidden, seqOut):
		pass