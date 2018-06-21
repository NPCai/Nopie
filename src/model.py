import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
	def __init__(self, embedding_size=500, hidden_size=1000, vocab_size):
		super().__init__()
		self.embedding = nn.Embedding(vocab_size, hidden_size)
		self.gru = nn.GRU(hidden_size, hidden_size)

	def forward(self, sentence): # Takes all input at once
		embedded = self.embedding(sentence).view(len(sentence), 1, -1) # (seq_len, batch len, input_size)
		return self.gru(embedded)[1] # [1] is the hidden state, can throw away output i.e. [0]

class Decoder(nn.Module):
	def __init__(self, embedding_size = 500, hidden_size=1000, output_size = 300, vocab_size):
		super().__init__()
		self.embedding = nn.Embedding(vocab_size, hidden_size)
		self.gru = nn.GRU(hidden_size, hidden_size)
		self.linear = nn.Linear(hidden_size, output_size)

	def forward(self, word, hidden): # Takes one input at a time
		embedded = self.embedding(word).view(1, 1, -1)
		hidden_state = self.gru(embedded, hidden)[1]
		probs = F.softmax(self.linear(hidden)) # i.e. the probs at the t'th step for beam search
		return probs, hidden_state