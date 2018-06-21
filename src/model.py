import torch.nn as nn

class Encoder(nn.Module):
	def __init__(self, embedding_size=500, hidden_size=1000, vocab_size):
		super().__init__()
		self.embedding = nn.Embedding(vocab_size, hidden_size)
		self.gru = nn.GRU(hidden_size, hidden_size)

	def forward(self, sentence):
		embed = self.embedding(sentence) 
		embedded = self.embedding(sentence).view(len(sentence), 1, -1) # (seq_len, batch len, input_size)
		return self.gru(embedded)[1]	

class Decoder(nn.Module):
	def __init__(self, embedding_size = 500, hidden_size=1000, output_size = 300, vocab_size):
		super().__init__()
		self.embedding = nn.Embedding(vocab_size, hidden_size)
		self.gru = nn.GRU(hidden_size, hidden_size)
		self.linear = nn.Linear(hidden_size, output_size)
		self.softmax = nn.Softmax()

	def forward(self, word, hidden): # Takes one input at once
		embedded = self.embedding(word).view(1, 1, -1)
		output, hidden_state = self.gru(embedded, hidden)
		linear = self.linear(output)
		return softmax, hidden_state