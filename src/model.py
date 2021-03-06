import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import torch

class RNNEncoder(nn.Module):
	def __init__(self, embedding_size=100, hidden_size=512):
		super().__init__()
		self.gru = nn.GRU(embedding_size, hidden_size)
	def forward(self, sentence): # Takes all input at once, sentence is a tensor
		# (seq_len, batch len, input_size), [1] is the hidden state
		return self.gru(sentence)

class RNNDecoder(nn.Module):
	def __init__(self, embedding_size=100, hidden_size=512, vocab_size = utils.getVocabSize()):
		super().__init__()
		self.gru = nn.GRU(embedding_size, hidden_size)
		self.linear = nn.Linear(hidden_size, vocab_size)
	def forward(self, word, hidden, temperature=1.0, batch_size=10): # Takes one input at a time
		new_hidden = self.gru(word.view(-1, batch_size, 100), hidden)[1]
		probs = F.softmax(self.linear(new_hidden).view(batch_size, -1), dim=1) # NOTE: softmax expects 2-dim input or else everything breaks
		return probs, new_hidden
	'''def tempSoftmax(self, vector, temperature): # softmax with temperature for differentiable decoding
		exp = torch.exp(torch.div(vector.double(), temperature))
		denom = exp.sum()
		return torch.div(exp, denom).float()'''

class RNNAttentionDecoder(nn.Module):
	def __init__(self, embedding_size = 100, hidden_size = 512, vocab_size = utils.getVocabSize(), dropout_p = 0.1, batch_size = 3):
		super().__init__()
		self.hidden_size = hidden_size
		self.vocab_size = vocab_size

		self.linear = nn.Linear(hidden_size, vocab_size)
		self.gru = nn.GRU(hidden_size, hidden_size)
		self.attn = nn.Linear((hidden_size + embedding_size)*batch_size, vocab_size)
		self.attn_combine = nn.Linear((hidden_size + embedding_size), hidden_size)
		self.dropout = nn.Dropout(dropout_p)
		self.output = nn.Linear(hidden_size, vocab_size)

	def forward(self, word, hidden, encoder_output, batch_size = 3, hidden_size = 512, embedding_size = 100):
		wordEmbed = self.dropout(word.contiguous().view(1,1,-1))
		attn_weights_temp = torch.cat((wordEmbed[0],hidden[0].view(1,-1)),1)
		attn_weights = F.softmax(self.attn(attn_weights_temp), dim = 1)
		attn_weights = torch.t(attn_weights)
		print(attn_weights.unsqueeze(0),"\n")
		print(encoder_output[0].unsqueeze(0),"\n")
		print(attn_weights.unsqueeze(0).view(1,-1).size(),"\n")
		print(encoder_output[0][0].unsqueeze(0).size(),"\n")
		attn_toNetwork = torch.bmm(attn_weights.view(-1,1).unsqueeze(0),encoder_output[0].view(1,-1).unsqueeze(0))
		attn_toNetwork = torch.t(attn_toNetwork)
		inp = 10
		for i in str(inp):
			print("\n")
		print(wordEmbed[0],"\n")
		print(attn_toNetwork[0])
		probs_temp = torch.cat((wordEmbed[0],attn_toNetwork[0]),1)
		probs_temp = probs_temp.view(612,-1)
		probs = F.relu(self.attn_combine(probs_temp).unsqueeze(0))
		print(probs,"\n")
		probs, new_hidden = self.gru(probs, hidden)
		probs = F.log_softmax(self.linear(probs[0]).view(batch_size, -1), dim = 1)
		
		return probs, new_hidden, attn_weights