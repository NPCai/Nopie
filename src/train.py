import model
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

teacher_forcing_ratio = 0.5
SOS_token = 0
EOS_token = 1

class RNN(object):
	def __init__(self, input_size, output_size):
		super(RNN, self).__init__()

		self.encoder = RNNEncoder(input_size)
		self.decoder = RNNDecoder(output_size)

		self.loss = nn.CrossEntropyLoss()
		self.encoder_optimizer = optim.Adam(self.encoder.parameters())
		self.decoder_optimizer = optim.Adam(self.decoder.parameters())

		sos, eos = torch.LongTensor(1, 1).zero_(), torch.LongTensor(1, 1).zero_()
		sos[0,0], eos[0,0] = 0, 1

		self.sos, self.eos = sos, eos


def train(self, input, target):
	encoder_hidden = self.encoder.initHidden()	
	self.encoder_optimizer.zero_grad()
	self.decoder_optimizer.zero_grad()


	target.insert(0, self.sos)
	target.append(self.eos)
	loss = 0
	for i in range(len(target) - 1):
		_, softmax, hidden_state = self.decoder.forward(target[i], hidden_state)
		loss += self.loss(softmax, target[i+1][0])

	loss.backward()

	self.encoder_optimizer.step()
	self.decoder_optimizer.step()
	return loss.data[0]
def evaluation
