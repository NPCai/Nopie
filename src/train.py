import model
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

SOS_token = 0 # Start of sentence
EOS_token = 1 # End of sentence

class RNN(object):
	def __init__(self, input_size, output_size):
		super(RNN, self).__init__()

		self.encoder = RNNEncoder(input_size) # Defining the encoder
		self.decoder = RNNDecoder(output_size) # Defining the decoder

		self.loss = nn.CrossEntropyLoss() # Loss function also known as log loss
		self.encoder_optimizer = optim.Adam(self.encoder.parameters()) # Adam optimizer is a stochastic gradient descent which maintains a single learning rate
																	   # for all weight updates and does not change during training. A learning rate is maintained
																	   # for each network weight and separately adapted as learning progresses
		self.decoder_optimizer = optim.Adam(self.decoder.parameters())

		sos, eos = torch.LongTensor(1, 1).zero_(), torch.LongTensor(1, 1).zero_()
		sos[0,0], eos[0,0] = 0, 1 # Defines the SOS and EOS as 1x1 tensors with values 0 and 1 respectively

		self.sos, self.eos = sos, eos


def train(self, input, target):
	self.encoder_optimizer.zero_grad()
	self.decoder_optimizer.zero_grad()

	# Encoder stuff
	_, encoder_hidden = self.encoder.forward(1, encoder_hidden)

	# Decoder stuff
	target.insert(0, self.sos)
	target.append(self.eos)
	loss = 0
	for i in range(len(target) - 1):
		_, softmax, encoder_hidden = self.decoder.forward(target[i], encoder_hidden)
		loss += self.loss(softmax, target[i+1][0])

	loss.backward()

	self.encoder_optimizer.step()
	self.decoder_optimizer.step()
	return loss.data[0]

def evaluation(self, input):
	# Encoder stuff
	_, encoder_hidden = self.encoder.forward(1, encoder_hidden)

	sentence = []
	input = self.sos


def save(self):
	torch.save(self.encoder.state_dict(), "encoder.ckpt")
	torch.save(self.decoder.state_dict(), "decoder.ckpt")
