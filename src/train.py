import model
import numpy as np
import utils
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

SOS_token = 0 # Start of sentence
EOS_token = 1 # End of sentence

class RNN():

	def __init__(self, input_size, output_size, vocab_size):
		super().__init__()
		self.encoder = RNNEncoder(input_size) 
		self.decoder = RNNDecoder(output_size)
		self.lossFn = nn.CrossEntropyLoss()
		self.encoder_optimizer = optim.Adam(self.encoder.parameters()) 												   
		self.decoder_optimizer = optim.Adam(self.decoder.parameters())
		self.sos, self.eos = torch.zeros(100), torch.ones(100) # start and end special tokens, unk is handled by utils

	def train(self, seqIn, seqOut): 
		''' Train one iteration, no batch '''
		self.encoder_optimizer.zero_grad() 
		self.decoder_optimizer.zero_grad()

		hidden = self.encoder.forward(1, seqIn) # Encode sentence

		# Decoder stuff
		seqOut.insert(0, self.sos) # Insert the start of sentence token into the first position
		seqOut.append(self.eos) # Append the end of sentence token into the end
		loss = 0
		for i in range(len(seqOut) - 1):
			_, softmax, hidden = self.decoder.forward(seqOut[i], hidden)
			loss += self.lossFn(softmax, seqOut[i+1][0]) # Calculating loss

		loss.backward() # Compute grads with respect to the network
		self.encoder_optimizer.step() # Update using the stored grad
		self.decoder_optimizer.step()
		return loss.data[0]

	def evaluation(self, seqIn):
		# Encoder stuff
		_, encoder_hidden = self.encoder.forward(1, encoder_hidden) # Forward propogation to hidden layer

		sentence = []
		seqIn = self.sos

		# Decoder stuff
		while seqIn.data[0,0] != 1:
			output, hidden_state = self.decoder.forward(input, hidden_state)
			word = torch.max(output.data, dim = 1).reshape((1,1))
			input = torch.LongTensor(word)
			sentence.append(word)
		return sentence

	def save(self): # Saving the trained network to a .ckpt file
		torch.save(self.encoder.state_dict(), "encoder.ckpt")
		torch.save(self.decoder.state_dict(), "decoder.ckpt")