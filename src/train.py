import model
import numpy as np
import utils
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

class EncoderDecoder():
	def __init__(self):
		super().__init__()
		self.encoder = RNNEncoder() 
		self.decoder = RNNDecoder()
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
		torch.save(self.encoder.state_dict(), "RNNencoder.ckpt")
		torch.save(self.decoder.state_dict(), "RNNdecoder.ckpt")