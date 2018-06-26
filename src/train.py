import model
import numpy as np
import utils
import torch
import torch.nn as nn
from torch import optim
from model import *
import torch.nn.functional as F
import utils
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderDecoder():
	def __init__(self):
		super().__init__()
		self.encoder = RNNEncoder().to(device)
		self.decoder = RNNDecoder().to(device)
		self.lossFn = nn.NLLLoss()
		self.loss = 0
		self.encoder_optimizer = optim.Adam(self.encoder.parameters()) 												   
		self.decoder_optimizer = optim.Adam(self.decoder.parameters())

	def train(self, seqIn, seqOutOneHot, seqOutEmbedding): 
		''' Train one iteration, no batch '''
		hidden = self.encoder(seqIn) # Encode sentence
		# Decoder stuff
		
		#for i in range(len(seqOutOneHot) - 1):
		#	softmax, hidden = self.decoder(seqOutEmbedding[i], hidden)
		#	self.loss += self.lossFn(softmax.view(1,-1), utils.onehot(seqOutOneHot[i+1]).view(1,-1)) # Calculating loss
		for i in range(len(seqOutOneHot) - 1):
			softmax, hidden = self.decoder(seqOutEmbedding[i], hidden)
			self.loss += self.lossFn(softmax.view(1,-1), torch.tensor([seqOutOneHot[i+1]]).to(device))

	def backprop(self):
		before = time.time()
		print("doing backward")
		self.loss = self.loss / 5
		self.loss.backward() # Compute grads with respect to the network
		print("doing encoder step")
		self.encoder_optimizer.step() # Update using the stored grad
		print("doing decoder step")
		self.decoder_optimizer.step()
		self.encoder_optimizer.zero_grad() 
		self.decoder_optimizer.zero_grad()
		reportedLoss = self.loss.item()
		self.loss = 0
		after = time.time()
		return reportedLoss, (after - before)

	'''def evaluation(self, seqIn):
		# Encoder stuff
		_, encoder_hidden = self.encoder(encoder_hidden) # Forward propogation to hidden layer

		sentence = []
		seqIn = self.sos

		# Decoder stuff
		while seqIn.data[0,0] != 1:
			output, hidden_state = self.decoder(input, hidden_state)
			word = torch.max(output.data, dim = 1).reshape((1,1))
			input = torch.LongTensor(word)
			sentence.append(word)
		return sentence'''

	def save(self): # Saving the trained network to a .ckpt file
		torch.save(self.encoder.state_dict(), "RNNencoder.ckpt")
		torch.save(self.decoder.state_dict(), "RNNdecoder.ckpt")