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
		self.lossFn = nn.CrossEntropyLoss()
		self.loss = 0
		self.encoder_optimizer = optim.Adam(self.encoder.parameters()) 												   
		self.decoder_optimizer = optim.Adam(self.decoder.parameters())

	def train(self, seqIn, seqOutOneHot, seqOutEmbedding): 
		''' Train one iteration, no batch '''
		hidden = self.encoder(seqIn) # Encode sentence
		for i in range(len(seqOutOneHot) - 1):
			softmax, hidden = self.decoder(seqOutEmbedding[i], hidden)
			self.loss += self.lossFn(softmax.view(1,-1), torch.tensor([seqOutOneHot[i+1]]).to(device))

	def backprop(self):
		before = time.time()
		print("doing backward")
		self.loss = self.loss * 1000
		self.loss.backward(retain_variables=True) # Compute grads with respect to the network
		print("Encoder sum", list(self.encoder.parameters())[-1].grad)
		print("Decoder sum", list(self.decoder.parameters())[-1].grad)
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

	def predict(self, seqIn):
		with torch.no_grad():
			hidden = self.encoder(seqIn) # Forward propogation to hidden layer
			sentence = []
			glove = torch.zeros(100).to(device) # start token, all zeros
			word = "START"
			max_len, num_words = 40, 0 # failsafe
			while word != "END" and num_words < max_len:
				softmax, hidden = self.decoder(glove, hidden)
				word = utils.num2word(torch.argmax(softmax).item())
				sentence.append(word)
				glove = utils.word2glove(word)
				num_words += 1
			return sentence

	def save(self): # Saving the trained network to a .ckpt file
		torch.save(self.encoder.state_dict(), "RNNencoder.ckpt")
		torch.save(self.decoder.state_dict(), "RNNdecoder.ckpt")

	def load(self): # Loading a trained network from a .ckpt file
		torch.load("RNNencoder.ckpt", map_location=lambda storage, loc: storage)
		torch.load("RNNdecoder.ckpt", map_location=lambda storage, loc: storage)