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
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_forcing_ratio = 0.5
seq_loss_penalty = 0.25 # Higher means longer sequences discouraged (i.e. higher -> shorter sequences)
start = torch.zeros(100).to(device)

class EncoderDecoder():
	def __init__(self):
		super().__init__()
		self.encoder = RNNEncoder().to(device)
		self.decoder = RNNDecoder().to(device)
		self.lossFn = nn.CrossEntropyLoss()
		self.encoder_optimizer = optim.Adam(self.encoder.parameters()) 												   
		self.decoder_optimizer = optim.Adam(self.decoder.parameters())

	def train(self, seqIn, seqOutOneHot, seqOutEmbedding): 
		''' Train one iteration, no batch '''
		self.encoder_optimizer.zero_grad() 
		self.decoder_optimizer.zero_grad()
		loss = 0
		hidden = self.encoder(seqIn) # Encode sentence

		if random.random() < teacher_forcing_ratio:
			for i in range(len(seqOutOneHot) - 1):
				softmax, hidden = self.decoder(seqOutEmbedding[i], hidden)
				loss += self.lossFn(softmax, torch.tensor([seqOutOneHot[i+1]]).to(device))
		else:
			glove = start
			for i in range(len(seqOutOneHot) - 1):
				softmax, hidden = self.decoder(glove, hidden)
				word = utils.num2word(torch.argmax(softmax).item())
				glove = utils.word2glove(word)
				loss += self.lossFn(softmax, torch.tensor([seqOutOneHot[i+1]]).to(device))

		before = time.time()
		loss = loss / ((len(seqOutOneHot) - 1) ** seq_loss_penalty) # to not penalize long sequences,  + 7 over two rule
		loss.backward() # Compute grads with respect to the network
		self.encoder_optimizer.step() # Update using the stored grad
		self.decoder_optimizer.step()
		reportedLoss = loss.item()
		after = time.time()
		return reportedLoss, (after - before)

	# TODO(jacob) beam search
	'''def predict(self, seqIn):
		with torch.no_grad():
			hidden = self.encoder(seqIn) # Forward propogation to hidden layer
			sentence = []
			glove = torch.zeros(100).to(device) # start token, all zeros
			word = "START"
			max_len, num_words = 40, 0 # failsafe
			top3seqs = [(0, "")] # (prob, string)
			while word != "END" and num_words < max_len:
				for seq in top3seqs: # TODO(jacob) implement
					if seq[1].endswith("END"): # sequence is done
						continue
					softmax, hidden = self.decoder(glove, hidden)
					word = utils.num2word(torch.argmax(softmax).item())
					top3vals, top3probs = softmax.topk(3)
					for val, prob in zip(top3vals, top3probs):


					sentence.append(word)
					glove = utils.word2glove(word)
					num_words += 1
			return sentence'''

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