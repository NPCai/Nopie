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

if torch.cuda.is_available():
	torch.set_default_tensor_type(torch.cuda.FloatTensor)
	device = torch.device("cuda")
else:
	torch.set_default_tensor_type(torch.FloatTensor)
	device = torch.device("cpu")
	
teacher_forcing_ratio = 0.5
seq_loss_penalty = 0.4 # Higher means longer sequences discouraged (i.e. higher -> shorter sequences)
start = torch.zeros(100).to(device)

class EncoderDecoder():
	def __init__(self):
		super().__init__()
		self.encoder = RNNEncoder().to(device)
		self.decoder = RNNDecoder().to(device)
		self.lossFn = nn.CrossEntropyLoss()
		self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=1e-2) 												   
		self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=1e-2)

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
		loss = loss / ((len(seqOutOneHot) - 1) ** seq_loss_penalty) # length normalization

		loss.backward() # Compute grads with respect to the network
		self.encoder_optimizer.step() # Update using the stored grad
		self.decoder_optimizer.step()
		reportedLoss = loss.item()
		after = time.time()
		print("Model has been updated by backprop:  ")
		return reportedLoss, (after - before)

	def predict(self, seqIn):
		with torch.no_grad():
			
			hidden = self.encoder(seqIn) # Forward propogation to hidden layer
			top3seqs = [(hidden, start, "", 1.0)] # (hidden, last_token_glove, full_string, prob)

			while True:
				exit = True
				for _, _, string, _ in top3seqs:
					if (not string.endswith("END")) and len(string) < 40: # i.e. still a seq one in the mix
						exit = False
				if exit:
					break

				newTop3seqs = []
				for hidden, glove, string, prob in top3seqs:
					if string.endswith("END"):
						newTop3seqs.append((hidden, glove, string, prob))
						continue
					softmax, hidden = self.decoder(glove, hidden)
					top3probs, top3vals = softmax.topk(3)
					for probNew, valNew in zip(top3probs[0], top3vals[0]):
						wordNew = utils.num2word(valNew.item())
						gloveNew = utils.word2glove(wordNew)
						newTop3seqs.append((hidden, gloveNew, string + " " + wordNew, probNew*prob))
				
				newTop3seqs.sort(key=lambda tup: tup[3], reverse=True)
				top3seqs = newTop3seqs[0:3] # keep the top 3 of 9

			return top3seqs[0][2] # just return the string

	'''def predict(self, seqIn):
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
			return sentence'''

	def save(self): # Saving the trained network to a .ckpt file
		torch.save(self.encoder.state_dict(), "RNNencoder.ckpt")
		torch.save(self.decoder.state_dict(), "RNNdecoder.ckpt")

	def load(self): # Loading a trained network from a .ckpt file
		torch.load("RNNencoder.ckpt", map_location=lambda storage, loc: storage)
		torch.load("RNNdecoder.ckpt", map_location=lambda storage, loc: storage)