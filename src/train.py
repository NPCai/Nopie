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
import customLoss
from torch.distributions import Categorical

if torch.cuda.is_available():
	torch.set_default_tensor_type(torch.cuda.FloatTensor)
	device = torch.device("cuda")
else:
	torch.set_default_tensor_type(torch.FloatTensor)
	device = torch.device("cpu")

teacher_forcing_ratio = 0.6
seq_loss_penalty = 0.4 # Higher means longer sequences discouraged (i.e. higher -> shorter sequences)
start = torch.zeros(100).to(device)

class EncoderDecoder():
	def __init__(self):
		super().__init__()
		self.encoder = RNNEncoder().to(device)
		self.decoder = RNNDecoder().to(device)
		weight = torch.ones(400003)
		weight[utils.word2num("pad")] = 0.0
		lossFn = nn.CrossEntropyLoss(weight=weight)
		self.critic = customLoss.TupleCritic()
		self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=5e-3)
		self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=5e-3)

	def train(self, seqIn, seqOutOneHot, seqOutEmbedding, seq_lengths): 
		''' Train one iteration, no batch '''
		
		self.encoder_optimizer.zero_grad() 
		self.decoder_optimizer.zero_grad()
		loss = 0
		encoder_output, hidden = self.encoder(seqIn) # Encode sentence
		# PROBLEM FOUND: zeroing out with a mask doesnt help ya dufus... the class label is still 0 i.e. "the"
		# PROBLEM 2: mask multiplication not working properly
		#if random.random() < teacher_forcing_ratio:
		glove = torch.zeros(100).to(device)
		for i in range(seqOutOneHot.shape[1] - 1):
			softmax, hidden = self.decoder(seqOutEmbedding[:,i], hidden)
			#print(mask)
			#print("mask is ", mask)
			# mask is 5 x 1
			#softmax = torch.t(mask.unsqueeze(0)) * softmax
			#softmax[:, 0] = (0 == mask).float() # invert the bool mask
			#print("softmax shape", softmax)
			#print("seqOutOneHot is ", seqOutOneHot[:, i+1].long())
			loss += lossFn(softmax, seqOutOneHot[:, i+1].long())
			#print("delta loss is ", x)
		'''else:
			glove = start
			for i in range(len(seqOutOneHot) - 1):
				softmax, hidden = self.decoder(glove, hidden)
				word = utils.num2word(torch.argmax(softmax).item())
				glove = utils.word2glove(word)
				loss += self.lossFn(softmax, torch.tensor([seqOutOneHot[:, i+1]]).to(device))'''
		before = time.time()
		#loss = loss / ((seqOutOneHot.shape[1] - 1) ** seq_loss_penalty) # length normalization
		loss = loss / seq_lengths.float().mean().item()
		loss.backward() # Compute grads with respect to the network
		self.encoder_optimizer.step() # Update using the stored grad
		self.decoder_optimizer.step()
		
		reportedLoss = loss.item()
		after = time.time()
		return reportedLoss, (after - before)

	def rltrain(self, seqIn, seqOutOneHot, seqOutEmbedding, sentence):
		''' Train one iteration, no batch '''
		self.encoder_optimizer.zero_grad() 
		self.decoder_optimizer.zero_grad()
		loss = 0
		hidden = self.encoder(seqIn) # Encode sentence
		glove = start
		tup = []
		log_probs = []
		for i in range(len(seqOutOneHot) - 1):
			softmax, hidden = self.decoder(glove, hidden, 1.0) # lowish softmax temperature
			m = Categorical(softmax)
			wordPos = m.sample()
			glove = utils.word2glove(utils.num2word(wordPos))
			log_probs.append(m.log_prob(wordPos))
			tup.append(utils.num2word(wordPos))

		tup_str = ''.join(i.lower() + ' ' for i in tup)
		total_reward = self.critic.forward(sentence, tup_str)
		seq_prob = torch.stack(log_probs).sum()
		print("SEQUENCE PROBABILITY:  ", seq_prob)
		loss = (seq_prob / seq_prob ) * total_reward * (-10)
		before = time.time()
		loss.backward() # Compute grads with respect to the network
		self.encoder_optimizer.step() # Update using the stored grad
		self.decoder_optimizer.step()
		reportedLoss = loss.item()
		after = time.time()
		print("Model has been updated by backprop:  ")
		return loss.item(), (after - before)

	def predict(self, seqIn):
		''' Beam Search Implementation '''
		with torch.no_grad():
			
			_, hidden = self.encoder(seqIn) # Forward propogation to hidden layer
			top3seqs = [(hidden, torch.zeros(100).to(device), "", 1.0)] # (hidden, last_token_glove, full_string, prob)

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
					softmax, hidden = self.decoder(glove, hidden, batch_size=1)
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

	def save(self, epoch): # Saving the trained network to a .ckpt file
		torch.save(self.encoder.state_dict(), "RNNencoder_epoch" + str(epoch) + ".ckpt")
		torch.save(self.decoder.state_dict(), "RNNdecoder_epoch" + str(epoch) + ".ckpt")

	def load(self): # Loading a trained network from a .ckpt file
		torch.load("RNNencoder.ckpt", map_location=lambda storage, loc: storage)
		torch.load("RNNdecoder.ckpt", map_location=lambda storage, loc: storage)