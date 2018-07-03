import torch
import spacy
from torch.autograd.function import Function
import re

nlp = spacy.load('en')

class RLLoss(Function):
	''' Advanced RL Loss function that tries to produce the named entities ''' 
	def forward(self):
		pass

class TupleRLLoss(Function):
	''' Loss function that tries to make output have good tuples '''
	def forward(self, model, target):
		reward = 0
		if model[0] == "<":
			reward += 1
		if model[-1] == ">":
			reward += 1
		groups = re.search(r'<([^>]+)>', model)
		commas = 0
		for i in groups.group(0):
			if i == ",":
				commas += 1
		if commas == 2:
			reward += 1
		return reward
