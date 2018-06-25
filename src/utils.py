import pandas as pd
import csv
import torch
import spacy
import wordvecs

def word2vec(word):
	''' Converts a string to a vector using GloVe '''
	v = None
	try:
		v = torch.tensor(wordvecs.words.loc[word].values, requires_grad=False, device=wordvecs.device) # Don't update embeddings
	except KeyError:
		v = torch.zeros(100)
	return v

def vec2word(vec): # Expects a torch tensor
	w = ""
	try:
		w = wordvecs.vecs[str(vec)]
	except KeyError:
		w = ""
	return w	

def string2vec(sentence):
	''' Takes in a sentence string and produces a variable-length vectorization'''
	# TODO(jacob) augment with spacy pos and dep data
	doc = wordvecs.nlp(sentence) # segment the sentence
	vecs = []
	for token in doc:
		vecs.append(word2vec(token.lower_))
	return torch.tensor(torch.stack(vecs), requires_grad=False) # Have to stack so the tensors are not on the inside

def vec2string(vectors):
	sentence = []
	for vector in vectors:
		print(vector)
		sentence.append(vec2word(vector))
	return sentence	


if __name__ == "__main__": # For testing purposes, convert into seq then back to words
	print("ready")
	while True:
		x =  string2vec(input())
		print(vec2string(x))