import pandas as pd
import csv
import torch
import spacy
import torchtext.vocab as vocab

torch.set_default_tensor_type(torch.FloatTensor)
nlp = spacy.load('en')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
glove = vocab.GloVe(name='6B', dim=100)
numToWord = glove.itos
wordToNum = glove.stoi

numToWord.append("UNK")
numToWord.append("START")
numToWord.append("END")
wordToNum["UNK"] = 400000
wordToNum["START"] = 400001
wordToNum["END"] = 400002


def word2glove(word):
	''' Converts a string to a vector using GloVe, used for encoding input '''
	if word == "END":
		return torch.ones(100)
	if word == "START":
		return torch.zeros(100)
	v = None
	try:
		v = glove.vectors[wordToNum[word.lower()]].to(device).float() # Don't update embeddings
	except KeyError:
		v = word2glove("unk")
	return v.to(device)

def string2gloves(sentence):
	''' Takes in a sentence string and produces a variable-length vectorization '''
	doc = nlp(sentence) # segment the sentence
	vecs = []
	for token in doc:
		vecs.append(word2glove(token.lower_))
	return torch.tensor(torch.stack(vecs), requires_grad=False).to(device).float() # Have to stack so the tensors are not on the inside
	
def num2word(num): 
	''' Used for beam search '''
	if num < 0 or num >= len(numToWord):
		return None
	return numToWord[num]

def word2num(word):
	''' Usedw for encoding output '''
	v = None
	try:
		v = wordToNum[word]
	except KeyError:
		v = word2num("unk") # return the unk token if not in dataset
	return v

def sentence2nums(sentence):
	doc = nlp(sentence) # segment the sentence
	nums = []
	for token in doc:
		nums.append(word2num(token.lower_))
	return nums

def getVocabSize():
	return len(numToWord) # for START, END, UNK

def onehot(index):
	if index == -1:
		# This is the end of sentence token
		return torch.ones(getVocabSize()).to(device).float()
	x = torch.zeros(getVocabSize()).to(device).float()
	if index != -2:
		x[index] = 1.0
	return x