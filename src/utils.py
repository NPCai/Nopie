import pandas as pd
import csv
import torch
import spacy

torch.set_default_tensor_type(torch.FloatTensor)
nlp = spacy.load('en')
device = torch.device("cuda" if False else "cpu")
words = pd.read_table("../data/glove_100d.txt", sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
numToWord = {}
wordToNum = {}

for num, word in enumerate(words.index.values):
	numToWord[num] = word
	wordToNum[word] = num
	
numToWord[num+1] = "UNK"
numToWord[num+2] = "START"
numToWord[num+3] = "END"
wordToNum["UNK"] = num+1
wordToNum["START"] = num+2
wordToNum["END"] = num+3

def word2glove(word):
	''' Converts a string to a vector using GloVe, used for encoding input '''
	v = None
	try:
		v = torch.tensor(words.loc[word].values, requires_grad=False).to(device).float() # Don't update embeddings
	except KeyError:
		v = torch.zeros(100).float()
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
	return len(words) + 3 # for START, END, UNK

def onehot(index):
	if index == -1:
		# This is the end of sentence token
		return torch.ones(getVocabSize()).to(device).float()
	x = torch.zeros(getVocabSize()).to(device).float()
	if index != -2:
		x[index] = 1.0
	return x