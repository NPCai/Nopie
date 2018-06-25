import pandas as pd
import csv
import torch
import spacy

torch.set_default_tensor_type(torch.FloatTensor)
nlp = spacy.load('en')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
words = pd.read_table("../data/glove_100d.txt", sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
numToWord = {}
wordToNum = {}

for num, word in enumerate(words.index.values):
	numToWord[num] = word
	wordToNum[word] = num

def word2glove(word):
	''' Converts a string to a vector using GloVe, used for encoding input '''
	v = None
	try:
		v = torch.tensor(words.loc[word].values, requires_grad=False, device=device).float() # Don't update embeddings
	except KeyError:
		v = torch.zeros(100).float()
	return v

def string2gloves(sentence):
	''' Takes in a sentence string and produces a variable-length vectorization '''
	# TODO(jacob) augment with spacy pos and dep data
	doc = nlp(sentence) # segment the sentence
	vecs = []
	for token in doc:
		vecs.append(word2glove(token.lower_))
	return torch.tensor(torch.stack(vecs), requires_grad=False).float() # Have to stack so the tensors are not on the inside

def num2word(num): 
	''' Used for beam search '''
	if num < 0 or num >= len(numToWord):
		return None
	return numToWord[num]

def word2num(word):
	''' Used for encoding output '''
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
	return len(words)

def onehot(index):
	print(index)
	if index == -1:
		# This is the end of sentence token
		return torch.ones(len(words)).float()
	x = torch.zeros(len(words)).float()
	if index != -2:
		x[index] = 1.0
	return x