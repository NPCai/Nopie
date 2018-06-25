import pandas as pd
import csv
import torch
import spacy

# torch.set_default_tensor_type(torch.FloatTensor)
nlp = spacy.load('en')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
words = pd.read_table("../data/glove_100d.txt", sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
numToWord = enumerate(words.index.values)

def word2vec(word):
	''' Converts a string to a vector using GloVe '''
	v = None
	try:
		v = torch.tensor(words.loc[word].values, requires_grad=False, device=device) # Don't update embeddings
	except KeyError:
		v = torch.zeros(100)
	return v

def string2vec(sentence):
	''' Takes in a sentence string and produces a variable-length vectorization'''
	# TODO(jacob) augment with spacy pos and dep data
	doc = nlp(sentence) # segment the sentence
	vecs = []
	for token in doc:
		vecs.append(word2vec(token.lower_))
	return torch.tensor(torch.stack(vecs), requires_grad=False).float() # Have to stack so the tensors are not on the inside

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
	return torch.tensor(torch.stack(vecs), requires_grad=False) # Have to stack so the tensors are not on the inside

def num2word(num):
	if num < 0 or num >= len(numToWord):
		return None
	return numToWord[num]