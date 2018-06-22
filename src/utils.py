import pandas as pd
import csv
import torch
import spacy

nlp = spacy.load('en')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
words = pd.read_table("../data/glove_100d.txt", sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)

def glove(word):
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
		vecs.append(glove(token.lower_))
	return torch.tensor(torch.stack(vecs)) # Have to stack so the tensors are not on the inside

def vec2string():
	pass
	# TODO(jacob)

if __name__ == "__main__": # For testing purposes
	print("ready")
	while True:
		print(string2vec(input())) 