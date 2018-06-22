import pandas as pd
import csv
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
words = pd.read_table("../data/glove_100d.txt", sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)

def glove(word):
	v = None
	try:
		v = torch.tensor(words.loc[word].values, requires_grad=False, device=device) # Don't update embeddings
	except KeyError:
		v = torch.zeros(50)
	return v

def sentence2vec(sentence):
	pass
	''' Takes in a sentence string and produces a variable-length vectorization augmented with spacy dep and pos'''

def tuple2vec(tups):
	''' Takes in a tuple and produces a variable-length vectorization '''
	pass

if __name__ == "__main__": # For testing purposes
	print("ready")
	while True:
		print(glove(input())) 