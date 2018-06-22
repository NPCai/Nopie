import pandas as pd
import csv
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
words = pd.read_table("../data/glove50d.txt", sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)

def vec(word):
	v = None
	try:
		v = torch.tensor(words.loc[word].values, requires_grad=False, device=device) # Don't update embeddings
	except KeyError:
		v = torch.zeros(50)
	return v

def sentence_vec(sentence):
	''' Takes in a spacy sentence object and produces a 

if __name__ == "__main__": # For testing purposes
	print("ready")
	while True:
		print(vec(input()))
