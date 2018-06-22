import pandas as pd
import csv
import torch
import spacy
import wordvecs

nlp = spacy.load('en')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
words = pd.read_table("../data/glove_100d.txt", sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
vecs = {}
for index, row in words.iterrows():
	vecs[torch.tensor(row.values, requires_grad=False)] = index