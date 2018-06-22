import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from model import Encoder, Decoder
from train import RNN
import pandas as pd
import csv

words = pd.read_table("../data/glove50d.txt", sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)

def vec(w):
  return words.loc[w].as_matrix()

def main():
	print("ready")
	while True:
		print(vec(input()))

	data = "This is a sample sentence" # Pass the data through here later
	rnn = RNN(data.input_size, data.output_size)
	encoder = RNNEncoder()
	decoder = RNNDecoder()

if __name__ == "__main__":
	main()