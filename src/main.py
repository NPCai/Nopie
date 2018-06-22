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


dataSize = 1000

def main():
	print("ready")
	while True:
		print(vec(input()))

	data = "This is a sample sentence" # Pass the data through here later
	rnn = RNN(data.input_size, data.output_size)
	catchingLs = []
	for i, batch in data.sentence:
		input, target = batch

		loss = rnn.train(input, target)
		catchingLs.append(loss)

		if i % 100 is 0:
			print("Loss at epoch %d: %.2f" % (i, loss))
			rnn.save()

def finalOutput():
	data = "This is a sample sentence"
	rnn = RNN(data.input_size, data.output_size)
	print(rnn.eval(data))


if __name__ == "__main__":
	main()