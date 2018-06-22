import model
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from model import RNNEncoder, RNNDecoder
from train import RNN, train, evaluation, save


dataSize = 1000

def main():
	data = "This is a sample sentence" # Pass the data through here later
	rnn = RNN(data.input_size, data.output_size)
	encoder = RNNEncoder()
	decoder = RNNDecoder()

	catchingLs = []
	for i, batch in sentences:
		input, target = batch

		loss = rnn.train(input, target)
		catchingLs.append(loss)

		if i % 100 is 0:
			print("Loss at epoch %d: %.2f" % (i, loss))
			rnn.save()
	


if __name__ == "__main__":
	main()