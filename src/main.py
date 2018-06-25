import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from model import RNNEncoder, RNNDecoder
from train import RNN
from utils import word2vec, vec2word, string2vec, vec2string
import wordvecs


dataSize = 1000

def main():
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
			print("Saved", "\n")

def finalOutput():
	sentence = "This is a sample sentence" # Pass data through later
	data = glove(sentence)
	rnn = RNN(data.input_size, data.output_size)
	vectors = data.string2vec("This is a sample sentence <EOS>")

	finalOutput = rnn.evaluation(vectors)
	print(data.vec2string(finalOutput))


if __name__ == "__main__":
	main()
	finalOutput()