import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from model import RNNEncoder, RNNDecoder
from train import RNN
from utils import word2vec, vec2word, string2vec, vec2string
from wordvecs import *
import json
import dataLoader

torch.set_default_tensor_type(torch.FloatTensor)

def main():
	ed = EncoderDecoder()
	data = dataLoader.pairs(devSet=True)

	for epoch in range(5):
		for pair in data: # TODO(jacob), batches
			seqIn = utils.seq2vec(pair['sentence'])
			seqOut = [ed.sos]
			for tup in pair['tuples']:
				seqOut.append(utils.seq2vec[tup])
			seqOut.append(ed.eos)
			seqOut = torch.stack(torch.tensor(seqOut))
			loss += ed.train(seqIn, seqOut)

		print("Total loss at epoch %d: %.2f" % (epoch, loss))
		print("Saved", "\n")
	ed.save()

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