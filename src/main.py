import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from train import EncoderDecoder
import dataLoader
import utils

def main():
	ed = EncoderDecoder()
	data = dataLoader.pairs(devSet=True)

	for epoch in range(5):
		for pair in data: # TODO(jacob), batches
			seqIn = utils.string2vec(pair['sentence'])
			seqOut = [ed.sos]
			for tup in pair['tuples']:
				seqOut.append(utils.string2vec([tup]))
			seqOut.append(ed.eos)
			seqOut = torch.stack(seqOut)



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

ed = EncoderDecoder()

data = dataLoader.pairs(devSet=True)
for epoch in range(5): 
	for pair in data: # TODO(jacob), batches, randomization
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

sentence = seq2vec(input())
finalOutput = rnn.evaluation(vectors)
print(data.vec2string(finalOutput))
