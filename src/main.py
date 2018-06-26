import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from train import EncoderDecoder
import dataLoader
import utils
import numpy as np

# TODO(jacob): resolve tabs becoming unk tokens
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.FloatTensor)

ed = EncoderDecoder()
STARTembed = torch.zeros(100).to(device)
ENDembed = torch.ones(100).to(device)
START = -2
END = -1

data = dataLoader.pairs(devSet=True)
for batch in range(100): 
	loss = 0
	minibatch = []
	for i in np.random.randint(len(data), size=20):
		minibatch.append(data[i])
	for pair in minibatch: # TODO(jacob), batches, randomization
		print(pair)
		seqIn = utils.string2gloves(pair['sentence'])
		seqOutOneHot = [START]
		seqOutEmbedding = [STARTembed]
		for tup in pair['tuples']:
			seqOutOneHot.extend(utils.sentence2nums(tup))
			seqOutEmbedding.extend(utils.string2gloves(tup))
		seqOutOneHot.append(END)
		seqOutEmbedding.append(ENDembed)
		ed.train(seqIn, seqOutOneHot, seqOutEmbedding)
	loss = ed.backprop()

	print("Total loss at epoch %d: %.2f" % (batch, loss))

print("Saved", "\n")
ed.save()

sentence = seq2vec(input())
finalOutput = rnn.evaluation(vectors)
print(data.vec2string(finalOutput))