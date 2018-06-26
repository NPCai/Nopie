import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from train import *
import dataLoader
import utils
import numpy as np
import time
from timey import *

ed = EncoderDecoder()

print("Training on dataset...","\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.FloatTensor)

STARTembed = torch.zeros(100).to(device)
ENDembed = torch.ones(100).to(device)

UNK = utils.getVocabSize() - 3
START = utils.getVocabSize() - 2
END = utils.getVocabSize() - 1

startTime = time.time()
batchRange = 200

data = dataLoader.pairs(devSet=True)
for batch in range(batchRange): 
	loss = 0
	minibatch = []
	for i in np.random.randint(len(data), size=1):
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

	if batch % 50 == 1:
		print('Total time elapsed: %s (Estimated time remaining: %d %d%%)' % (timeSince(startTime, batch / batchRange),batch, batch / batchRange * 100))


	print("Total loss at epoch %d: %.2f" % (batch, loss))

print("Saved", "\n")
ed.save()

# sentence = seq2vec(input())
# finalOutput = rnn.evaluation(vectors)
# print(data.vec2string(finalOutput))