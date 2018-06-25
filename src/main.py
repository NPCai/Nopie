import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from train import EncoderDecoder
import dataLoader
import utils

torch.set_default_tensor_type(torch.FloatTensor)
ed = EncoderDecoder()
STARTembed = torch.zeros(100)
ENDembed = torch.ones(100)
START = -2
END = -1

data = dataLoader.pairs(devSet=True)
for epoch in range(5): 
	loss = 0
	for pair in data: # TODO(jacob), batches, randomization
		seqIn = utils.string2gloves(pair['sentence'])
		seqOutOneHot = [START]
		seqOutEmbedding = [STARTembed]
		for tup in pair['tuples']:
			seqOutOneHot.extend(utils.sentence2nums(tup))
			seqOutEmbedding.extend(utils.string2gloves(tup))
		seqOutOneHot.append(END)
		seqOutEmbedding.append(ENDembed)
		loss += ed.train(seqIn, seqOutOneHot, seqOutEmbedding)

	print("Total loss at epoch %d: %.2f" % (epoch, loss))
	print("Saved", "\n")

ed.save()

sentence = seq2vec(input())
finalOutput = rnn.evaluation(vectors)
print(data.vec2string(finalOutput))