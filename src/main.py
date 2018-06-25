import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from train import EncoderDecoder
import dataLoader
import utils

ed = EncoderDecoder()

data = dataLoader.pairs(devSet=True)
for epoch in range(5): 
	for pair in data: # TODO(jacob), batches, randomization
		seqIn = utils.string2gloves(pair['sentence'])
		seqOut = [ed.sos]
		for tup in pair['tuples']:
			seqOut.append(utils.sentence2nums[tup])
		seqOut.append(ed.eos)
		seqOut = torch.stack(torch.tensor(seqOut))
		loss += ed.train(seqIn, seqOut)

	print("Total loss at epoch %d: %.2f" % (epoch, loss))
	print("Saved", "\n")
	
ed.save()

sentence = seq2vec(input())
finalOutput = rnn.evaluation(vectors)
print(data.vec2string(finalOutput))