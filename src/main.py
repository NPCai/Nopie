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

UNK = utils.word2num("UNK")
START = utils.word2num("START")
END = utils.word2num("END")

#startTime = time.time()
batchRange = 100000

data = dataLoader.pairs(devSet=True)

for batch in range(batchRange): 
	loss = 0
	minibatch = []
	for i in np.random.randint(len(data), size=1):
		minibatch.append(data[i])
	for pair in minibatch: # TODO(jacob), batches, randomization
		print(pair)
		seqIn = utils.string2gloves(pair['sentence'].replace("\t", ","))
		seqOutOneHot = [START]
		seqOutEmbedding = [STARTembed]
		for tup in pair['tuples']:
			seqOutOneHot.extend(utils.sentence2nums(tup))
			seqOutEmbedding.extend(utils.string2gloves(tup))
		seqOutOneHot.append(END)
		seqOutEmbedding.append(ENDembed)

		loss, time = ed.train(seqIn, seqOutOneHot, seqOutEmbedding)
		if batch % 10 == 0:
			print("Tuple prediciton:  ", ed.predict(seqIn))

	print("Total loss at epoch %d: %.2f, and took time %d" % (batch, loss, time))

print("Saved", "\n")
ed.save()

# sentence = seq2vec(input())
# finalOutput = rnn.evaluation(vectors)
# print(data.vec2string(finalOutput))