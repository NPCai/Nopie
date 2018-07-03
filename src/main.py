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

if torch.cuda.is_available():
	torch.set_default_tensor_type(torch.cuda.FloatTensor)
	device = torch.device("cuda")
else:
	torch.set_default_tensor_type(torch.FloatTensor)
	device = torch.device("cpu")

ed = EncoderDecoder()
print("Training on dataset...","\n")
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
		seqIn = utils.string2gloves(pair['sentence'])
		seqOutOneHot = [START]
		seqOutEmbedding = [STARTembed]
		for tup in pair['tuples']:
			seqOutOneHot.extend(utils.sentence2nums(tup.replace("\t", ",")))
			seqOutEmbedding.extend(utils.string2gloves(tup.replace("\t", ",")))
		seqOutOneHot.append(END)
		seqOutEmbedding.append(ENDembed)

		loss, time = ed.train(seqIn, seqOutOneHot, seqOutEmbedding)
		if batch % 10 == 0:
			print("\n","Squadie tuple: ", tup,"")
			print("Tuple prediciton:  ", ed.predict(seqIn))


	print("Total loss at epoch %d: %.2f, and took time %d" % (batch, loss, time))

print("Saved", "\n")
ed.save()