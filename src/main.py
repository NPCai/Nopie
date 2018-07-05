import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from train import *
import dataLoader
import utils
import numpy as np
import time
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random
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
	for i in np.random.randint(len(data), size=10):
		minibatch.append(data[i])
	batchSeqIn = []
	batchSeqOutOneHot = []
	batchSeqOutEmbedding = []
	for pair in minibatch: # TODO(jacob), batches, randomization
		seqIn = utils.string2gloves(pair['sentence'])
		seqOutOneHot = [START]
		seqOutEmbedding = [STARTembed]
		seqOutOneHot.extend(utils.sentence2nums(pair['sentence'].replace("\t", ",")))
		seqOutEmbedding.extend(utils.string2gloves(pair['sentence'].replace("\t", ",")))
		seqOutOneHot.append(END)
		seqOutEmbedding.append(ENDembed)
		batchSeqIn.append(seqIn)
		batchSeqOutEmbedding.append(seqOutEmbedding)
		batchSeqOutOneHot.append(seqOutOneHot)

		seq_lengths = torch.LongTensor(list(map(len, seqIn))).to(device)

		seq_tensor = torch.zeros((len(seqIn), seq_lengths.max())).long().to(device)
		for idx, (seq, seqlen) in enumerate(zip(batchSeqIn, seq_lengths)):
			seq_tensor[idx, :seqlen] = torch.FloatTensor(seq)

		seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
		seqIn = seqIn[perm_idx]
		seqIn = seqIn.transpose(0,1) # (B,L,D) -> (L,B,D)
		seqIn = pack_padded_sequence(seqIn, seq_lengths.cpu().numpy())

		loss, time = ed.train(seqIn, torch.stack(seqOutOneHot), torch.stack(seqOutEmbedding))
		if batch % 10 == 0:
			print("\n","Squadie tuple: ", pair['sentence'],"")
			print("Tuple prediciton:  ", ed.predict(seqIn))


	print("Total loss at epoch %d: %.2f, and took time %d" % (batch, loss, time))

print("Saved", "\n")
ed.save()