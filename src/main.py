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

device = utils.getDevice()

ed = EncoderDecoder()
print("Training on dataset...","\n")

STARTembed = torch.zeros(100).to(device)
ENDembed = torch.ones(100).to(device)
EMBED_DIM = 100
UNK = utils.word2num("UNK")
START = utils.word2num("START")
END = utils.word2num("END")

#startTime = time.time()
batchRange = 100000

data = dataLoader.pairs(devSet=True)

for batch in range(batchRange): 
	loss = 0
	minibatch = []
	#for i in np.random.randint(len(data), size=5):
	before = time.time()
	for i in np.random.randint(3, size=3):
		#minibatch.append(data[i])
		x = [j for j in range(i,i+4+i)]
		y = ""
		for i in x:
			y += str(i) + ' '
		minibatch.append({'sentence': "The cat in the hat sucked him off real fat"})
	batchSeqIn = []
	batchSeqOutOneHot = []
	batchSeqOutEmbedding = []
	for pair in minibatch: # TODO(jacob), batches, randomization
		seqIn = utils.string2gloves(pair['sentence'])
		seqOutOneHot = [START]
		seqOutEmbedding = [STARTembed]
		seqOutOneHot.extend(utils.sentence2nums(pair['sentence']))
		seqOutEmbedding.extend(utils.string2gloves(pair['sentence']))
		seqOutOneHot.append(END)
		seqOutEmbedding.append(ENDembed)
		batchSeqIn.append(seqIn)
		batchSeqOutEmbedding.append(seqOutEmbedding)
		batchSeqOutOneHot.append(seqOutOneHot)

	seq_lengths = torch.LongTensor(list(map(len, batchSeqIn))).to(device)
	out_lengths = seq_lengths + 2
	seq_tensor = torch.zeros((len(batchSeqIn), seq_lengths.max(), EMBED_DIM)).float().to(device)
	tgt_tensor = torch.full((len(batchSeqIn), out_lengths.max()), utils.word2num("pad")).to(device)
	embed_tensor = torch.zeros((len(batchSeqIn), out_lengths.max(), EMBED_DIM)).to(device)

	for idx, (seq, seqlen) in enumerate(zip(batchSeqIn, seq_lengths)):
		seq_tensor[idx][:seqlen] = seq
	for idx, (seqlen, one) in enumerate(zip(out_lengths, batchSeqOutOneHot)):
		tgt_tensor[idx][:seqlen] = torch.tensor(one).to(device)
	for idx, (seqlen, embed) in enumerate(zip(out_lengths, batchSeqOutEmbedding)):
		embed_tensor[idx][:seqlen] = torch.stack(embed)

	seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
	seq_tensor = seq_tensor[perm_idx] # Reorders by the seq length
	tgt_tensor = tgt_tensor[perm_idx]
	embed_tensor = embed_tensor[perm_idx]
	seq_tensor = seq_tensor.transpose(0,1) # (B,L,D) -> (L,B,D)
	
	packed = pack_padded_sequence(seq_tensor, seq_lengths.cpu().numpy())
	loss, time_prop = ed.train(packed, tgt_tensor, embed_tensor, out_lengths)
	after = time.time()
	percent = (after - before)
	if batch % 10 == 0:
		print("\n","Squadie tuple: ", pair['sentence'],"")
		print("grad sum on jawn is ", ed.encoder.gru.weight_hh_l0.grad.sum())
		print("jawn 2 is ", ed.encoder.gru.weight_ih_l0.grad.sum())
		print("Tuple prediciton:  ", ed.predict(seqIn.view(len(seqIn), 1, -1)))
		#ed.save(batch)

	print("Total loss at epoch %d: %.2f, and took time %f of total %f" % (batch, loss, percent, time_prop))

print("Saved", "\n")
	