import model
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

def train(input, target, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
	encoder_hidden = encoder.initHidden()

print("PooPoo")