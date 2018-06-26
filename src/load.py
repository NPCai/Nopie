import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from train import *
import dataLoader
import utils
import numpy as np

tORe = 0

def loadModel():
	ed = EncoderDecoder()
	print("evaluate or train: ")
	if input() == "evaluate":
		try:
			ed = ed.load()
		except IOError:
			print("You have no saved model","\n","Creating a new model...","\n")
			tORe = 1
	return ed, tORe