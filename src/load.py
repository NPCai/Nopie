import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from train import *
import dataLoader
import utils
import numpy as np

tORe = 0

ed,tORe = loadModel()
if tORe == 0:
	sys.exit("Evaluation not programmed yet")
else:
	print("Training on dataset...","\n")


def loadModel():
	ed = EncoderDecoder()
	try:
		ed = ed.load()
	except IOError:
		print("You have no saved model","\n")
		print("Creating a new model...","\n")
		tORe = 1
	return ed, tORe