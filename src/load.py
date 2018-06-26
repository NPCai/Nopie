import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from train import *
import dataLoader
import utils
import numpy as np

def loadModel():
	ed = EncoderDecoder()
	try:
		ed = ed.load()
	except IOError:
		print("You have no saved model","\n")
		sys.exit("Run main.py if you would like to train a dataset")
	# evaluate(ed)