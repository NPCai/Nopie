import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from train import *
import dataLoader
import utils
import numpy as np

def loadModel(self):
	print("evaluate or train: ")
	if input() == "evaluate":
		try:
			ed = ed.load()
		except IOError:
			print("You have no saved model","\n","Creating a new model...","\n")