import model
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from model import RNNEncoder, RNNDecoder
from train import RNN, train, evaluation, save

def main():
	data = "This is a sample sentence" # Pass the data through here later
	rnn = RNN(data.input_size, data.output_size)
	encoder = RNNEncoder()
	decoder = RNNDecoder()

	


if __name__ == "__main__":
	main()