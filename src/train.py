import model
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

teacher_forcing_ratio = 0.5
SOS_token = 0
EOS_token = 1

class RNN(object):
	def __init__(self, input_size, output_size):
		super(RNN, self).__init__()

		self.encoder = RNNEncoder(input_size)
		self.decoder = RNNDecoder(output_size)


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, lossFunction):
	encoder_hidden = encoder.initHidden()

	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()

	input_length = input_tensor.size(0)
	target_length = target_tensor.size(0)

	encoder_outputs = torch.zeros(encoder.hidden_size)

	loss = 0

	for i in range(input_length):
		encoder_output, encoder_hidden = encoder(input_tensor[i], encoder_hidden)
		encoder_outputs[i] = encoder_output[0, 0]

	decoder_input = torch.tensor([[SOS_token]])
	decoder_hidden = encoder_hidden

	use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

	if use_teacher_forcing:
		# Teacher forcing: feed the target as the next input
		for i in range(target_length):
			decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
			loss += lossFunction(decoder_output, target_tensor[i])
	else:



	loss.backward()

	encoder_optimizer.step()
	decoder_optimizer.step()
def evaluation
