import model
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

teacher_forcing_ratio = 0.5

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
	encoder_hidden = encoder.initHidden()

	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()

	input_length = input_tensor.size(0)
	target_length = target_tensor.size(0)

	encoder_outputs = torch.zeros(encoder.hidden_size, device = device)

	loss = 0

	for i in range(input_length):
		encoder_output, encoder_hidden = encoder(input_tensor[i], encoder_hidden)
		encoder_outputs[i] = encoder_output[0, 0]

	decoder_input = torch.tensor([[SOS_token]], device = device)

	decoder_hidden = encoder_hidden

	use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

	if use_teacher_forcing:
		# Teacher forcing: feed the target as the next input
		for i in range(target_length):
			
			