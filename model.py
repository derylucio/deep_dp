from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Categorical

EPSILON = 1e-8

class Encoder(nn.Module):
	def __init__(self, input_dim, hidden_size, num_factors, latent_dim):
		super(Encoder, self).__init__()
		self.num_factors = num_factors
		self.latent_factors = Variable(torch.rand(num_factors, latent_dim)) #TODO: change from random uniform (0, 1) to something else ! 
		self.latent_encoder = nn.Sequential(OrderedDict([
									('enc_fc', nn.Linear(latent_dim, hidden_size)), 
									('enc_relu', nn.ReLU()),
							  ]))
		self.latent_mean = nn.Linear(hidden_size, hidden_size)
		self.latent_std = nn.Linear(hidden_size, hidden_size)

		self.multinomial_code = nn.Sequential(OrderedDict([
									('multinom_fc1', nn.Linear(input_dim, input_dim / 2)),
									('multinom_relu', nn.ReLU()),
									('multinom_fc2', nn.Linear(input_dim / 2, num_factors)),
									('softmax', nn.Softmax()),
								]))

	def forward (self, x):
		multinomial = self.multinomial_code(x)
		latent = self.latent_encoder(self.latent_factors)
		means = self.latent_mean(latent)
		stds = self.latent_std(latent)
		return means, stds, multinomial




class Decoder(nn.Module):
	def __init__(self, hidden_size, output_size):
		super(Decoder, self).__init__()
		self.decoder = nn.Sequential(OrderedDict([
							('decoder_fc1', nn.Linear(hidden_size, output_size)),
							('decoder_elu', nn.ELU()),
							('decoder_fc2', nn.Linear(output_size, output_size))
					   ]))

	def forward (self, x):
		return self.decoder(x)


class DeepDP(nn.Module):
	def __init__(self, input_dim, hidden_size, num_factors, latent_dim):
		super(DeepDP, self).__init__()
		self.encoder = Encoder(input_dim, hidden_size, num_factors, latent_dim)
		self.decoder = Decoder(hidden_size, input_dim)

	def forward(self, x):
		means, stds, multinom = self.encoder(x)
		categorical = Categorical(multinom)
		assignments = categorical.sample() # We can have an implicit number of samples as being x replicated some times.
		epsilons = Variable(torch.normal(mean=0.0, std=torch.ones(assignments.size())), requires_grad=False)

		chosen_means = means[assignments]
		chosen_stds = stds[assignments]

		#compute kl term here: #This form is special to the unit normal prior
		# kl_loss = 1 + torch.log(stds*stds) - means*means - stds*stds 
		# kl_loss = 0.5*torch.sum(kl_loss)
		kl_loss = 1 + torch.log(chosen_stds*chosen_stds) - chosen_means*chosen_means - chosen_stds*chosen_stds 
		kl_loss = 0.5*torch.sum(kl_loss) / np.prod(means.size())

		#now need to reconstruct
		final_latent = chosen_means + chosen_stds*epsilons
		xhat = self.decoder(final_latent)
		# computing log P(x | z) - same as square loss 
		delta_x = (x - xhat)
		recon_loss = torch.sum(delta_x*delta_x, dim=1)
		rewards = recon_loss.detach() # storing this for re-inforce call !
		recon_loss = torch.sum(recon_loss) / np.prod(xhat.size())

		#policy_loss
		rewards = 1.0/rewards
		rewards = (rewards - rewards.mean()) / (EPSILON + rewards.std())
		policy_loss = -categorical.log_prob(assignments)*rewards
		policy_loss = policy_loss.sum()

		return kl_loss , recon_loss , policy_loss, multinom[0]
 


		





