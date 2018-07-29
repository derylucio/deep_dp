from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Categorical

EPSILON = 1e-9

class Encoder(nn.Module):
	def __init__(self, input_dim, hidden_size, num_factors, latent_dim, latent_std=1, num_layers=1):
		super(Encoder, self).__init__()
		self.num_factors = num_factors
		latents = np.random.normal (scale=latent_std, size=(num_factors, latent_dim))
		self.latent_factors = Variable(torch.FloatTensor (latents)) #TODO: change from random uniform (0, 1) to something else !
		self.latent_encoder = nn.Sequential(
								OrderedDict([('enc_fc', nn.Linear(latent_dim, hidden_size)), 
											('enc_relu', nn.ReLU())]))
		self.latent_mean = nn.Linear(hidden_size, hidden_size)
		self.latent_std = nn.Sequential(OrderedDict([
									('std_fc', nn.Linear(hidden_size, hidden_size)), 
									('std_relu', nn.ReLU()),
						  ]))
		start_h_size = input_dim / 2 if num_layers > 1 else num_factors
		multinom_layers = [('multinom_fc', nn.Linear(input_dim, start_h_size))]
		for i in range(num_layers - 1):
			end_h_size = num_factors if i == num_layers - 2 else start_h_size / 2
			this_layer = [	('multinom_relu_{}'.format(i), nn.ReLU())
							('multinom_fc_{}'.format(i), nn.Linear(start_h_size, end_h_size))]
			start_h_size = end_h_size
			multinom_layers.extend(this_layer)
		multinom_layers.append(('softmax', nn.Softmax()))
		self.multinomial_code = nn.Sequential(OrderedDict(multinom_layers))

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
							('decoder_elu1', nn.ELU()),
							('decoder_fc2', nn.Linear(output_size, output_size)),
							('decoder_elu2', nn.ReLU()),
					   ]))

	def forward (self, x):
		return self.decoder(x)


class DeepDP(nn.Module):
	def __init__(self, input_dim, hidden_size, num_factors, latent_dim, latent_std):
		super(DeepDP, self).__init__()
		# self.factor_kl_method = kl_method
		self.encoder = Encoder(input_dim, hidden_size, num_factors, latent_dim, latent_std=latent_std)
		self.decoder = Decoder(hidden_size, input_dim)

	def forward(self, x):
		means, stds, multinom = self.encoder(x)
		stds += EPSILON

		#compute kl term here: #This form is special to the unit normal prior
		kl_normal = 1 + torch.log(stds*stds) - means*means - stds*stds 
		kl_normal = 0.5*torch.sum(kl_normal)

		#compute kl term for multinomial. Assume uniform prior
		kl_multinom = -(multinom*torch.log(multinom.size()[1]*multinom + EPSILON))
		kl_multinom = kl_multinom.sum()

		kl_loss = -(kl_multinom + kl_normal)

		batch_dim, _ = x.size()
		epsilons = torch.normal(mean=0.0, std=torch.ones(stds.size()))
		epsilons = Variable(epsilons, requires_grad=False)

		#now need to reconstruct
		final_latent = means + stds*epsilons
		final_latent = multinom.matmul(final_latent)
		xhat = self.decoder(final_latent)
		# computing log P(x | z) - same as square loss 
		delta_x = (x - xhat)
		recon_loss = torch.sum(delta_x*delta_x, dim=1)
		recon_loss = torch.sum(recon_loss) / batch_dim

		return kl_loss , recon_loss, multinom[0], xhat

