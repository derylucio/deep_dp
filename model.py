from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import OneHotCategorical
import pdb

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
									('std_relu', nn.ReLU())
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
		self.mvg_beta = 0.5 
		self.num_factors = num_factors
		self.mvg, self.mvg_t = np.array([0.0]*num_factors), np.array([1]*num_factors)

	def forward(self, x):
		means, stds, multinom = self.encoder(x)
		dist = OneHotCategorical(multinom)
		one_hot = dist.sample()
		stds += EPSILON

		#compute kl term here: #This form is special to the unit normal prior
		all_kl_normal = 1 + torch.log(stds*stds) - means*means - stds*stds 
		chosen_kl_normal = one_hot.matmul(all_kl_normal) 
		kl_normal = 0.5*torch.sum(chosen_kl_normal)

		#compute kl term for multinomial. Assume uniform prior
		kl_multinom = -(multinom*torch.log(multinom.size()[1]*multinom + EPSILON))
		kl_multinom = kl_multinom.sum()

		kl_loss = -(kl_multinom + kl_normal)

		batch_dim, _ = x.size()
		epsilons = torch.normal(mean=0.0, std=torch.ones(stds.size()))
		epsilons = Variable(epsilons, requires_grad=False)

		#now need to reconstruct
		final_latent = means + stds*epsilons
		final_latent = one_hot.matmul(final_latent)
		xhat = self.decoder(final_latent)
		# computing log P(x | z) - same as square loss 
		delta_x = (x - xhat)
		recon_loss = torch.sum(delta_x*delta_x, dim=1)

		max_ind = torch.argmax(one_hot, dim=1).data.numpy()
		detached_recon = Variable (recon_loss.data, requires_grad=False)
		detached_kl = Variable (kl_loss.data, requires_grad=False)
		# advantage = -self.mvg[max_ind] + detached_recon
		# advantage = advantage.type_as(detached_recon)
		advantage = 1.0/((detached_recon + detached_kl)*1e-6)
		# advantage = advantage / detached_recon # (1/a  - 1/b) / (1/b) need to invese to keep positive
		encoder_loss = -dist.log_prob(one_hot) * advantage

		#not completely correct implementation
		det_recon_numpy = (1 - self.mvg_beta)*detached_recon.data.numpy()
		for ind, val in enumerate(max_ind):
			self.mvg[val] = self.mvg_beta * self.mvg[val] + (1 - self.mvg_beta)*det_recon_numpy[ind]
			# self.mvg[ind] /= (1.0 - self.mvg_beta**self.mvg_t[ind])
			# self.mvg_t[ind] += 1 
		# print(self.mvg, recon_loss)
		# print (kl_loss , recon_loss.sum(), encoder_loss.sum())
		# print(self.mvg, detached_recon.data.numpy())
		return kl_loss , recon_loss.sum(), multinom, xhat, encoder_loss.sum()

