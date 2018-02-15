from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Categorical
import torchvision.models as models
from pretrain import VGG

EPSILON = 1e-9

class MultinomCell(nn.Module):
	def __init__(self, input_dim, batch_dim, hidden_size, num_factors, gru_cell=True):
		self.hidden = Variable(torch.rand(batch_dim, num_factors))
		self.enc_in = nn.Sequential(
						OrderedDict([
							('multinom_fc1', nn.Linear(feature_dim, hidden_size)),
							('multinom_relu', nn.ReLU()),
						]))
		self.rnn = nn.GRUCell(hidden_size, num_factors) if gru_cell else nn.LSTMCell(hidden_size, num_factors)
		self.enc_out = nn.Sequential(
						OrderedDict([
							('multinom_fc2', nn.Linear(num_factors, num_factors)),
							('softmax', nn.Softmax())
						]))

	def forward(x):
		enc_in = self.enc_in(x)
		self.hidden = self.rnn(enc_in, self.hidden)
		enc_out = self.enc_out(self.hidden)
		return enc_out


class Encoder(nn.Module):
	def __init__(
			self, input_dim, batch_dim, hidden_size, num_factors, 
			latent_dim, gru_cell=True):
		super(Encoder, self).__init__()
		self.num_factors = num_factors
		self.image_enc = 
		self.latent_factors = Variable(torch.rand(num_factors, latent_dim)) #TODO: change from random uniform (0, 1) to something else ! 
		self.latent_encoder = nn.Sequential(OrderedDict([
									('enc_fc', nn.Linear(latent_dim, hidden_size)), 
									('enc_relu', nn.ReLU()),
							  ]))
		self.latent_mean = nn.Linear(hidden_size, hidden_size)
		self.latent_std = nn.Sequential(OrderedDict([
									('std_fc', nn.Linear(hidden_size, hidden_size)), 
									('std_relu', nn.ReLU()),
						  ]))

		self.multinomial_code = MultinomCell(input_dim, batch_dim, hidden_size, num_factors, gru_cell=gru_cell)

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
	def __init__(self, feature_dim, batch_dim, hidden_size, num_factors, latent_dim):
		super(DeepDP, self).__init__()
		self.VGG = self.getVGG()
		self.batch_dim = batch_dim
		self.encoder = Encoder(feature_dim, batch_dim, hidden_size, num_factors, latent_dim)
		self.decoder = Decoder(hidden_size, feature_dim)
		self.prev_x = Variable(torch.zeros(batch_dim, feature_dim), requires_grad=False) 

	def getVGG(self, finetune=False):
		# This is how to chop of layers ! 
		my_vgg = VGG()
		pretrained = models.vgg16(pretrained=True)
		my_vgg_state = my_vgg.state_dict()
		pretrained_state = pretrained.state_dict()
		# filter out the unnecessary layers 
		pretrained_state = {k:v for k, v in pretrained_state if k in my_vgg_state}
		my_vgg_state.update(pretrained_state)
		my_vgg.load_state_dict(pretrained_state)
		if not finetune:
			for param in my_vgg.parameters():
				param.requires_grad = False 
		return my_vgg

	def rewind(self):
		self.prev_x = Variable(torch.zeros(batch_dim, feature_dim), requires_grad=False)  

	def forward(self, x):
		features = self.VGG(x)
		means, stds, multinom = self.encoder(self.prev_x)
		stds += EPSILON

		#compute kl term here: #This form is special to the unit normal prior
		kl_normal = 1 + torch.log(stds*stds) - means*means - stds*stds 
		kl_normal = 0.5*torch.sum(kl_normal)

		#compute kl term for multinomial. Assume uniform prior
		kl_multinom = -(multinom*torch.log(multinom.size()[1]*multinom))
		kl_multinom = kl_multinom.sum()

		kl_loss = -(kl_multinom + kl_normal)

		epsilons = torch.normal(mean=0.0, std=torch.ones(stds.size()))
		epsilons = Variable(epsilons, requires_grad=False)

		#now need to reconstruct
		final_latent = means + stds*epsilons
		final_latent = multinom.matmul(final_latent)
		xhat = self.decoder(final_latent)
		# computing log P(x | z) - same as square loss 
		delta_x = (features - xhat)
		recon_loss = torch.sum(delta_x*delta_x, dim=1)
		recon_loss = torch.sum(recon_loss) / self.batch_dim

		#set to previous
		self.prev_x = features.detach()

		return kl_loss , recon_loss, multinom[0]
 


		





