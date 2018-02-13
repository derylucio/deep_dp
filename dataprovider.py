import glob 
import numpy as np 
from PIL import Image 
import torch 
import os
from sklearn.datasets import make_blobs
from torch.autograd import Variable

NUM_DATA_POINTS = 5000
NUM_CENTERS = 4
TRAIN_FRAC = 0.8


class DataProvider(object):
	def __init__(self, sample_size, dim):
		super(DataProvider, self).__init__()
		self.sample_size = sample_size
		x, y = make_blobs(n_samples=NUM_DATA_POINTS, n_features=dim, centers=NUM_CENTERS, cluster_std=3)
		self.train_x, self.train_y = x[:int(len(x) * TRAIN_FRAC)],  y[:int(len(x) * TRAIN_FRAC)]
		self.test_x, self.test_y = x[int(len(x) * TRAIN_FRAC):],  y[int(len(x) * TRAIN_FRAC):]

	def data_iterator(self, train=True):
		x, y = (self.train_x, self.train_y) if train else (self.test_x, self.test_y)
		for ind, i in enumerate(x):
			i = np.expand_dims(i, axis=0)
			z = np.repeat(i, self.sample_size, axis=0)
			yield (Variable(torch.FloatTensor(z)), y[ind])
