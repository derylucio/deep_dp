import glob 
import numpy as np 
from PIL import Image 
import torch 
import os
from sklearn.datasets import make_blobs
from torch.autograd import Variable
from sklearn import datasets

import torchvision.transforms as transforms

NUM_DATA_POINTS = 5000
NUM_CENTERS = 4
TRAIN_FRAC = 0.8

RANDOM_SEED = 123
MAX_PIX_VAL = 255.0
np.random.seed (RANDOM_SEED)

def image_loader(filename):
	"""
	Loads image from filename.

	Args:
		filename: (string) path of image to be loaded

	Returns:
		image: (Tensor) contains data of the image
	"""
	image = Image.open(filename)    # PIL image
	return np.asarray(image).flatten() / MAX_PIX_VAL

def load_set(filenames):
	"""
	Load all images in filenames.

	Args:
		filenames: (list) contains all filenames from which images are to be loaded.

	Returns:
		images: (Tensor) contains all image data
	"""
	images = []
	for filename in filenames:
		images.append(image_loader(filename))

	# images is a list where each image is a Tensor with dim 1 x 3 x 64 x 64
	# we concatenate them into one Tensor of dim len(images) x 3 x 64 x 64
	return images


def load_data(data_dir, split):
	data = {}    
	path = os.path.join(data_dir, "{}".format(split)) #"{}_signs".format(split))
	filenames = os.listdir(path)
	filenames = [os.path.join(path, f) for f in filenames if f.endswith('.jpg')]

	# load the images from the corresponding files
	images = np.array(load_set(filenames))
	perm = np.random.permutation(len(images))
	images = images[perm]

	# labels are present in the filename itself for SIGNS
	labels = np.array([int(filename.split('/')[-1].split("_")[0]) for filename in filenames])
	labels = labels[perm]
	return images, labels

class DataProvider(object):
	def __init__(self, dim, num_data_pts = None):
		super(DataProvider, self).__init__()
		# x, y = make_blobs(n_samples=NUM_DATA_POINTS, n_features=dim, centers=NUM_CENTERS, cluster_std=3)
		self.train_x, self.train_y = load_data("mnist", "train")
		#x[:int(len(x) * TRAIN_FRAC)],  y[:int(len(x) * TRAIN_FRAC)]
		self.test_x, self.test_y = load_data("mnist", "val")
		#x[int(len(x) * TRAIN_FRAC):],  y[int(len(x) * TRAIN_FRAC):]

	def data_iterator(self,  batch_size=64, train=True):
		x, y = (self.train_x, self.train_y) if train else (self.test_x, self.test_y)
		splits = (np.arange(len(x)//batch_size) + 1)*batch_size
		x, y = np.split(x, splits), np.split(y, splits)
		for ind, i in enumerate(x):
			yield (Variable(torch.FloatTensor(i)), y[ind])



