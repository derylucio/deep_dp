import glob 
import numpy as np 
from PIL import Image 
import torch 
import os
import glob
from sklearn.datasets import make_blobs
from torch.autograd import Variable
import torch
import torchvision.transforms as transforms
from PIL import Image 

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


class VideoDataProvider(object):
	def __init__(self, vid_dir, transcript_dir, sample_size):
		super(VideoDataProvider, self).__init__()
		self.dir = vid_dir 
		self.transcript_dir = transcript_dir
		self.sample_size = sample_size
		self.videos_list = glob.glob(os.path.join(self.dir , "*"))
		self.test_list = 
		self.loader = transforms.Compose([
						    transforms.Resize(224),  # VGG size
						    transforms.Normalize([103.939, 116.779, 123.68], [1, 1, 1]) # VGG normalization factors
						    transforms.ToTensor()
					  ])


	def data_iterator(self):
		for vid_folder in self.videos_list:
			vid_name = vid_folder.split("/")[-1].split("_capture1")[0]
			trans_file = vid_name + ".txt"
			trans_file = os.path.join(self.transcript_dir, trans_file) 
			files = np.array(glob.glob(os.path.join(vid_folder, "*")))
			frame_nums = [int(fname.split("/")[-1].split(".")[0]) for fname in files]

			trans_data = open(trans_file, "r").readlines()
			frame_offset = int(trans_data[1].rstrip().split(" ")[0]) - 1 # 1-indexed
        	last_frame_index = int(data[-1].rstrip().split(" ")[1]) - frame_offset

			files = files[sorted(frame_nums)][frame_offset:last_frame_index]
			assignments = [-1]*len(files)
			seen_labels = set()
			for line in trans_data[1:]: 
	    		start_frame, end_frame, label = line.rstrip().split(" ")
	    		if label not in seen_labels:
	    			seen_labels.append(label)
            	new_start_frame = int(start_frame) - frame_offset - 1
            	new_end_frame = int(end_frame) - frame_offset
            	assignments[new_start_frame:new_start_frame] = seen_labels.index(label)

			for fname, assign in zip(files, assignments):
				image = Image.open(fname) 
	    		image = self.loader(image)
	    		image = image.unsqueeze(0)
	    		image = image.repeat([self.sample_size, 1, 1, 1])
	    		if torch.cuda.is_available():
	    			image = image.cuda()
	    		yield (Variable(image), assign, vid_name)
	    	yield (None, None, vid_name)
	    	# TODO (ldery): do I need to create a dummy here to mark the end of a video? 
	    	# TODO (ldery): Is the current approach too slow?