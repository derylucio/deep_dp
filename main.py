from torch.optim import Adam, SGD, RMSprop  
from dataprovider import VideoDataProvider
from model import DeepDP
from plotter import plot, plot_results_vid
from collections import Counter
from sklearn.metrics import normalized_mutual_info_score
import numpy as np 
import os

training = {
	'optimizer': 'adam', 
	'params' : {
		'learning_rate' : 1e-4,
	},
	'epochs' : 5, 
	'sample_size': 32,
	'result_dir': 'results/for_presentation1',
	'vid_data_dir': '', 
	'vid_transcript_dir':, "",
	'model_cfg' : {
		'input_dim' : 4096,
		'hidden_size' : 512,
		'num_factors' : 20, #overestimated value of k
		'latent_dim' : 300, #size of the latent dimension
		'kl_weight' : 1.0,
	},
}
	

def get_optimizer(model):
	lr = training['params']['learning_rate']
	optim_choice = training['optimizer']
	if optim_choice  == 'adam':
		return Adam(model.parameters(), lr=lr)
	elif optim_choice == 'rmsprop':
		return RMSprop(model.parameters(), lr=lr)
	else: 
		return SGD(model.parameters(), lr=lr)




def run_epoch(model, optimizer, dataprovider, epoch):
	train_iterator = dataprovider.data_iterator(train=True)
	vid_kl, vid_recon = [], []
	ys, preds = [], []
	running_nmi = []
	for ind, batch in enumerate(train_iterator):
		x, y, vid = batch
		if x == None:
			nmi = normalized_mutual_info_score(ys, preds)
			mean_kl, mean_recon = sum(vid_kl)/len(vid_kl), sum(vid_recon)/len(vid_recon)
			mean_loss = mean_kl + mean_recon

			plot_results_vid(training['result_dir'], ys, preds, vid_kl, vid_recon, vid, epoch)
			running_nmi.append(nmi)

			print 'NMI score is nmi ', nmi
			print 'kl_loss = ', mean_kl
			print 'recon_loss = ', mean_recon
			print 'total_loss = ', mean_loss
			print '\n\n'

			# Reset for next video
			ys, preds = [], []
			vid_kl, vid_recon = [], []
			model.rewind()

		optimizer.zero_grad()
		kl_loss , recon_loss, assigns = model(x)
		total_loss = training['model_cfg']['kl_weight']*kl_loss + recon_loss
		total_loss.backward()
		optimizer.step()

		ys.append(y) 
		preds.append(np.argmax(assigns.data.numpy()))

		vid_kl.apend(kl_loss.data[0])
		vid_recon.append(recon_loss.data[0])

	plot(training['result_dir'], running_nmi, "video", "nmi", "nmi_scores", epoch)


def main():
	model = DeepDP(training['model_cfg']['input_dim'], training['sample_size'], training['model_cfg']['hidden_size'], \
				   training['model_cfg']['num_factors'], training['model_cfg']['latent_dim'])

	if torch.cuda.is_available():
		model.cuda()

	dataprovider = VideoDataProvider(ttraining['vid_data_dir'], training['vid_transcript_dir'], raining['sample_size'])
	optimizer = get_optimizer(model)
	for i in range(training['epochs']):
		run_epoch(model, optimizer, dataprovider, i)

if __name__ == "__main__":
	main()

