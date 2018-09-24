from torch.optim import Adam, SGD, RMSprop  
from dataprovider import DataProvider
from model import DeepDP
from collections import Counter
from sklearn.metrics import normalized_mutual_info_score
import numpy as np 
import matplotlib.pyplot as plt 
import cPickle as pickle
import os
import pdb
 
RUN = 1
DATASET = 'mnist'
training = {
	'optimizer': 'adam', 
	'params' : {
		'learning_rate' : 1e-4,
	},
	'epochs' : 30, 
	'train_loc':"",
	'log_interval':10,
	'result_dir': 'results/reinforce_formulation_dataset_{}_run_{}'.format(DATASET, RUN),
	'model_cfg' : {
		'input_dim' : 784,
		'hidden_size' : 512,
		'num_factors' : 15, #overestimated value of k
		'latent_dim' : 256, #size of the latent dimension
		'recon_weight' : 1,
		'encoder_loss_weight' : 1.0,
		'kl_weight' : 1e-3,
		'latent_std' :  1.0,
		'batch_size':64,
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

def plot_results(nmis, bincounts, losses, imgs, epoch):
	path = os.path.join (training['result_dir'], 'imgs_epoch-{}'.format(str(epoch)))
	if not os.path.exists(path):
		os.makedirs(path)
	for ind, pair in enumerate(imgs):
		x, x_hat = pair[0], pair[1]
		fig, ax =  plt.subplots(1, 2)
		ax[0].imshow (np.reshape (x, (28, 28)))
		ax[1].imshow (np.reshape (x_hat, (28, 28)))
		ax[0].set_title ('Original')
		ax[1].set_title ('Reconstructed')
		plt.savefig ('{}/{}.jpg'.format(path, ind))
		plt.close()


	plt.plot(nmis)
	plt.xlabel("iter")
	plt.ylabel('NMI Score')
	plt.tight_layout()
	plt.savefig(training['result_dir']  + "/nmi_" + str(epoch) + ".jpg")
	# plt.show()
	plt.close()
	plt.bar(range(len(bincounts[-1])), bincounts[-1], color='green')
	plt.xlabel("cluster")
	plt.ylabel('Num Members')
	plt.savefig(training['result_dir']  + "/cluster_dist_" + str(epoch) + ".jpg")
	# plt.show()
	# plt.close()
	# for ind, i in enumerate(zip(*bincounts)):
	# 	plt.plot(i, label=ind)
	# plt.xlabel("iter")
	# plt.ylabel("num members")
	# plt.legend()
	# plt.savefig(training['result_dir']  + "/cluster_evol_" + str(epoch) + ".jpg")
	# plt.show()
	plt.close()
	labels = ['kl_loss', 'recon_loss', 'encoder_loss', 'total_loss']
	_, axs = plt.subplots(len(labels), 1)
	for ind, i in enumerate(zip(*losses)):
		axs[ind].plot(i, label=labels[ind])
		axs[ind].set_xlabel("iter")
		axs[ind].set_ylabel("Loss")
		axs[ind].legend()
	plt.tight_layout()
	plt.savefig(training['result_dir']  + "/losses_" + str(epoch) + ".jpg")
	# plt.show()
	plt.close()

def evaluate(model, dataprovider):
	batch_size = training['model_cfg']['batch_size']
	test_iterator = dataprovider.data_iterator(train=False, batch_size=batch_size)
	ys, preds = [], []
	sum_kl, sum_recon, sum_encoder, count = 0, 0, 0, 0
	for test_ind, test_batch in enumerate(test_iterator):
		test_x, test_y = test_batch
		kl_loss, recon_loss, multinom, _, encoder_loss = model(test_x)
		print 'before ', encoder_loss
		recon_loss = training['model_cfg']['recon_weight']*recon_loss
		encoder_loss = training['model_cfg']['encoder_loss_weight']*encoder_loss
		print 'after ', encoder_loss
		kl_loss = training['model_cfg']['kl_weight']*kl_loss

		ys.extend(test_y) 
		preds.extend(np.argmax(multinom.data.numpy(), axis=1))
		sum_kl += kl_loss / len(test_y)
		sum_recon += recon_loss / len(test_y)
		sum_encoder += encoder_loss / len(test_y)
		count += 1

	nmi = normalized_mutual_info_score(ys, preds)
	mean_kl, mean_recon, mean_encoder = sum_kl.data[0]/count, sum_recon.data[0]/count, sum_encoder.data[0]/count
	mean_loss = mean_kl + mean_recon + mean_encoder
	return nmi, np.bincount(preds), (mean_kl, mean_recon, mean_encoder, mean_loss)


def run_epoch(model, optimizer, dataprovider, epoch):
	batch_size = training['model_cfg']['batch_size']
	train_iterator = dataprovider.data_iterator(train=True, batch_size=batch_size)
	iter_nmis, iter_hists = [], [] 
	iter_xhat, iter_losses = [], []
	for ind, batch in enumerate(train_iterator):
		x, y = batch
		optimizer.zero_grad()
		kl_loss , recon_loss, assigns, x_hat, encoder_loss = model(x)
		recon_loss = training['model_cfg']['recon_weight']*recon_loss
		encoder_loss = training['model_cfg']['encoder_loss_weight']*encoder_loss
		kl_loss = training['model_cfg']['kl_weight']*kl_loss

		total_loss = (recon_loss + kl_loss + encoder_loss) / len(y)
		total_loss.backward() #need to implement batching
		optimizer.step()
		if (ind % training['log_interval'] == 0):
			print 'evaluating on iter : ', ind
			nmi, hist, losses = evaluate(model, dataprovider)
			iter_nmis.append(nmi)
			iter_hists.append(hist)
			iter_losses.append(losses)
			iter_xhat.append ([x[0].data.numpy(), x_hat[0].data.numpy()])
			print 'NMI score is nmi ', nmi
			print 'bincounts - ', hist
			print 'kl_loss = ', losses[0]
			print 'recon_loss = ', losses[1]
			print  'encoder_loss =', losses[2]
			print 'total_loss = ', losses[3]
			# print 'multinom = ', assigns
			print '\n\n'

	nmi, hist, losses = evaluate(model, dataprovider)
	iter_nmis.append(nmi)
	iter_hists.append(hist)
	iter_losses.append(losses)
	plot_results(iter_nmis, iter_hists, iter_losses, iter_xhat, epoch)


def main():
	model = DeepDP(training['model_cfg']['input_dim'], training['model_cfg']['hidden_size'], \
				   training['model_cfg']['num_factors'], training['model_cfg']['latent_dim'], \
				   training['model_cfg']['latent_std'])

	dataprovider = DataProvider(training['model_cfg']['input_dim'])
	optimizer = get_optimizer(model)
	for i in range(training['epochs']):
		run_epoch(model, optimizer, dataprovider, i)
	meta_file_loc = os.path.join (training['result_dir'], 'metafile.pkl') 
	with open (meta_file_loc, 'w+') as f:
		pickle.dump(training, f)

if __name__ == "__main__":
	main()

