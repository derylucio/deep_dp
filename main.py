from torch.optim import Adam, SGD, RMSprop  
from dataprovider import DataProvider
from model import DeepDP
from collections import Counter
from sklearn.metrics import normalized_mutual_info_score
import numpy as np 
import matplotlib.pyplot as plt 
import os

training = {
	'optimizer': 'adam', 
	'params' : {
		'learning_rate' : 1e-4,
	},
	'epochs' : 5, 
	'train_loc':"",
	'sample_size': 16,
	'log_interval':100,
	'result_dir': 'results/for_presentation',
	'model_cfg' : {
		'input_dim' : 60,
		'hidden_size' : 16,
		'num_factors' : 8, #overestimated value of k
		'latent_dim' : 32, #size of the latent dimension
		'kl_weight' : 0.02,
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

def plot_results(nmis, bincounts, losses, epoch):
	if not os.path.exists(training['result_dir']):
		os.makedirs(training['result_dir'])
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
	labels = ['kl_loss', 'recon_loss', 'policy_loss']
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
	test_iterator = dataprovider.data_iterator(train=False)
	ys, preds = [], []
	sum_kl, sum_recon, sum_policy, count = 0, 0, 0, 0
	for test_ind, test_batch in enumerate(test_iterator):
		test_x, test_y = test_batch
		kl_loss, recon_loss, policy_loss, multinom = model(test_x)
		ys.append(test_y) 
		preds.append(np.argmax(multinom.data.numpy()))
		sum_policy += policy_loss
		sum_kl += kl_loss
		sum_recon += recon_loss
		count += 1

	nmi = normalized_mutual_info_score(ys, preds)
	return nmi, np.bincount(preds), (kl_loss.data[0]/count, recon_loss.data[0]/count, sum_policy.data[0]/count)


def run_epoch(model, optimizer, dataprovider, epoch):
	train_iterator = dataprovider.data_iterator(train=True)
	iter_nmis, iter_hists, iter_losses = [], [], []
	for ind, batch in enumerate(train_iterator):
		x, y = batch
		optimizer.zero_grad()
		kl_loss , recon_loss , policy_loss, assigns = model(x)
		total_loss = kl_loss + recon_loss + policy_loss
		total_loss.backward()
		optimizer.step()
		if (ind % training['log_interval'] == 0):
			print 'evaluating on iter : ', ind
			nmi, hist, losses = evaluate(model, dataprovider)
			iter_nmis.append(nmi)
			iter_hists.append(hist)
			iter_losses.append(losses)
			print 'NMI score is nmi ', nmi
			print 'bincounts - ', hist
			print 'kl_loss = ', kl_loss
			print 'recon_loss = ', recon_loss 
			print 'policy_loss = ', policy_loss
			print '\n\n'

	nmi, hist, losses = evaluate(model, dataprovider)
	iter_nmis.append(nmi)
	iter_hists.append(hist)
	iter_losses.append(losses)
	plot_results(iter_nmis, iter_hists, iter_losses, epoch)


def main():
	model = DeepDP(training['model_cfg']['input_dim'], training['model_cfg']['hidden_size'], \
				   training['model_cfg']['num_factors'], training['model_cfg']['latent_dim'])

	dataprovider = DataProvider(training['sample_size'], training['model_cfg']['input_dim'])
	optimizer = get_optimizer(model)
	for i in range(training['epochs']):
		run_epoch(model, optimizer, dataprovider, i)

if __name__ == "__main__":
	main()

