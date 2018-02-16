from torch.optim import Adam, SGD, RMSprop
from dataprovider import VideoDataProvider
from model import DeepDP
from plotter import plot, plot_results_vid
from sklearn.metrics import normalized_mutual_info_score
import numpy as np
import torch

is_cuda = torch.cuda.is_available()

training = {
    'optimizer': 'adam',
    'params': {
        'learning_rate': 1e-4,
    },
    'epochs': 5,
    'sample_size': 32,
    'log_interval': 100,
    'result_dir': 'results/for_presentation1',
    'vid_data_dir': "/vision/u/ldery/datasets/Jigsaws/Needle_Passing/frames",
    'vid_transcript_dir': "/vision/u/ldery/datasets/Jigsaws/Needle_Passing/transcriptions",
    'model_cfg': {
        'input_dim': 4096,
        'hidden_size': 512,
        'num_factors': 20,  # overestimated value of k
        'latent_dim': 300,  # size of the latent dimension
        'kl_weight': 1.0,
    },
}


def get_optimizer(model):
    lr = training['params']['learning_rate']
    optim_choice = training['optimizer']
    param_list = [param for param in model.parameters() if param.requires_grad]
    if optim_choice == 'adam':
        return Adam(param_list, lr=lr)
    elif optim_choice == 'rmsprop':
        return RMSprop(param_list, lr=lr)
    else:
        return SGD(param_list, lr=lr)


def run_epoch(model, optimizer, dataprovider, epoch):
    train_iterator = dataprovider.data_iterator()
    vid_kl, vid_recon = [], []
    ys, preds = [], []
    running_nmi = []
    for ind, batch in enumerate(train_iterator):
        x, y, vid = batch
        if x is None:
            nmi = normalized_mutual_info_score(ys, preds)
            mean_kl, mean_recon = sum(vid_kl) / len(vid_kl), sum(vid_recon) / len(vid_recon)
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

            continue

        optimizer.zero_grad()
        kl_loss, recon_loss, assigns = model(x)
        total_loss = training['model_cfg']['kl_weight'] * kl_loss + recon_loss
        total_loss.backward(retain_graph=True)
        optimizer.step()

        ys.append(y)
        np_assigns = assigns.data.numpy() if not is_cuda else assigns.data.cpu().numpy()
        preds.append(np.argmax(np_assigns))

        vid_kl.append(kl_loss.data[0])
        vid_recon.append(recon_loss.data[0])
        if (ind % training['log_interval']) == 0:
            print 'kl_loss = ', kl_loss
            print 'recon_loss = ', recon_loss
            print 'total_loss = ', total_loss

    plot(training['result_dir'], running_nmi, "video", "nmi", "nmi_scores", epoch)


def main():
    model = DeepDP(training['model_cfg']['input_dim'], training['sample_size'], training['model_cfg']['hidden_size'], \
                   training['model_cfg']['num_factors'], training['model_cfg']['latent_dim'])
    if torch.cuda.is_available():
        model.cuda()
    dataprovider = VideoDataProvider(training['vid_data_dir'], training['vid_transcript_dir'], training['sample_size'])
    optimizer = get_optimizer(model)
    for i in range(training['epochs']):
        run_epoch(model, optimizer, dataprovider, i)


if __name__ == "__main__":
    main()
