import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


COLORS = ['#496d64', '#6d4952', '#a8baca', '#cab8a8', '#babfb7', '#ebe5ef', '#3d0062', '#256200', \
          '#566200', '#0c0062', '#1e02f7', '#dbf702', '#a0b6ad', '#96a9bf', '#aab0bf', '#a2c5b6', '#aba2a2', '#d6cbcb', \
          '#e6e6fa', '#2c2349', '#322044', '#222d4a', '#676767', '#4c0112', '#4c0000', '#800000', '#aaadbd', '#9fadc0',
          '#778ba5', \
          '#bdd2f8', '#bad61f', '#4522f3', '#f87d99', '#647285', '#070707', '#707070', '#777777', '#676767', '#196419',
          '#686472', '#191919', '#131313']

NAMES = ["Ground Truth", "Proposed"]


def plot(arr, save_dir, xlabel, ylabel, name, epoch):
    plt.plot(arr)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(save_dir + "/" + name + "_" + str(epoch) + ".jpg")
    plt.close()


def assignments_to_segments(assignments):
    segments, labels = [], []
    prev, prev_ind = assignments[0], 0
    for ind, i in enumerate(assignments[1:]):
        if i != prev:
            segments.append((prev_ind, ind + 1))
            labels.append(int(prev))
            prev = i
            prev_ind = ind + 1
    segments.append((prev_ind, ind + 2))  # Plus two because the last index is not used!
    labels.append(int(prev))
    return segments, labels


def plot_results_vid(save_dir, ys, preds, vid_kl, vid_recon, vid, epoch):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plot(vid_kl, save_dir, "frame", "kl-loss", vid + "_kl", epoch)
    plot(vid_recon, save_dir, "frame", "recon-loss", vid + "_recon", epoch)
    true_segs, true_seg_labels = assignments_to_segments(ys)
    prop_segs, prop_seg_labels = assignments_to_segments(preds)
    visualize(save_dir, epoch, vid, true_seg_labels, true_segs, prop_seg_labels, prop_segs)


def visualize(save_dir, epoch, vid, true_seg_labels, true_segs, prop_seg_labels, prop_segs):
    '''Takes in the true and proposed segmentation and visualizes them
    param array true_seg_labels : An array containing the true labels of each segement
    param array true_segs       : An array of (starttime, endtime) tuples of each true segment
    param array prop_seg_labels : An array containing the proposed labels of each segement
    param array prop_segs       : An array of (starttime, endtime) tuples of each proposed segment
    '''
    labels = [true_seg_labels, prop_seg_labels]
    for ind, segmentation in enumerate([true_segs, prop_segs]):
        for seg_ind, segment in enumerate(segmentation):
            start, end = segment
            plt.barh(ind, end - start, height=1, align='center', left=start,
                     color=COLORS[labels[ind][seg_ind] % len(COLORS)])
    plt.yticks(range(len(NAMES)), NAMES)
    plt.xlabel('Frame')
    plt.title("Truth Vs Proposed Segmentation")
    plt.savefig(save_dir + "/" + vid + "_segresults_" + str(epoch) + ".jpg")
    plt.close()
