import matplotlib.pyplot as plt
import json
import glob
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--title')
args = parser.parse_args()
# Usage e.g. call:
# python plot_score.py --title "1st Submission"

def plot_kf_scores(title, kf_scores_file):
    
    kf_scores = json.load(open(kf_scores_file, 'r'))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    for k, v in kf_scores.items():
        ax.plot(range(1,len(v)+1), v, label='Fold {}'.format(k))
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')

    ax.plot(range(1,len(v)+1), np.mean([v for v in kf_scores.values()], axis=0),
            label='Average', lw=3)

    fig_title = title+'\nAccuracy= '+kf_scores_file[14:-5]
    ax.set_title(fig_title)
    plt.legend()
    plt.savefig('KFold '+(' ').join(fig_title.split('\n'))+'.png', bbox_inches='tight', dpi=144)

def plot_test_scores(title, test_scores_file):
    
    scores = json.load(open(test_scores_file, 'r'))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.plot(range(1,len(scores)+1), scores)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    
    fig_title = title+'\nFinal epoch loss = '+test_scores_file[12:-5]
    ax.set_title(fig_title)
    plt.savefig('Test set '+(' ').join(fig_title.split('\n'))+'.png', bbox_inches='tight', dpi=144)

title = args.title if args.title else ''
kf_score_file = glob.glob('kf_scores*')
test_score_file = glob.glob('test_scores*')

plot_kf_scores(title, kf_score_file[0])
plot_test_scores(title, test_score_file[0])
