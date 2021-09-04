import argparse
import csv
import matplotlib.pyplot as plt
import numpy as np
import sys

import constants
import dimex

AGREED = 'agreed'
ORIGINAL = 'original'
MEMORY = 'memory'
NETWORK = 'network'
SEED_LABELS = 'runs/seed_balanced_Y.npy'
EXPERIMENT = 4

suffixes = {
    AGREED : constants.agreed_suffix,
    ORIGINAL: constants.original_suffix,
    MEMORY: constants.amsystem_suffix,
    NETWORK: constants.nnetwork_suffix}

def seed_frequencies():
    count = np.zeros(constants.n_labels, dtype=int)
    labels = np.load(SEED_LABELS)
    for label in labels:
        count[label] += 1
    return count

def plot_graph(suffix, stage, bot, top, top_err):
    plt.clf()
    plt.figure(figsize=(6.4,4.8))

    labels = dimex.labels_to_phns
    width = 0.75
    # plt.bar(labels, bot, width, label='Seed')
    # plt.bar(labels, top, width, yerr=top_err, bottom=bot, label='Learned')
    plt.bar(labels, top, width, yerr=top_err, label='Learned')
    plt.ylabel('Data')
    plt.xlabel('Phonemes')
    plt.legend()
    suffix = constants.learning_data_learnt + suffix
    filename = constants.picture_filename(suffix, EXPERIMENT, stage=stage)
    plt.savefig(filename, dpi=600)

def sort (seed, means, stdvs):
    total = seed + means
    total, seed, means, stdvs = (list(t) for t in zip(*sorted(zip(total, seed, means, stdvs), reverse=True))) 
    return seed, means, stdvs

def get_stats(suffix, stage):
    seed = seed_frequencies()
    stats = np.zeros((constants.n_labels, constants.n_folds), dtype=int)
    for fold in range(constants.n_folds):
        filename = constants.learned_labels_filename(suffix, fold, stage)
        labels = np.load(filename)
        count = np.zeros(constants.n_labels, dtype=int)
        for label in labels:
            count[label] += 1
        stats[:,fold] = count
    means = np.mean(stats, axis = 1)
    stdvs = np.std(stats, axis = 1)
    # seed, means, stdvs = sort(seed, means, stdvs)
    seed, means, stdvs = sort(np.zeros(constants.n_labels), means, stdvs)
    plot_graph(suffix, stage, seed, means, stdvs)

if __name__== "__main__" :
    parser = argparse.ArgumentParser(description='Analysis of learned data in experiment 4.')
    parser.add_argument('-c', nargs='?',choices=(AGREED, ORIGINAL, MEMORY, NETWORK),
        dest='category', default=AGREED,
        help=f'Category of learned data to analyse (default: {AGREED})')
    parser.add_argument("stage", type=int,
        help="Stage of the learning process (integer from zero)")
    args = parser.parse_args()

    get_stats(suffixes[args.category], args.stage)
