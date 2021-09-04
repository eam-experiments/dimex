import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

def plot_learning_graph(suffix, stage, bot, top, top_err):
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

def plot_recognition_graph(stage, tolerance, means, errs):
    plt.clf()
    fig = plt.figure()
    x = range(constants.n_folds)
    plt.ylim(0.4, 1.0)
    plt.errorbar(x, means[:,0], fmt='r-o', yerr=errs[:,0], label='Correct to network produced')
    plt.errorbar(x, means[:,1], fmt='g-d', yerr=errs[:,1], label='Correct to memory produced')
    plt.errorbar(x, means[:,2], fmt='b-s', yerr=errs[:,2], label='Network produced to memory produced')

    plt.ylabel('Normalized distance')
    plt.xlabel('Folds')
    plt.legend()
    suffix = '-' + constants.recognition_prefix
    filename = constants.picture_filename(suffix, EXPERIMENT, tolerance, stage)
    fig.savefig(filename, dpi=600)

def sort (seed, means, stdvs):
    total = seed + means
    total, seed, means, stdvs = (list(t) for t in zip(*sorted(zip(total, seed, means, stdvs), reverse=True))) 
    return seed, means, stdvs

def learning_stats(suffix, stage):
    # seed = seed_frequencies()
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
    plot_learning_graph(suffix, stage, seed, means, stdvs)

def recognition_stats(tolerance: int, stage: int):
    means = np.zeros((constants.n_folds, 3))
    stdvs = np.zeros((constants.n_folds, 3))
    for fold in range(constants.n_folds):
        filename = constants.recog_filename(constants.recognition_prefix, EXPERIMENT,
            fold, tolerance, stage)
        df = pd.read_csv(filename)
        df['C2N'] = df['Cor2Net'] / df['CorrSize']
        df['C2M'] = df['Cor2Mem'] / df['CorrSize']
        df['N2M'] = 2*df['Net2Mem'] / (df['NetSize'] + df['MemSize'])

        stats = df.describe(include=[np.number])
        means[fold,:] = stats.loc['mean'].values[-3:]
        stdvs[fold,:] = stats.loc['std'].values[-3:]
    print(means[:,1])
    plot_recognition_graph(stage, tolerance, means, stdvs)

if __name__== "__main__" :
    parser = argparse.ArgumentParser(description='Analysis of learned data in experiment 4.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-l', nargs='?', choices=(AGREED, ORIGINAL, MEMORY, NETWORK),
        dest='category', default=AGREED,
        help=f'Category of learned data to analyse (default: {AGREED})')
    group.add_argument('-r', nargs='?', dest='tolerance', type=int, 
        help='Analyse recognition instead of learning')
    parser.add_argument("stage", type=int,
        help="Stage of the learning process (integer from zero)")
    args = parser.parse_args()

    if args.tolerance is None:
        learning_stats(suffixes[args.category], args.stage)
    else:
        recognition_stats(args.tolerance, args.stage)
