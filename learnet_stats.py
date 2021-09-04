import argparse
import csv
import numpy as np
import sys

import constants

AGREED = 'agreed'
ORIGINAL = 'original'
MEMORY = 'memory'
NETWORK = 'network'

suffixes = {
    AGREED : constants.agreed_suffix,
    ORIGINAL: constants.original_suffix,
    MEMORY: constants.amsystem_suffix,
    NETWORK: constants.nnetwork_suffix}

def get_stats(suffix, stage):
    stats = np.zeros((constants.n_labels, constants.n_folds))
    for fold in range(constants.n_folds):
        filename = constants.learned_labels_filename(suffix, fold, stage)
        labels = np.load(filename)
        for label in labels:
            stats[labels,fold] += 1
    f = csv.writer(sys.stdout)
    for label in range(constants.n_labels):
        f.writerow(stats[label,:])

if __name__== "__main__" :
    parser = argparse.ArgumentParser(description='Analysis of learned data in experiment 4.')
    parser.add_argument('-c', nargs='?',choices=(AGREED, ORIGINAL, MEMORY, NETWORK),
        dest='category', default=AGREED,
        help=f'Category of learned data to analyse (default: {AGREED})')
    parser.add_argument("stage", type=int,
        help="Stage of the learning process (integer from zero)")
    args = parser.parse_args()

    get_stats(suffixes[args.category], args.stage)