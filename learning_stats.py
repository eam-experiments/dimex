# Copyright [2020] Luis Alberto Pineda Cortés, Gibrán Fuentes Pineda,
# Rafael Morales Gamboa.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Entropic Associative Memory Experiments

Usage:
  learning_stats -h | --help
  learning_stats [--path=<path>] [--stages=<stages>] [-x] [--learned=<learned_data>] [--tolerance=<tolerance>] [--lang=<language>]

Options:
  -h                        Show this screen.
  --path=<path>             Directory where results are found [default: runs].
  --stages=<stages>         Number of stages to consider [default: 10].
  -x                        Sets all stages as eXtended.
  --learned=<learned_data>  Selects which learned data is used for learning [default: 0].
  --tolerance=<tolerance>   Allow Tolerance (unmatched features) in memory [default: 0].
  --lang=<language>         Chooses language for  graphs [default: en].            
"""
from docopt import docopt
import gettext
import matplotlib.pyplot as plt
import numpy as np
import constants
import dimex

n_stages = 10
path = 'run'
extended = False

# Considering whether is correct recognition,
# incorrect recognition, or not recognition at all.
n_recogs = 3


def plot_learning_graph(suffix, means, stdevs, es):
    plt.clf()
    plt.figure(figsize=(6.4, 4.8))
    labels = dimex.labels_to_phns
    width = 0.75
    fig, ax = plt.subplots()
    cumm = np.zeros(constants.n_labels, dtype=float)
    for i in range(len(means)):
        ax.bar(labels, means[i, :], width, bottom=cumm, label=f'Stage {i}')
        cumm += means[i, :]
        # median = np.full(constants.n_labels, np.max(cumm))
        # ax.plot(labels, median)
    # median = np.full(constants.n_labels, np.median(cumm))
    # ax.plot(labels, median)
    ax.set_ylabel('Data')
    ax.set_xlabel('Phonemes')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=5)
    suffix = constants.learning_data_learned + suffix
    filename = constants.picture_filename(suffix, es)
    plt.savefig(filename, dpi=600)


def sort(seed, means, stdvs):
    total = seed + means
    total, seed, means, stdvs = (list(t) for t in zip(
        *sorted(zip(total, seed, means, stdvs), reverse=True)))
    return seed, means, stdvs


def learning_stats(es):
    # seed = seed_frequencies()
    n_folds = constants.n_folds
    n_labels = constants.n_labels
    stats = np.zeros((n_folds, n_stages, n_labels), dtype=int)
    for fold in range(constants.n_folds):
        stats[fold] = fold_stats(es, fold)
    means = np.mean(stats, axis=0)
    stdvs = np.std(stats, axis=0)
    plot_learning_graph('-labels', means, stdvs, es)


def fold_stats(es: constants.ExperimentSettings, fold):
    stats = []
    es.stage = 0
    lds = dimex.LearnedDataSet(es, fold)
    previous = np.zeros(constants.n_labels, dtype=int)
    for stage in range(n_stages):
        es.stage = stage
        lds = dimex.LearnedDataSet(es, fold)
        s = lds.get_distribution()
        stats.append(s-previous)
        previous = s
    return np.array(stats)


def label_stats(labels):
    stats = np.zeros(constants.n_labels, dtype=int)
    for label in labels:
        stats[label] = stats[label] + 1
    return stats


if __name__ == "__main__":
    args = docopt(__doc__)
    # Processing language.
    lang = args['--lang']
    if (lang != 'en'):
        print('Entering if')
        translation = gettext.translation(
            'eam', localedir='locale', languages=[lang])
        translation.install()

    constants.run_path = args['--path']
    # Processing stages.
    try:
        n_stages = int(args['--stages'])
        if (n_stages < 0):
            raise Exception('Number must be positive.')
    except:
        constants.print_error(
            '<stages> must be a positive integer.')
        exit(1)
    extended = args['-x']

    # Processing learned data.
    try:
        learned = int(args['--learned'])
        if (learned < 0) or (learned >= constants.learned_data_groups):
            raise Exception('Number out of range.')
    except:
        constants.print_error(
            '<learned_data> must be a positive integer.')
        exit(1)

    # Processing tolerance.
    try:
        tolerance = int(args['--tolerance'])
        if (tolerance < 0) or (tolerance > constants.domain):
            raise Exception('Number out of range.')
    except:
        constants.print_error(
            '<tolerance> must be a positive integer.')
        exit(1)

    exp_set = constants.ExperimentSettings(0, learned, extended, tolerance)
    print(f'Experimental settings: {exp_set}')
    learning_stats(exp_set)
