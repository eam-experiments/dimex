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
  -x                        Sets last stage as eXtended.
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
last_extended = False

# Considering whether is correct recognition,
# incorrect recognition, or not recognition at all.
n_recogs = 3

def seed_frequencies():
    count = np.zeros(constants.n_labels, dtype=int)
    labels = np.load('seed_labels')
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
    suffix = constants.learning_data_learned + suffix
    filename = constants.picture_filename(suffix, 'xperiment', stage=stage)
    plt.savefig(filename, dpi=600)

def sort (seed, means, stdvs):
    total = seed + means
    total, seed, means, stdvs = (list(t) for t in zip(*sorted(zip(total, seed, means, stdvs), reverse=True))) 
    return seed, means, stdvs

def learning_stats(es):
    # seed = seed_frequencies()
    n_folds = constants.n_folds
    n_labels = constants.n_labels
    stats = np.zeros((n_folds, n_stages, n_labels, n_recogs), dtype=int)
    for fold in range(constants.n_folds):
        stats[fold] = fold_stats(es, fold)

def fold_stats(es: constants.ExperimentSettings, fold):
    stats = []
    for stage in range(n_stages):
        es.stage = stage
        es.extended = last_extended and (stage == (n_stages-1))
        s = stage_stats(es, fold)
        stats.append(s)
    return np.array(stats)

def stage_stats(es: constants.ExperimentSettings, fold):
    suffixes = constants.learning_suffixes[es.learned]
    for suffix in suffixes:
        filename = constants.learned_data_filename(suffix, es, fold)
        data = np.load(filename)
        filename = constants.learned_labels_filename(suffix, es, fold)
        labels = np.load(filename)
        print(f'Fold: {fold}, Stage: {es.stage}, Suffix: {suffix}, Size: {len(labels)}')
        
if __name__== "__main__" :
    args = docopt(__doc__)
    # Processing language.
    lang = args['--lang']
    if (lang != 'en'):
        print('Entering if')
        translation = gettext.translation('eam', localedir='locale', languages=[lang])
        translation.install()

    path = args['--path']
    # Processing stages.
    try:
        n_stages = int(args['--stages'])
        if (n_stages < 0):
            raise Exception('Number must be positive.')
    except:
        constants.print_error(
            '<stages> must be a positive integer.')
        exit(1)
    last_extended = args['-x']

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

    exp_set = constants.ExperimentSettings(0, learned, False, tolerance)
    print(f'Experimental settings: {exp_set}')
    learning_stats(exp_set)
