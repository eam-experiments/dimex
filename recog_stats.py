# Copyright [2021] Luis Alberto Pineda Cortés,
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

"""Nnets-stats: Generate graph of neural networks improvements.

Usage:
  recog_stats -h | --help
  recog_stats [--dir=<dir>] [--stages=<stages>] [--learned=<learned>] [--tolerance=<tolerance>] [--lang=<lang>]

Options:
  -h                        Show this screen.
  --dir=<dir>               Directory where data is [default: runs]
  --stages=<stages>         How many stages to consider [default: 6]
  --learned=<learned>       Index of data learned (original, agreed, ...) [default: 4].
  --tolerance=<tolerance>   Differences allowed between memory and cue. [default: 0].
  --lang=<language>         Chooses Language for graphs [default: en].

Arguments:
  language: en - English
  language: es - Spanish
"""

from docopt import docopt
import gettext
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import constants

# Translation
gettext.install('eam', localedir=None, codeset=None, names=None)

stages = 6
tolerance = 0
learned = 4
extended = True
runpath = 'runs'

print(f'Getting data from {constants.run_path}')

def plot_recognition_graph(means, errs, es):
    plt.clf()
    fig = plt.figure()
    x = range(stages)
    plt.ylim(0, 1.0)
    plt.xlim(0, stages)
    plt.autoscale(True)
    plt.errorbar(x, means[:,0], fmt='r-o', yerr=errs[:,0], label='Correct to network')
    plt.errorbar(x, means[:,1], fmt='b-s', yerr=errs[:,1], label='Correct to memory')
    plt.errorbar(x, means[:,2], fmt='g-D', yerr=errs[:,2], label='Correct to network (simplified)')
    plt.errorbar(x, means[:,3], fmt='m-*', yerr=errs[:,3], label='Correct to memory (simplified)')

    plt.ylabel('Normalized distance')
    plt.xlabel('Stages')
    plt.legend()

    prefix = constants.recognition_prefix
    filename = constants.picture_filename(prefix, es)
    fig.savefig(filename, dpi=600)


def get_fold_stats(es, fold):
    prefix = constants.recognition_prefix
    filename = constants.recog_filename(prefix, es, fold)
    print(f'Reading file: {filename}')
    df = pd.read_csv(filename)
    df = df[['CorrSize', 'Cor2Net', 'Cor2Mem','Cor2NetND', 'Cor2MemND']]
    data = df.to_numpy(dtype=float)
    data[:, 1] = data[:,1] / (data[:,1] + data[:,0])
    data[:, 2] = data[:,2] / (data[:,2] + data[:,0])
    data[:, 3] = data[:,3] / (data[:,3] + data[:,0])
    data[:, 4] = data[:,4] / (data[:,4] + data[:,0])
    return data[:,1:]

def main(es):
    stats = []
    for stage in range(stages):
        es.stage = stage
        stage_stats = []
        for fold in range(constants.n_folds):
            fold_stats = get_fold_stats(es, fold)
            stage_stats.append(fold_stats)
        stage_stats = np.array(stage_stats, dtype=float)
        stats.append(stage_stats)

    stats = np.array(stats, dtype=float)    
    print(stats.shape)
    # Reduce folds to their means.
    means = np.mean(stats, axis=1)
    # Means and standard deviations of measures per stage.
    stdvs = np.std(means, axis=1)
    means = np.mean(means, axis=1)
    plot_recognition_graph(means, stdvs, es)

if __name__== "__main__" :
    args = docopt(__doc__)

    # Processing language
    languages = ['en', 'es']
    valid = False
    for lang in languages:
        if lang == args['--lang']:
            valid = True
            break
    if valid:
        lang = args['--lang']
        if lang != 'en':
            lang = gettext.translation('eam', localedir='locale', languages=[lang])
            lang.install()
    else:
        print(__doc__)
        exit(1)

    # Processing base dir.
    runpath = args['--dir']

    # Processing learned data.
    if args['--learned']:
        try:
            learned = int(args['--learned'])
            if (learned < 0) or (learned >= constants.learned_data_groups):
                raise Exception('Number out of range.')
        except:
            constants.print_error(
                f'<learned> must be an integer between 0 and {constants.learned_data_groups}.')
            exit(1)
    
    # Processing tolerance.
    if args['--tolerance']:
        try:
            tolerance = int(args['--tolerance'])
            if (tolerance < 0) or (tolerance >= constants.domain):
                raise Exception('Number out of range.')
        except:
            constants.print_error(
                f'<tolerance> must be an integer between 0 and {constants.domain}.')
            exit(1)

    es = constants.ExperimentSettings(
        learned=learned, tolerance = tolerance, extended=extended)
    constants.run_path = runpath

    main(es)

