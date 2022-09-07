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
  learning_stats [--lang=<language>] <means_csv> <stdevs_csv> <graph_svg>

Options:
  -h                        Show this screen.
  --lang=<language>         Chooses language for  graphs [default: en].            
"""
from docopt import docopt
import gettext
import matplotlib.pyplot as plt
import numpy as np
import constants
import dimex

def plot_learning_graph(means, stdevs, filename):
    plt.clf()
    plt.figure(figsize=(6.4, 4.8))
    labels = dimex.labels_to_phns
    width = 0.75
    fig, ax = plt.subplots()
    cumm = np.zeros(constants.n_labels, dtype=float)
    for i in range(len(means)):
        ax.bar(labels, means[i, :], width, yerr=stdevs[i, :], capsize=2, bottom=cumm, label=f'Stage {i}')
        cumm += means[i, :]
    ax.set_ylabel('Data')
    ax.set_xlabel('Phonemes')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=5)
    plt.savefig(filename, dpi=600)


def learning_stats(means_filename, stdvs_filename, graph_filename):
    means = np.genfromtxt(means_filename, delimiter=',', dtype=float)
    stdvs = np.genfromtxt(stdvs_filename, delimiter=',', dtype=float)
    constants.print_csv(means)
    constants.print_csv(stdvs)
    plot_learning_graph(means, stdvs, graph_filename)

if __name__ == "__main__":
    args = docopt(__doc__)
    # Processing language.
    lang = args['--lang']
    if (lang != 'en'):
        print('Entering if')
        translation = gettext.translation(
            'eam', localedir='locale', languages=[lang])
        translation.install()

    means_filename=args['<means_csv>']
    stdvs_filename=args['<stdevs_csv>']
    graph_filename=args['<graph_svg>']
    print(means_filename, stdvs_filename)
    learning_stats(means_filename, stdvs_filename, graph_filename)
