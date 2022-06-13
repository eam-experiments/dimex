# Copyright [2020] Luis Alberto Pineda Cortés, Rafael Morales Gamboa.
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

import csv
import gettext
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import constants
import dimex

# Translation
gettext.install('eam', localedir=None, codeset=None, names=None)

_ALL_IDS = 'ids.csv'
# milliseconds
min_for_crop = constants.phn_duration + 10

def plot_freqs(frequencies, prefix, es):
    plt.clf()
    x = dimex.labels_to_phns
    width = 5       # the width of the bars: can also be len(x) sequence
    plt.bar(x, frequencies, width)
    plt.xlabel(_('Phonemes'))
    plt.ylabel(_('Frequency'))
    graph_filename = constants.picture_filename(prefix + _('-english'), es)
    plt.savefig(graph_filename, dpi=600)

def plot_matrix(matrix, prefix, es):
    plt.clf()
    plt.figure(figsize=(6.4, 4.8))
    max_value = np.max(matrix)
    tick_labels = dimex.labels_to_phns
    seaborn.heatmap(matrix/max_value, xticklabels=tick_labels,
        yticklabels=tick_labels, vmin=0.0, vmax=1.0, cmap='coolwarm')
    plt.xlabel(_('Second Phoneme'))
    plt.ylabel(_('First Phoneme'))
    filename = constants.picture_filename(prefix + _('-english'), es)
    plt.savefig(filename, dpi=600)

def get_mods_ids(file_name):
    mods_ids = []
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        # Skip the header.
        next(reader, None)
        for row in reader:
            mods_ids.append(tuple(row))
    return mods_ids

def bigram_matrix(id_filename):
    mods_ids = get_mods_ids(id_filename)

    frequencies = np.zeros(constants.n_labels, dtype=np.double)
    matrix = np.zeros((constants.n_labels, constants.n_labels), dtype=np.double)
    counter = 0
    for mod, id in mods_ids:
        phnms_filename = dimex.get_phonemes_filename(mod, id)
        with open(phnms_filename) as file:
            reader=csv.reader(file, delimiter=' ')
            row = next(reader)
            # skip the headers
            while (len(row) == 0) or (row[0] != 'END'):
                row = next(reader, None)
            previous = constants.n_labels
            for row in reader:
                if len(row) < 3:
                    continue
                phn = row[2]
                if (phn == '.sil') or (phn == '.bn'):
                    continue
                label = dimex.phns_to_labels[phn]
                frequencies[label] += 1
                if previous != constants.n_labels:
                    matrix[previous, label] += 1
                previous = label
        counter += 1
        constants.print_counter(counter,100,10)
    return matrix, frequencies

if __name__== "__main__" :
    es = constants.ExperimentSettings()
    matrix, frequencies = bigram_matrix(_ALL_IDS)
    plot_matrix(matrix, 'bigrams', es)
    filename = constants.csv_filename('bigrams',es)
    np.savetxt(filename, matrix, fmt='%d', delimiter=',')
    totals = np.sum(matrix, axis=1)
    matrix = matrix / totals[:, None]
    filename = constants.data_filename('bigrams',es)
    np.save(filename, matrix)
    plot_freqs(frequencies, 'frequencies', es)
    filename = constants.csv_filename('frequencies', es)
    np.savetxt(filename, frequencies, fmt='%d', delimiter=',')
    frequencies = frequencies/np.sum(frequencies)
    filename = constants.data_filename('frequencies',es)
    np.save(filename, frequencies)

