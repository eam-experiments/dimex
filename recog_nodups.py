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

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dimex
import eam
import constants

stages = 1
learned = 4
extended = True
extended_suffix = {False: '', True: '-x'}
p_weight = 0.5
none = constants.n_labels
n_jobs=1
split_size = 20
runpath = f'runs-d{learned}{extended_suffix[extended]}' 
constants.run_path = runpath

def remove_errors_fold(probs, es, fold):
    prefix = constants.recognition_prefix
    filename = constants.recog_filename(prefix, es, fold)
    df = pd.read_csv(filename)
    print(f'File {filename} read')
    correct_labels = phtolab(df['Correct'].values)
    network_labels = phtolab(df['Network'].values)
    memories_labels = phtolab(df['Memories'].values)
    network_labels = remove_errors(probs, network_labels)
    memories_labels = remove_errors(probs, memories_labels)
    stats = stats_per_label(correct_labels)
    print(stats)
    stats = stats_per_label(memories_labels)
    print(stats)
    network_sizes = [len(ls) for ls in network_labels]
    memories_sizes = [len(ls) for ls in memories_labels]
    correct_to_network = distances(correct_labels, network_labels)
    correct_to_memories = distances(correct_labels, memories_labels)
    memories_to_network = distances(memories_labels, network_labels)
    network_labels = labtoph(network_labels)
    memories_labels = labtoph(memories_labels)
    df['NetworkND'] = network_labels
    df['NetSizeND'] = network_sizes
    df['MemoriesND'] = memories_labels
    df['MemSizeND'] = memories_sizes
    df['Cor2NetND'] = correct_to_network
    df['Cor2MemND'] = correct_to_memories
    df['Mem2NetND'] = memories_to_network
    if 'Net2MemND' in df.columns:
        df.drop(columns='Net2MemND', inplace=True)
    filename = constants.recog_filename(prefix, es, fold)
    df.to_csv(filename, index=False)

def stats_per_label(labels):
    stats = np.zeros(constants.n_labels, dtype=int)
    for label in labels:
        stats[label] += 1
    return stats

def distances(aes, bes):
    ds = []
    for d in \
        Parallel(n_jobs=n_jobs, verbose=50)(
            delayed(distances_aux)(pairs) for pairs in eam.split_every(split_size, zip(aes, bes))):
        ds += d
    return ds

def distances_aux(pairs):
    ds = []
    for a, b in pairs:
        d = eam.levenshtein(a, b)
        ds.append(d)
    return ds

def remove_errors(probs, sequences):
    seqs_cleaned = []
    for labels in sequences:
        cleaned = []
        n = len(labels)
        previous = none
        for i in range(n):
            current = labels[i]
            nexto = none if i == (n - 1) else labels[i+1]
            p = current_prob(previous, current, nexto)
            probs.append(p)
            if p > i_probs[current]:
                cleaned.append(current)
                previous = current
        seqs_cleaned.append(cleaned)
    return seqs_cleaned

def current_prob(previous, current,  nexto):
    pCP = i_probs[current] if previous == none \
        else c_probs[previous, current]
    pCN = i_probs[current] if nexto == none \
        else c_probs[current, nexto]*i_probs[current]/i_probs[nexto]
    p = p_weight*pCP + (1.0 - p_weight)*pCN
    return p

def phtolab(phonemes):
    labels = []
    for s in phonemes:
        l = dimex.phonemesToLabels(s)
        labels.append(l)
    return labels

def labtoph(labels):
    phonemes = []
    for l in labels:
        s = dimex.labelsToPhonemes(l)
        phonemes.append(s)
    return phonemes

def load_probs(prefix):
    es = constants.ExperimentSettings()
    filename = constants.data_filename(prefix, es)
    probs = np.load(filename)
    print(f'File {filename} read.')
    return probs

if __name__== "__main__" :
    print(f'Getting data from {constants.run_path}')
    _INDI_PROBS_PREFIX = 'frequencies'
    _COND_PROBS_PREFIX = 'bigrams'
    i_probs = load_probs(_INDI_PROBS_PREFIX)
    c_probs = load_probs(_COND_PROBS_PREFIX)
    all_probs = []
    es = constants.ExperimentSettings(learned=learned, extended=extended)
    for stage in range(stages):
        es.stage = stage
        print(f'Processing stage {stage}')
        for fold in range(constants.n_folds):
            print(f'\tProcessing fold {fold}')
            remove_errors_fold(all_probs, es, fold)

    mean = np.mean(all_probs)
    stdv = np.std(all_probs)
    print(f'Mean prob: {mean}, Mean stdev: {stdv}')
