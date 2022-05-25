from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dimex
import eam
import constants

stages = 6
tolerance = 0
learned = 4
sigma = 0.5
iota = 0.0
kappa = 0.0
extended = True
threshold = 1.0 /constants.n_labels
p_weight = 2.0/3.0
none = constants.n_labels
n_jobs=1
split_size = 20
runpath = f'runs-d{learned}-t{tolerance}-i{iota:.1f}-k{kappa:.1f}-s{sigma:.2f}'
constants.run_path = runpath
es = constants.ExperimentSettings(learned=learned, tolerance = tolerance, extended=extended,
        iota=iota, kappa=kappa, sigma=sigma)

print(f'Getting data from {constants.run_path}')

def remove_errors_fold(es, fold):
    prefix = constants.recognition_prefix
    filename = constants.recog_filename(prefix, es, fold)
    print(f'Reading file: {filename}')
    df = pd.read_csv(filename)
    correct_labels = phtolab(df['Correct'].values)
    network_labels = phtolab(df['Network'].values)
    memories_labels = phtolab(df['Memories'].values)
    network_labels = remove_errors(network_labels)
    memories_labels = remove_errors(memories_labels)
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
    df['Net2MemND'] = memories_to_network
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

def remove_errors(sequences):
    seqs_cleaned = []
    for labels in sequences:
        cleaned = []
        n = len(labels)
        previous = none
        for i in range(n):
            current = labels[i]
            nexto = none if i == (n - 1) else labels[i+1]
            p = current_prob(previous, current, nexto)
            all_probs.append(p)
            if p > threshold:
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
    return probs

if __name__== "__main__" :
    _INDI_PROBS_PREFIX = 'frequencies'
    _COND_PROBS_PREFIX = 'bigrams'
    i_probs = load_probs(_INDI_PROBS_PREFIX)
    c_probs = load_probs(_COND_PROBS_PREFIX)
    all_probs = []
    
    for stage in range(stages):
        es.stage = stage
        for fold in range(constants.n_folds):
            remove_errors_fold(es, fold)

    mean = np.mean(all_probs)
    stdv = np.std(all_probs)
    print(f'Mean prob: {mean}, Mean stdev: {stdv}')
