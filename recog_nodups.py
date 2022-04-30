import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dimex
import eam
import constants

stages = 10
tolerance = 0
learned = 4
sigma = 0.10
iota = 0.30
kappa = 1.50
extended = True
runpath = f'runs-d{learned}-t{tolerance}-i{iota:.1f}-k{kappa:.1f}-s{sigma:.2f}'
constants.run_path = runpath
es = constants.ExperimentSettings(learned=learned, tolerance = tolerance, extended=extended,
        iota=iota, kappa=kappa, sigma=sigma)

print(f'Getting data from {constants.run_path}')

def remove_duplicates_fold(es, fold):
    prefix = constants.recognition_prefix
    filename = constants.recog_filename(prefix, es, fold)
    print(f'Reading file: {filename}')
    df = pd.read_csv(filename)
    correct_labels = phtolab(df['Correct'].values)
    network_labels = phtolab(df['Network'].values)
    memories_labels = phtolab(df['Memories'].values)
    network_labels = remove_duplicates(network_labels)
    memories_labels = remove_duplicates(memories_labels)
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
    df['Network'] = network_labels
    df['Memories'] = memories_labels
    df['NetSize'] = network_sizes
    df['MemSize'] = memories_sizes
    df['Cor2Net'] = correct_to_network
    df['Cor2Mem'] = correct_to_memories
    df['Net2Mem'] = memories_to_network
    filename = constants.recog_filename(prefix, es, fold)
    df.to_csv(filename, index=False)

def stats_per_label(labels):
    stats = np.zeros(constants.n_labels, dtype=int)
    for label in labels:
        stats[label] += 1
    return stats

def distances(aes, bes):
    ds = []
    for a, b in zip(aes, bes):
        d = eam.levenshtein(a, b)
        ds.append(d)
    return ds

def remove_duplicates(labels):
    nodups = []
    for l in labels:
        nd = []
        prev = -1
        for v in l:
            if v == prev:
                continue
            else:
                nd.append(v)
                prev = v
        nodups.append(nd)
    return nodups

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

for stage in range(stages):
    es.stage = stage
    for fold in range(constants.n_folds):
        remove_duplicates_fold(es, fold)
