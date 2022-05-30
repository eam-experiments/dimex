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
  eam -h | --help
  eam (-n | -f | -a | -c | -e | -i | -r) <stage> [--learned=<learned_data>] [-x] [--tolerance=<tolerance>] [--sigma=<sigma>] [--iota=<iota>] [--kappa=<kappa>] [--runpath=<runpath>] [ -l (en | es) ]

Options:
  -h        Show this screen.
  -n        Trains the encoder + classifier Neural networks.
  -f        Generates Features for all data using the encoder.
  -a        Trains the encoder + decoder (Autoencoder) neural networks.
  -c        Generates graphs Characterizing classes of features (by label).
  -e        Run the experiment 1 (Evaluation).
  -i        Increase the amount of data (learning).
  -r        Run the experiment 2 (Recognition).
  --learned=<learned_data>      Selects which learneD Data is used for evaluation, recognition or learning [default: 0].
  -x        Use the eXtended data set as testing data for memory.
  --tolerance=<tolerance>       Allow Tolerance (unmatched features) in memory [default: 0].
  --sigma=<sigma>   Scale of standard deviation of the distribution of influence of the cue [default: 0.25]
  --iota=<iota>     Scale of the expectation (mean) required as minimum for the cue to be accepted by a memory [default: 0.0]
  --kappa=<kappa>   Scale of the expectation (mean) rquiered as minimum for the cue to be accepted by the system of memories [default: 0.0]
  --runpath=<runpath>           Sets the path to the directory where everything will be saved [default: runs]
  -l        Chooses Language for graphs.            

The parameter <stage> indicates the stage of learning from which data is used.
Default is the last one.
"""
from docopt import docopt
import copy
import csv
from datetime import datetime
import sys
sys.setrecursionlimit(10000)
import gc
import gettext
from itertools import islice
import numpy as np
from numpy.core.einsumfunc import einsum_path
from joblib import Parallel, delayed
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
import json
from numpy.core.defchararray import array
import seaborn
from associative import AssociativeMemory, AssociativeMemorySystem
import ciempiess
import constants
import dimex
import recnet

# Translation
gettext.install('ame', localedir=None, codeset=None, names=None)


def plot_pre_graph (pre_mean, rec_mean, acc_mean, ent_mean, \
    pre_std, rec_std, acc_std, ent_std, es, tag = '', \
        xlabels = constants.memory_sizes, xtitle = None, \
        ytitle = None):

    plt.clf()
    plt.figure(figsize=(6.4,4.8))

    full_length = 100.0
    step = 0.1
    main_step = full_length/len(xlabels)
    x = np.arange(0, full_length, main_step)

    # One main step less because levels go on sticks, not
    # on intervals.
    xmax = full_length - main_step + step

    # Gives space to fully show markers in the top.
    ymax = full_length + 2

    # Replace undefined precision with 1.0.
    pre_mean = np.nan_to_num(pre_mean, copy=False, nan=100.0)

    plt.errorbar(x, pre_mean, fmt='r-o', yerr=pre_std, label=_('Precision'))
    plt.errorbar(x, rec_mean, fmt='b--s', yerr=rec_std, label=_('Recall'))
    if not ((acc_mean is None) or (acc_std is None)):
        plt.errorbar(x, acc_mean, fmt='y:d', yerr=acc_std, label=_('Accuracy'))

    plt.xlim(0, xmax)
    plt.ylim(0, ymax)
    plt.xticks(x, xlabels)

    if xtitle is None:
        xtitle = _('Range Quantization Levels')
    if ytitle is None: 
        ytitle = _('Percentage')

    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.legend(loc=4)
    plt.grid(True)

    entropy_labels = [str(e) for e in np.around(ent_mean, decimals=1)]

    cmap = mpl.colors.LinearSegmentedColormap.from_list('mycolors',['cyan','purple'])
    Z = [[0,0],[0,0]]
    levels = np.arange(0.0, xmax, step)
    CS3 = plt.contourf(Z, levels, cmap=cmap)

    cbar = plt.colorbar(CS3, orientation='horizontal')
    cbar.set_ticks(x)
    cbar.ax.set_xticklabels(entropy_labels)
    cbar.set_label(_('Entropy'))

    s = tag + 'graph_prse_MEAN' + _('-english')
    graph_filename = constants.picture_filename(s, es)
    plt.savefig(graph_filename, dpi=600)


def plot_size_graph (response_size, size_stdev, es):
    plt.clf()

    full_length = 100.0
    step = 0.1
    main_step = full_length/len(response_size)
    x = np.arange(0, full_length, main_step)

    # One main step less because levels go on sticks, not
    # on intervals.
    xmax = full_length - main_step + step
    ymax = constants.n_labels

    plt.errorbar(x, response_size, fmt='g-D', yerr=size_stdev, label=_('Average number of responses'))
    plt.xlim(0, xmax)
    plt.ylim(0, ymax)
    plt.xticks(x, constants.memory_sizes)
    plt.yticks(np.arange(0,ymax+1, 1), range(constants.n_labels+1))

    plt.xlabel(_('Range Quantization Levels'))
    plt.ylabel(_('Size'))
    plt.legend(loc=1)
    plt.grid(True)

    graph_filename = constants.picture_filename('graph_size_MEAN' + _('-english'), es)
    plt.savefig(graph_filename, dpi=600)


def plot_behs_graph(no_response, no_correct, no_chosen, correct, es):

    for i in range(len(no_response)):
        total = (no_response[i] + no_correct[i] + no_chosen[i] + correct[i])/100.0
        no_response[i] /= total
        no_correct[i] /= total
        no_chosen[i] /= total
        correct[i] /= total

    plt.clf()

    full_length = 100.0
    step = 0.1
    main_step = full_length/len(constants.memory_sizes)
    x = np.arange(0.0, full_length, main_step)

    # One main step less because levels go on sticks, not
    # on intervals.
    xmax = full_length - main_step + step
    ymax = full_length
    width = 5       # the width of the bars: can also be len(x) sequence

    plt.bar(x, correct, width, label=_('Correct response chosen'))
    cumm = np.array(correct)
    plt.bar(x, no_chosen,  width, bottom=cumm, label=_('Correct response not chosen'))
    cumm += np.array(no_chosen)
    plt.bar(x, no_correct, width, bottom=cumm, label=_('No correct response'))
    cumm += np.array(no_correct)
    plt.bar(x, no_response, width, bottom=cumm, label=_('No responses'))

    plt.xlim(-width, xmax + width)
    plt.ylim(0.0, ymax)
    plt.xticks(x, constants.memory_sizes)

    plt.xlabel(_('Range Quantization Levels'))
    plt.ylabel(_('Labels'))

    plt.legend(loc=0)
    plt.grid(axis='y')

    graph_filename = constants.picture_filename('graph_behaviours_MEAN' + _('-english'), es)
    plt.savefig(graph_filename, dpi=600)


def plot_features_graph(domain, means, stdevs, es):
    """ Draws the characterist shape of features per label.

    The graph is a dots and lines graph with error bars denoting standard deviations.
    """
    ymin = np.PINF
    ymax = np.NINF
    for i in constants.all_labels:
        yn = (means[i] - stdevs[i]).min()
        yx = (means[i] + stdevs[i]).max()
        ymin = ymin if ymin < yn else yn
        ymax = ymax if ymax > yx else yx
    main_step = 100.0 / domain
    xrange = np.arange(0, 100, main_step)
    fmts = constants.label_formats
    for i in constants.all_labels:
        plt.clf()
        plt.figure(figsize=(12,5))
        plt.errorbar(xrange, means[i], fmt=fmts[i], yerr=stdevs[i], label=str(i))
        plt.xlim(0, 100)
        plt.ylim(ymin, ymax)
        plt.xticks(xrange, labels='')
        plt.xlabel(_('Features'))
        plt.ylabel(_('Values'))
        plt.legend(loc='right')
        plt.grid(True)
        filename = constants.features_name(es) + '-' + str(i).zfill(3) + _('-english')
        plt.savefig(constants.picture_filename(filename, es), dpi=600)


def plot_conf_matrix(matrix, tags, prefix, es):
    plt.clf()
    plt.figure(figsize=(6.4,4.8))
    seaborn.heatmap(matrix, xticklabels=tags, yticklabels=tags, vmin = 0.0, vmax=1.0, annot=False, cmap='Blues')
    plt.xlabel(_('Prediction'))
    plt.ylabel(_('Label'))
    filename = constants.picture_filename(prefix, es)
    plt.savefig(filename, dpi=600)


def plot_memory(memory: AssociativeMemory, prefix, es, fold):
    plt.clf()
    plt.figure(figsize=(6.4, 4.8))
    seaborn.heatmap(memory.relation/memory.max_value, vmin=0.0, vmax=1.0,
                    annot=False, cmap='coolwarm')
    plt.xlabel(_('Characteristics'))
    plt.ylabel(_('Values'))
    filename = constants.picture_filename(prefix, es, fold)
    plt.savefig(filename, dpi=600)

def plot_memories(ams, es, fold):
    for label in ams:
        prefix = f'memory-{label}-state'
        plot_memory(ams[label], prefix, es, fold)

def get_label(memories, weights = None, entropies = None):
    if len(memories) == 1:
        return memories[0]
    random.shuffle(memories)
    if (entropies is None) or (weights is None):
        return memories[0]
    else:
        i = memories[0] 
        entropy = entropies[i]
        weight = weights[i]
        penalty = entropy/weight if weight > 0 else float('inf')
        for j in memories[1:]:
            entropy = entropies[j]
            weight = weights[j]
            new_penalty = entropy/weight if weight > 0 else float('inf')
            if new_penalty < penalty:
                i = j
                penalty = new_penalty
        return i

def msize_features(features, msize, min_value, max_value):
    return np.round((msize-1)*(features-min_value) / (max_value-min_value)).astype(np.int16)

def rsize_recall(recall, msize, min_value, max_value):
    return (max_value - min_value)*recall/(msize-1) + min_value

TP = (0,0)
FP = (0,1)
FN = (1,0)
TN = (1,1)

def conf_sum(cms, t):
    return np.sum([cms[i][t] for i in range(len(cms))])

def memories_precision(cms):
    total = conf_sum(cms, TP) + conf_sum(cms, FN)
    if total == 0:
        return 0.0
    precision = 0.0
    for m in range(len(cms)):
        denominator = (cms[m][TP] + cms[m][FP])
        if denominator == 0:
            m_precision = 1.0
        else:
            m_precision = cms[m][TP] / denominator
        weight = (cms[m][TP] + cms[m][FN]) / total
        precision += weight*m_precision
    return precision

def memories_recall(cms):
    total = conf_sum(cms, TP) + conf_sum(cms, FN)
    if total == 0:
        return 0.0
    recall = 0.0
    for m in range(len(cms)):
        m_recall = cms[m][TP] / (cms[m][TP] + cms[m][FN])
        weight = (cms[m][TP] + cms[m][FN]) / total
        recall += weight*m_recall
    return recall
 
def memories_accuracy(cms):
    total = conf_sum(cms, TP) + conf_sum(cms, FN)
    if total == 0:
        return 0.0
    accuracy = 0.0
    for m in range(len(cms)):
        m_accuracy = (cms[m][TP] + cms[m][TN]) / total
        weight = (cms[m][TP] + cms[m][FN]) / total
        accuracy += weight*m_accuracy
    return accuracy

def register_in_memory(memory, features_iterator):
    for features in features_iterator:
        memory.register(features)

def memory_entropy(m, memory: AssociativeMemory):
    return m, memory.entropy

def recognize_by_memory(fl_pairs, ams, entropy):
    n_mems = constants.n_labels
    response_size = np.zeros(n_mems, dtype=int)
    cms = np.zeros((n_mems, 2, 2), dtype='int')
    behaviour = np.zeros(
        (n_mems, constants.n_behaviours), dtype=np.float64)
    for features, label in fl_pairs:
        correct = label
        memories = []
        weights = {}
        for k in ams:
            recognized, weight = ams[k].recognize(features)
            if recognized:
                memories.append(k)
                weights[k] = weight
                response_size[correct] += 1
            # For calculation of per memory precision and recall
            cms[k][TP] += (k == correct) and recognized
            cms[k][FP] += (k != correct) and recognized
            cms[k][TN] += not ((k == correct) or recognized)
            cms[k][FN] += (k == correct) and not recognized
        if len(memories) == 0:
            # Register empty case
            behaviour[correct, constants.no_response_idx] += 1
        elif not (correct in memories):
            behaviour[correct, constants.no_correct_response_idx] += 1
        else:
            l = get_label(memories, weights, entropy)
            if l != correct:
                behaviour[correct, constants.no_correct_chosen_idx] += 1
            else:
                behaviour[correct, constants.correct_response_idx] += 1
    return response_size, cms, behaviour

def split_by_label(fl_pairs):
    label_dict = {}
    for label in range(constants.n_labels):
        label_dict[label] = []
    for features, label in fl_pairs:
        label_dict[label].append(features)
    return label_dict.items()

def split_every(n, iterable):
    i = iter(iterable)
    piece = list(islice(i, n))
    while piece:
        yield piece
        piece = list(islice(i, n))

def get_ams_results(midx, msize, domain, trf, tef, trl, tel, 
    es: constants.ExperimentSettings, fold):
    # Round the values
    max_value = trf.max()
    other_value = tef.max()
    max_value = max_value if max_value > other_value else other_value

    min_value = trf.min()
    other_value = tef.min()
    min_value = min_value if min_value < other_value else other_value

    trf_rounded = msize_features(trf, msize, min_value, max_value)
    tef_rounded = msize_features(tef, msize, min_value, max_value)

    n_labels = constants.n_labels
    n_mems = n_labels

    measures = np.zeros(constants.n_measures, dtype=np.float64)
    entropy = np.zeros(n_mems, dtype=np.float64)
    behaviour = np.zeros(
        (constants.n_labels, constants.n_behaviours), dtype=np.float64)

    # Confusion matrix for calculating precision and recall per memory.
    cms = np.zeros((n_mems, 2, 2), dtype='int')

    # Create the required associative memories.
    ams = dict.fromkeys(range(n_mems))
    for m in ams:
        ams[m] = AssociativeMemory(domain, msize, es.tolerance, 
                    es.sigma, es.iota, es.kappa)
    # Registration in parallel, per label.
    Parallel(n_jobs=constants.n_jobs, require='sharedmem', verbose=50)(
        delayed(register_in_memory)(ams[label], features_list) \
            for label, features_list in split_by_label(zip(trf_rounded, trl)))
    print(f'Filling of memories done for fold {fold}')

    # Calculate entropies
    means = []
    for m in ams:
        entropy[m] = ams[m].entropy
        means.append(ams[m].mean)

    # Recognition
    response_size = np.zeros(n_mems, dtype=int)
    split_size = 500
    for rsize, scms, sbehavs in \
         Parallel(n_jobs=constants.n_jobs, verbose=50)(
            delayed(recognize_by_memory)(fl_pairs, ams, entropy) \
            for fl_pairs in split_every(split_size, zip(tef_rounded, tel))):
        response_size = response_size + rsize
        cms  = cms + scms
        behaviour = behaviour + sbehavs
    counters = [np.count_nonzero(tel == i) for i in range(n_labels)]
    counters = np.array(counters)
    behaviour[:,constants.response_size_idx] = response_size/counters
    all_responses = len(tef_rounded) - np.sum(behaviour[:,constants.no_response_idx], axis=0)
    all_precision = np.sum(behaviour[:, constants.correct_response_idx], axis=0)/float(all_responses)
    all_recall = np.sum(behaviour[:, constants.correct_response_idx], axis=0)/float(len(tef_rounded))

    behaviour[:,constants.precision_idx] = all_precision
    behaviour[:,constants.recall_idx] = all_recall

    positives = conf_sum(cms, TP) + conf_sum(cms, FP)
    details = True
    if positives == 0:
        print('No memory responded')
        measures[constants.precision_idx] = 1.0
        details = False
    else:
        measures[constants.precision_idx] = memories_precision(cms)
    measures[constants.recall_idx] = memories_recall(cms)
    measures[constants.accuracy_idx] = memories_accuracy(cms)
    measures[constants.entropy_idx] = np.mean(entropy)
 
    if details:
        for i in range(n_mems):
            positives = cms[i][TP] + cms[i][FP]
            if positives == 0:
                print(f'Memory {i} of size {msize} in fold {fold} did not respond.')
    return (midx, measures, behaviour, cms)
    

def test_memories(domain, es):
    entropy = []
    precision = []
    recall = []
    accuracy = []
    all_precision = []
    all_recall = []
    all_cms = []


    no_response = []
    no_correct_response = []
    no_correct_chosen = []
    correct_chosen = []
    response_size = []

    print('Testing the memories')

    for fold in range(constants.n_folds):
        gc.collect()
        print(f'Fold: {fold}')
        suffix = constants.filling_suffix
        filling_features_filename = constants.features_name(es) + suffix        
        filling_features_filename = constants.data_filename(filling_features_filename, es, fold)
        filling_labels_filename = constants.labels_name(es) + suffix        
        filling_labels_filename = constants.data_filename(filling_labels_filename, es, fold)

        suffix = constants.testing_suffix
        testing_features_filename = constants.features_name(es) + suffix        
        testing_features_filename = constants.data_filename(testing_features_filename, es, fold)
        testing_labels_filename = constants.labels_name(es) + suffix        
        testing_labels_filename = constants.data_filename(testing_labels_filename, es, fold)

        filling_features = np.load(filling_features_filename)
        filling_labels = np.load(filling_labels_filename)
        testing_features = np.load(testing_features_filename)
        testing_labels = np.load(testing_labels_filename)

        measures_per_size = np.zeros(
            (len(constants.memory_sizes), constants.n_measures),
            dtype=np.float64)
        behaviours = np.zeros(
            (constants.n_labels,
            len(constants.memory_sizes),
            constants.n_behaviours))
        list_measures = []
        list_cms = []
        for midx, msize in enumerate(constants.memory_sizes):
            results = get_ams_results(midx, msize, domain,
                filling_features, testing_features, filling_labels, testing_labels, es, fold)
            list_measures.append(results)
        for midx, measures, behaviour, cms in list_measures:
            measures_per_size[midx, :] = measures
            behaviours[:, midx, :] = behaviour
            list_cms.append(cms)
        
        ###################################################################3##
        # Measures by memory size

        # Average entropy among al digits.
        entropy.append(measures_per_size[:,constants.entropy_idx])

        # Average precision and recall as percentage
        precision.append(measures_per_size[:,constants.precision_idx]*100)
        recall.append(measures_per_size[:,constants.recall_idx]*100)
        accuracy.append(measures_per_size[:,constants.accuracy_idx]*100)

        all_precision.append(np.mean(behaviours[:, :, constants.precision_idx], axis=0) * 100)
        all_recall.append(np.mean(behaviours[:, :, constants.recall_idx], axis=0) * 100)
        all_cms.append(np.array(list_cms))
        no_response.append(np.sum(behaviours[:, :, constants.no_response_idx], axis=0))
        no_correct_response.append(np.sum(behaviours[:, :, constants.no_correct_response_idx], axis=0))
        no_correct_chosen.append(np.sum(behaviours[:, :, constants.no_correct_chosen_idx], axis=0))
        correct_chosen.append(np.sum(behaviours[:, :, constants.correct_response_idx], axis=0))
        response_size.append(np.mean(behaviours[:, :, constants.response_size_idx], axis=0))

    # Every row is training fold, and every column is a memory size.
    entropy = np.array(entropy)
    precision = np.array(precision)
    recall = np.array(recall)
    accuracy = np.array(accuracy)

    all_precision = np.array(all_precision)
    all_recall = np.array(all_recall)
    all_cms = np.array(all_cms)

    average_entropy = np.mean(entropy, axis=0)
    stdev_entropy = np.std(entropy, axis=0)
    average_precision = np.mean(precision, axis=0)
    stdev_precision = np.std(precision, axis=0)
    average_recall = np.mean(recall, axis=0)
    stdev_recall = np.std(recall, axis=0)
    average_accuracy = np.mean(accuracy, axis=0)
    stdev_accuracy = np.std(accuracy, axis=0)

    no_response = np.array(no_response)
    no_correct_response = np.array(no_correct_response)
    no_correct_chosen = np.array(no_correct_chosen)
    correct_chosen = np.array(correct_chosen)
    response_size = np.array(response_size)

    all_precision_average = np.mean(all_precision, axis=0)
    all_precision_stdev = np.std(all_precision, axis=0)
    all_recall_average = np.mean(all_recall, axis=0)
    all_recall_stdev = np.std(all_recall, axis=0)
    main_no_response = np.mean(no_response, axis=0)
    main_no_correct_response = np.mean(no_correct_response, axis=0)
    main_no_correct_chosen = np.mean(no_correct_chosen, axis=0)
    main_correct_chosen = np.mean(correct_chosen, axis=0)
    main_response_size = np.mean(response_size, axis=0)
    main_response_size_stdev = np.std(response_size, axis=0)

    best_memory_size = constants.memory_sizes[
        np.argmax(main_correct_chosen)]
    main_behaviours = [main_no_response, main_no_correct_response, \
        main_no_correct_chosen, main_correct_chosen, main_response_size]

    np.savetxt(constants.csv_filename('memory_average_precision', es), precision, delimiter=',')
    np.savetxt(constants.csv_filename('memory_average_recall', es), recall, delimiter=',')
    np.savetxt(constants.csv_filename('memory_average_accuracy', es), accuracy, delimiter=',')
    np.savetxt(constants.csv_filename('memory_average_entropy', es), entropy, delimiter=',')
    np.savetxt(constants.csv_filename('all_precision', es), all_precision, delimiter=',')
    np.savetxt(constants.csv_filename('all_recall', es), all_recall, delimiter=',')
    np.savetxt(constants.csv_filename('main_behaviours', es), main_behaviours, delimiter=',')
    np.save(constants.data_filename('memory_cms', es), all_cms)
    np.save(constants.data_filename('behaviours', es), behaviours)
    plot_pre_graph(average_precision, average_recall, average_accuracy, average_entropy,\
        stdev_precision, stdev_recall, stdev_accuracy, stdev_entropy, es)
    plot_pre_graph(all_precision_average, all_recall_average, None, average_entropy, \
        all_precision_stdev, all_recall_stdev, None, stdev_entropy, es, 'overall')
    plot_size_graph(main_response_size, main_response_size_stdev, es)
    plot_behs_graph(main_no_response, main_no_correct_response, main_no_correct_chosen,\
        main_correct_chosen, es)
    print('Memory size evaluation completed!')
    return best_memory_size

def remember_by_memory(fl_pairs, ams, entropy):
    n_mems = constants.n_labels
    cms = np.zeros((n_mems, 2, 2), dtype='int')
    cmatrix = np.zeros((2,2), dtype='int')
    mismatches = 0
    for features, label in fl_pairs:
        mismatches += ams[label].mismatches(features)
        memories = []
        weights = {}
        for k in ams:
            recognized, weight = ams[k].recognize(features)
            if recognized:
                memories.append(k)
                weights[k] = weight
            # For calculation of per memory precision and recall
            cms[k][TP] += (k == label) and recognized
            cms[k][FP] += (k != label) and recognized
            cms[k][TN] += not ((k == label) or recognized)
            cms[k][FN] += (k == label) and not recognized
        if (len(memories) == 0):
            cmatrix[FN] += 1
        else:
            l = get_label(memories, weights, entropy)
            if l == label:
                cmatrix[TP] += 1
            else:
                cmatrix[FP] += 1
    return mismatches, cms, cmatrix


def get_recalls(ams, msize, domain, min_value, max_value, trf, trl, tef, tel, idx, fill):
    n_mems = constants.n_labels

    # To store precisión, recall, accuracy and entropies
    measures = np.zeros(constants.n_measures, dtype=np.float64)
    entropy = np.zeros(n_mems, dtype=np.float64)

    # Confusion matrix for calculating precision, recall and accuracy
    # per memory.
    cms = np.zeros((n_mems, 2, 2))
    TP = (0,0)
    FP = (0,1)
    FN = (1,0)
    TN = (1,1)

    # Confusion matrix for calculating overall precision and recall.
    cmatrix = np.zeros((2,2))

    # Registration in parallel, per label.
    Parallel(n_jobs=constants.n_jobs, require='sharedmem', verbose=50)(
        delayed(register_in_memory)(ams[label], features_list) \
            for label, features_list in split_by_label(zip(trf, trl)))

    print(f'Filling of memories done for idx {idx}')

    # Calculate entropies
    means = []
    for m in ams:
        entropy[m] = ams[m].entropy
        means.append(ams[m].mean)

    # Total number of differences between features and memories.
    mismatches = 0
    split_size = 500
    for mmatches, scms, cmatx in \
         Parallel(n_jobs=constants.n_jobs, verbose=50)(
            delayed(remember_by_memory)(fl_pairs, ams, entropy) \
            for fl_pairs in split_every(split_size, zip(tef, tel))):
        mismatches += mmatches
        cms  = cms + scms
        cmatrix = cmatrix + cmatx
    positives = conf_sum(cms, TP) + conf_sum(cms, FP)
    details = True
    if positives == 0:
        print('No memory responded')
        measures[constants.precision_idx] = 1.0
        details = False
    else:
        measures[constants.precision_idx] = memories_precision(cms)
    measures[constants.recall_idx] = memories_recall(cms)
    measures[constants.accuracy_idx] = memories_accuracy(cms)
    measures[constants.entropy_idx] = np.mean(entropy)
 
    if details:
        for i in range(n_mems):
            positives = cms[i][TP] + cms[i][FP]
            if positives == 0:
                print(f'Memory {i} filled with {fill} in run {idx} did not respond.')
    positives = cmatrix[TP] + cmatrix[FP]
    if positives == 0:
        print(f'System filled with {fill} in run {idx} did not respond.')
        total_precision = 1.0
    else: 
        total_precision = cmatrix[TP] / positives
    total_recall = cmatrix[TP] / len(tel)
    mismatches /= len(tel)
    return measures, total_precision, total_recall, mismatches

def test_recalling_fold(n_memories, mem_size, domain, es, fold):
    # Create the required associative memories.
    ams = dict.fromkeys(range(n_memories))
    for j in ams:
        ams[j] = AssociativeMemory(domain, mem_size, es.tolerance,
                    es.sigma, es.iota, es.kappa)

    suffix = constants.filling_suffix
    filling_features_filename = constants.features_name(es) + suffix        
    filling_features_filename = constants.data_filename(filling_features_filename, es, fold)
    filling_labels_filename = constants.labels_name(es) + suffix        
    filling_labels_filename = constants.data_filename(filling_labels_filename, es, fold)

    suffix = constants.testing_suffix
    testing_features_filename = constants.features_name(es) + suffix        
    testing_features_filename = constants.data_filename(testing_features_filename, es, fold)
    testing_labels_filename = constants.labels_name(es) + suffix        
    testing_labels_filename = constants.data_filename(testing_labels_filename, es, fold)

    filling_features = np.load(filling_features_filename)
    filling_labels = np.load(filling_labels_filename)
    testing_features = np.load(testing_features_filename)
    testing_labels = np.load(testing_labels_filename)

    filling_max = filling_features.max()
    testing_max = testing_features.max()
    fillin_min = filling_features.min()
    testing_min = testing_features.min()

    maximum = filling_max if filling_max > testing_max else testing_max
    minimum = fillin_min if fillin_min < testing_min else testing_min

    filling_features = msize_features(filling_features, mem_size, minimum, maximum)
    testing_features = msize_features(testing_features, mem_size, minimum, maximum)

    total = len(filling_labels)
    percents = np.array(constants.memory_fills)
    steps = np.round(total*percents/100.0).astype(int)

    fold_entropies = []
    fold_precision = []
    fold_recall = []
    fold_accuracy = []
    total_precisions = []
    total_recalls = []
    mismatches = []

    start = 0
    for end in steps:
        features = filling_features[start:end]
        labels = filling_labels[start:end]

        # recalls, measures, step_precision, step_recall, mis_count = get_recalls(ams, mem_size, domain, \
        measures, step_precision, step_recall, mis_count = get_recalls(ams, mem_size, domain, \
            minimum, maximum, features, labels, testing_features, testing_labels, fold, end)

        # A list of tuples (position, label, features)
        # fold_recalls += recalls
        # An array with average entropy per step.
        fold_entropies.append(measures[constants.entropy_idx])
        # Arrays with precision, recall and accuracy per step
        fold_precision.append(measures[constants.precision_idx])
        fold_recall.append(measures[constants.recall_idx])
        fold_accuracy.append(measures[constants.accuracy_idx])
        # Overall recalls and precisions per step
        total_recalls.append(step_recall)
        total_precisions.append(step_precision)
        mismatches.append(mis_count)
        start = end
    plot_memories(ams, es, fold)
    fold_entropies = np.array(fold_entropies)
    fold_precision = np.array(fold_precision)
    fold_recall = np.array(fold_recall)
    fold_accuracy = np.array(fold_accuracy)
    total_precisions = np.array(total_precisions)
    total_recalls = np.array(total_recalls)
    mismatches = np.array(mismatches)
    return fold, fold_entropies, fold_precision, \
        fold_recall, fold_accuracy, total_precisions, total_recalls, mismatches

def test_recalling(domain, mem_size, es):
    n_memories = constants.n_labels
    memory_fills = constants.memory_fills
    testing_folds = constants.n_folds
    # All recalls, per memory fill and fold.
    # all_memories = {}
    # All entropies, precision, and recall, per fold, and fill.
    total_entropies = np.zeros((testing_folds, len(memory_fills)))
    total_precisions = np.zeros((testing_folds, len(memory_fills)))
    total_recalls = np.zeros((testing_folds, len(memory_fills)))
    total_accuracies = np.zeros((testing_folds, len(memory_fills)))
    sys_precisions = np.zeros((testing_folds, len(memory_fills)))
    sys_recalls = np.zeros((testing_folds, len(memory_fills)))
    total_mismatches = np.zeros((testing_folds, len(memory_fills)))

    list_results = []
    for fold in range(testing_folds):
        results = test_recalling_fold(n_memories, mem_size, domain, es, fold)
        list_results.append(results)
    # for fold, memories, entropy, precision, recall, accuracy, \
    for fold, entropy, precision, recall, accuracy, \
        sys_precision, sys_recall, mismatches in list_results:
        # all_memories[fold] = memories
        total_precisions[fold] = precision
        total_recalls[fold] = recall
        total_accuracies[fold] = accuracy
        total_mismatches[fold] = mismatches
        total_entropies[fold] = entropy
        sys_precisions[fold] = sys_precision
        sys_recalls[fold] = sys_recall
    main_avrge_entropies = np.mean(total_entropies,axis=0)
    main_stdev_entropies = np.std(total_entropies, axis=0)
    main_avrge_mprecision = np.mean(total_precisions,axis=0)
    main_stdev_mprecision = np.std(total_precisions,axis=0)
    main_avrge_mrecall = np.mean(total_recalls,axis=0)
    main_stdev_mrecall = np.std(total_recalls,axis=0)
    main_avrge_maccuracy = np.mean(total_accuracies,axis=0)
    main_stdev_maccuracy = np.std(total_accuracies,axis=0)
    main_avrge_sys_precision = np.mean(sys_precisions,axis=0)
    main_stdev_sys_precision = np.std(sys_precisions,axis=0)
    main_avrge_sys_recall = np.mean(sys_recalls,axis=0)
    main_stdev_sys_recall = np.std(sys_recalls,axis=0)
    
    np.savetxt(constants.csv_filename('main_average_precision', es), \
        main_avrge_mprecision, delimiter=',')
    np.savetxt(constants.csv_filename('main_average_recall', es), \
        main_avrge_mrecall, delimiter=',')
    np.savetxt(constants.csv_filename('main_average_accuracy', es), \
        main_avrge_maccuracy, delimiter=',')
    np.savetxt(constants.csv_filename('main_average_entropy', es), \
        main_avrge_entropies, delimiter=',')
    np.savetxt(constants.csv_filename('main_stdev_precision', es), \
        main_stdev_mprecision, delimiter=',')
    np.savetxt(constants.csv_filename('main_stdev_recall', es), \
        main_stdev_mrecall, delimiter=',')
    np.savetxt(constants.csv_filename('main_stdev_accuracy', es), \
        main_stdev_maccuracy, delimiter=',')
    np.savetxt(constants.csv_filename('main_stdev_entropy', es), \
        main_stdev_entropies, delimiter=',')
    np.savetxt(constants.csv_filename('main_total_recalls', es), \
        main_avrge_sys_recall, delimiter=',')
    np.savetxt(constants.csv_filename('main_total_precision', es), \
        main_avrge_sys_precision, delimiter=',')
    np.savetxt(constants.csv_filename('main_total_mismatches', es), \
        total_mismatches, delimiter=',')

    plot_pre_graph(main_avrge_mprecision*100, main_avrge_mrecall*100, main_avrge_maccuracy*100, main_avrge_entropies,\
        main_stdev_mprecision*100, main_stdev_mrecall*100, main_stdev_maccuracy*100, main_stdev_entropies, es, 'recall-', \
            xlabels = constants.memory_fills, xtitle = _('Percentage of memory corpus'))
    plot_pre_graph(main_avrge_sys_precision*100, main_avrge_sys_recall*100, None, main_avrge_entropies, \
        main_stdev_sys_precision*100, main_stdev_sys_recall*100, None, main_stdev_entropies, es, 'total_recall-', \
            xlabels = constants.memory_fills, xtitle = _('Percentage of memory corpus'))

    bfp = best_filling_percentage(main_avrge_sys_precision, main_avrge_sys_recall)
    print('Best filling percent: ' + str(bfp))
    print('Filling evaluation completed!')
    return bfp


def best_filling_percentage(precisions, recalls):
    n = 0
    i = 0
    avg = -float('inf')
    for precision, recall in zip(precisions, recalls):
        new_avg = (precision + recall) / 2.0
        if avg < new_avg:
            n = i
            avg = new_avg
        i += 1
    return constants.memory_fills[n]


def get_all_data(prefix, es):
    data = None
    for fold in range(constants.n_folds):
        filename = constants.data_filename(prefix, es, fold)
        if data is None:
            data = np.load(filename)
        else:
            newdata = np.load(filename)
            data = np.concatenate((data, newdata), axis=0)
    return data

def save_history(history, prefix, es):
    """ Saves the stats of neural networks.

    Neural networks stats may come either as a History object, that includes
    a History.history dictionary with stats, or directly as a dictionary.
    """
    stats = {}
    stats['history'] = []
    for h in history:
        while not ((type(h) is dict) or (type(h) is list)):
            h = h.history
        stats['history'].append(h)
    with open(constants.json_filename(prefix,es), 'w') as outfile:
        json.dump(stats, outfile)

def save_learn_params(mem_size, fill_percent, es):
    name = constants.learn_params_name(es)
    filename = constants.data_filename(name, es)
    np.save(filename, np.array([mem_size, fill_percent], dtype=int))

def load_learn_params(es):
    name = constants.learn_params_name(es)
    filename = constants.data_filename(name, es)
    lp = np.load(filename)
    return (int)(lp[0]), (int)(lp[1])

def save_conf_matrix(matrix, prefix, es):
    name = prefix + constants.matrix_suffix
    plot_conf_matrix(matrix, dimex.phonemes, name, es)
    filename = constants.data_filename(name, es)
    np.save(filename, matrix)


def save_learned_data(pairs, suffix, es, fold):
    random.shuffle(pairs)
    data = [p[0] for p in pairs]
    filename = constants.learned_data_filename(suffix, es, fold)
    np.save(filename, data)

    labels = [p[1] for p in pairs]
    filename = constants.learned_labels_filename(suffix, es, fold)
    np.save(filename, labels)


def recognition_on_ciempiess(data, es, fold):
    agreed = []
    nnet = []
    amsys = []
    original = []
    lds = dimex.LearnedDataSet(es, fold)
    distrib = lds.get_distribution()
    maximum = lds.learning_maximum()
    for d in data:
        n = len(d.net_labels)
        for i in range(n):
            orig_mfcc = d.segments[i]
            mem_label = d.ams_labels[i]
            mem_mfcc = d.ams_segments[i]
            net_label = d.net_labels[i]
            net_mfcc = d.net_segments[i]
            if mem_label is None:
                mem_label = constants.n_labels
            
            if (mem_label == net_label) and (distrib[mem_label] < maximum):
                mfcc = (orig_mfcc + mem_mfcc)/2
                agreed.append((mfcc, mem_label))
                original.append((orig_mfcc, mem_label))
                nnet.append((net_mfcc, net_label))
                amsys.append((mem_mfcc, mem_label))
                distrib[mem_label] += 1
    save_learned_data(agreed, constants.agreed_suffix, es, fold)
    save_learned_data(original, constants.original_suffix, es, fold)
    save_learned_data(amsys, constants.amsystem_suffix, es, fold)
    save_learned_data(nnet, constants.nnetwork_suffix, es, fold)
    print('Updated distribution: ', end=' ')
    print(distrib)

def list_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def ams_process_samples_batch(samples, ams: AssociativeMemorySystem,
    minimum, maximum, decode=False):
    for sample in samples:
        features = msize_features(sample.features, ams.m, minimum, maximum)
        if not decode:
            sample.ams_labels = [ams.recall(f)[0] for f in features]
        else:
            labels = []
            recalls = []
            for f in features:
                label, recall = ams.recall(f)
                labels.append(label)
                recall = rsize_recall(recall, ams.m, minimum, maximum)
                recalls.append(recall)
            sample.ams_labels = labels
            sample.ams_features = recalls
    return samples

def ams_process_samples(samples, ams, minimum, maximum, decode=False):
    chunk_size = 10
    chunks = list_chunks(samples, chunk_size)
    processed = []
    segment_sizes = []
    for p in Parallel(n_jobs=constants.n_jobs, verbose=50)(
        delayed(ams_process_samples_batch)(chunk, ams, minimum, maximum, decode) \
                for chunk in chunks):
        processed.append(p)
    return [sample for chunk in processed for sample in chunk]

def learn_new_data(domain, mem_size, fill_percent, es):
    model_prefix = constants.model_name(es)
    confusion_matrix = np.zeros(
        (constants.n_folds, constants.n_labels, constants.n_labels), dtype=float)
    for fold in range(constants.n_folds):
        print(f'Learning new data at stage {es.stage}')
        suffix = constants.filling_suffix
        filling_features_filename = constants.features_name(es) + suffix        
        filling_features_filename = constants.data_filename(filling_features_filename, es, fold)
        filling_labels_filename = constants.labels_name(es) + suffix        
        filling_labels_filename = constants.data_filename(filling_labels_filename, es, fold)
        filling_features = np.load(filling_features_filename)
        filling_labels = np.load(filling_labels_filename)        
        # Apply reduction to given percent of filling data.
        n = int(len(filling_labels)*fill_percent/100.0)
        filling_features = filling_features[:n]
        filling_labels = filling_labels[:n]

        maximum = filling_features.max()
        minimum = filling_features.min()
        filling_features = msize_features(filling_features, mem_size, minimum, maximum)

        suffix = constants.testing_suffix
        testing_features_filename = constants.features_name(es) + suffix
        testing_features_filename = constants.data_filename(testing_features_filename, es, fold)
        testing_labels_filename = constants.labels_name(es) + suffix
        testing_labels_filename = constants.data_filename(testing_labels_filename, es, fold)
        testing_features = np.load(testing_features_filename)
        testing_labels = np.load(testing_labels_filename)
        testing_features = msize_features(testing_features, mem_size, minimum, maximum)

        ams = AssociativeMemorySystem(constants.all_labels, domain, mem_size,
            es.tolerance, es.sigma, es.iota, es.kappa)
        for label, features in zip(filling_labels, filling_features):
            ams.register(label,features)
        for label, features in zip(testing_labels, testing_features):
            l, _ = ams.recall(features)
            if not (l is None):
                confusion_matrix[fold, label, l] += 1
        nds = ciempiess.NextDataSet(es)
        new_data = nds.get_data()
        new_data = recnet.process_samples(new_data, model_prefix, es, fold, decode=True)
        new_data = ams_process_samples(new_data, ams, minimum, maximum, decode=True)
        new_data = recnet.reprocess_samples(new_data, model_prefix, es, fold)
        recognition_on_ciempiess(new_data, es, fold)
    confusion_matrix = np.sum(confusion_matrix, axis=0)
    confusion_matrix = confusion_matrix / np.sum(confusion_matrix, axis=1)[:, np.newaxis]
    save_conf_matrix(confusion_matrix, constants.memories_name(es), es)
    print(f'Learning at stage {es.stage} completed!')

def levenshtein(s, t):
    # create two work vectors of integer distances
    v0 = np.zeros((len(t)+1), dtype=int)
    v1 = np.zeros((len(t)+1), dtype=int)
    # initialize v0 (the previous row of distances)
    # that distance is the number of characters to append to  s to make t.
    for i in range(len(t)+1):
        v0[i] = i
    for i in range(len(s)):
        # calculate v1 (current row distances) from the previous row v0
        v1[0] = i + 1
        # use formula to fill in the rest of the row
        for j in range(len(t)):
            deletionCost = v0[j + 1] + 1
            insertionCost = v1[j] + 1
            if s[i] == t[j]:
                substitutionCost = v0[j]
            else:
                substitutionCost = v0[j] + 1
            v1[j + 1] = min(deletionCost, insertionCost, substitutionCost)
        # copy v1 (current row) to v0 (previous row) for next iteration
        # since data in v1 is always invalidated, a swap without copy could be more efficient
        v0, v1 = v1, v0
    # after the last swap, the results of v1 are now in v0
    return v0[len(t)]

def save_recognitions(samples, dp, fold, es):
    filename = constants.recog_filename(constants.recognition_prefix, es, fold)
    with open(filename, 'w') as file:
        writer = csv.writer(file)
        writer.writerow([
            'Id', 'Text', 'Correct', 'CorrSize', 'Network',
            'NetSize', 'Memories', 'MemSize', 'Cor2Net', 'Cor2Mem', 'Mem2Net',
            'NetworkND', 'NetSizeND', 'MemoriesND', 'MemSizeND',
            'Cor2NetND', 'Cor2MemND', 'Mem2NetND'])
        for sample in samples:
            # sample is a Tagged Audio
            correct_phns = dp.get_phonemes(sample.labels)
            corr_size = len(sample.labels)
            nnet_phns = dp.get_phonemes(sample.net_labels)
            nnet_size = len(sample.net_labels)
            ams_phns = dp.get_phonemes(sample.ams_labels)
            ams_size = len(sample.ams_labels)
            cor_net = levenshtein(sample.labels, sample.net_labels)
            cor_mem = levenshtein(sample.labels, sample.ams_labels)
            net_mem = levenshtein(sample.ams_labels, sample.net_labels)
            net_labels = remove_duplicates(sample.net_labels)
            ams_labels = remove_duplicates(sample.ams_labels)
            nnet_phns_nodups = dp.get_phonemes(net_labels)
            nnet_size_nodups = len(net_labels)
            ams_phns_nodups = dp.get_phonemes(ams_labels)
            ams_size_nodups = len(ams_labels)
            cor_net_nodups = levenshtein(sample.labels, net_labels)
            cor_mem_nodups = levenshtein(sample.labels, ams_labels)
            net_mem_nodups = levenshtein(ams_labels, net_labels)

            row = [
                sample.id, sample.text, correct_phns, corr_size,
                nnet_phns, nnet_size, ams_phns, ams_size]
            row += [cor_net, cor_mem, net_mem]
            row += [
                nnet_phns_nodups, nnet_size_nodups,
                ams_phns_nodups, ams_size_nodups]
            row += [cor_net_nodups, cor_mem_nodups, net_mem_nodups]
            writer.writerow(row)

def remove_duplicates(labels):
    nodups = []
    previous = -1
    for label in labels:
        if label == previous:
            continue
        else:
            nodups.append(label)
            previous = label
    return nodups

def process_sample(sample, ams, dp, mem_size, minimum, maximum):
    sample.net_labels = dp.process(sample.net_labels)
    ams_labels = []
    for f in sample.features:
        features = msize_features(f, mem_size, minimum, maximum)
        label, _ = ams.recall(features)
        ams_labels.append(label)
    sample.ams_labels = dp.process(ams_labels)
    return sample


def test_recognition(domain, mem_size, filling_percent, es: constants.ExperimentSettings):
    model_prefix = constants.model_name(es)
    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    print(f'{now} Loading samples...', end=" ")
    ds = dimex.Sampler()
    print('done!')
    for fold in range(constants.n_folds):
        now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        print(f'{now} Reading filling corpus...', end=" ")
        suffix = constants.filling_suffix
        filling_features_filename = constants.features_name(es) + suffix
        filling_features_filename = constants.data_filename(filling_features_filename, es, fold)
        filling_labels_filename = constants.labels_name(es) + suffix
        filling_labels_filename = constants.data_filename(filling_labels_filename, es, fold)

        filling_features = np.load(filling_features_filename)
        filling_labels = np.load(filling_labels_filename)
        maximum = filling_features.max()
        minimum = filling_features.min()
        # Apply reduction to given percent of filling data.
        n = int(len(filling_labels)*filling_percent/100.0)
        filling_features = filling_features[:n]
        filling_labels = filling_labels[:n]
        filling_features = msize_features(filling_features, mem_size, minimum, maximum)
        print('done!')

        dp = dimex.PostProcessor()
        now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        print(f'{now} Filling memories...', end=" ")
        ams = AssociativeMemorySystem(constants.all_labels,domain,mem_size,
            es.tolerance, es.sigma, es.iota, es.kappa)
        for label, features in zip(filling_labels, filling_features):
            ams.register(label,features)
        print('done!')
        now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        print(f'{now} Getting samples...', end=" ")
        samples = ds.get_samples(fold)
        print('done!')
        samples = recnet.process_samples(samples, model_prefix, es, fold)
        now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        print(f'{now} Processing samples with memories...', end=" ")
        samples = Parallel(n_jobs=constants.n_jobs, verbose=50)(
            delayed(process_sample)(sample, ams, dp, mem_size, minimum, maximum) \
                for sample in samples)
        print('done!')
        now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        print(f'{now} Saving recognitions...', end=" ")
        save_recognitions(samples, dp, fold, es)
        print('done!')
    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    print(f'{now} Recognition at stage {es.stage} completed!')


##############################################################################
# Main section

def create_and_train_classifiers(es):
    model_prefix = constants.model_name(es)
    stats_prefix = model_prefix + constants.classifier_suffix
    history, conf_matrix = recnet.train_classifier(model_prefix, es)
    save_history(history, stats_prefix, es)
    save_conf_matrix(conf_matrix, stats_prefix, es)
 
def produce_features_from_data(es):
    model_prefix = constants.model_name(es)
    features_prefix = constants.features_name(es)
    labels_prefix = constants.labels_name(es)
    data_prefix = constants.data_name(es)
    recnet.obtain_features(
        model_prefix, features_prefix, labels_prefix, data_prefix, es)

def create_and_train_autoencoder(es):
    model_prefix = constants.model_name(es)
    stats_prefix = model_prefix + constants.decoder_suffix
    features_prefix = constants.features_name(es)
    data_prefix = constants.data_name(es)
    history = recnet.train_decoder(model_prefix, features_prefix, data_prefix, es)
    save_history(history, stats_prefix, es)

def characterize_features(es):
    """ Produces a graph of features averages and standard deviations.
    """
    features_prefix = constants.features_name(es)
    tf_filename = features_prefix + constants.testing_suffix
    labels_prefix = constants.labels_name(es)
    tl_filename = labels_prefix + constants.testing_suffix
    features = get_all_data(tf_filename, es)
    labels = get_all_data(tl_filename, es)
    d = {}
    for i in constants.all_labels:
        d[i] = []
    for (i, feats) in zip(labels, features):
        # Separates features per label.
        d[i].append(feats)
    means = {}
    stdevs = {}
    for i in constants.all_labels:
        # The list of features becomes a matrix
        d[i] = np.array(d[i])
        means[i] = np.mean(d[i], axis=0)
        stdevs[i] = np.std(d[i], axis=0)
    plot_features_graph(constants.domain, means, stdevs, es)
    
def run_evaluation(es):
    best_memory_size = test_memories(constants.domain, es)
    print(f'Best memory size: {best_memory_size}')
    best_filling_percent = test_recalling(constants.domain, best_memory_size, es)
    save_learn_params(best_memory_size, best_filling_percent, es)

def extend_data(es):
    mem_size, fill_percent = load_learn_params(es)
    print(f'Learning data with memory size of {mem_size} and fill of {fill_percent}%')
    learn_new_data(constants.domain, mem_size, fill_percent, es)
 
def recognize_sequences(es):
    mem_size, fill_percent = load_learn_params(es)
    print(f'Recognizing sequences with memory size {mem_size} filled with {fill_percent}% of corpus')
    test_recognition(constants.domain, mem_size, fill_percent, es)


if __name__== "__main__" :
    args = docopt(__doc__)

    # Processing language.
    lang = 'en'
    if args['es']:
        lang = 'es'
        es = gettext.translation('eam', localedir='locale', languages=['es'])
        es.install()

    # Processing stage. 
    stage = 0
    if args['<stage>']:
        try:
            stage = int(args['<stage>'])
            if stage < 0:
                raise Exception('Negative number.')
        except:
            constants.print_error('<stage> must be a positive integer.')
            exit(1)

    # Processing learned data.
    learned = 0
    if args['--learned']:
        try:
            learned = int(args['--learned'])
            if (learned < 0) or (learned >= constants.learned_data_groups):
                raise Exception('Number out of range.')
        except:
            constants.print_error('<learned_data> must be a positive integer.')
            exit(1)

    # Processing use of extended data as testing data for memory
    extended = args['-x']

    # Processing tolerance.
    tolerance = 0
    if args['--tolerance']:
        try:
            tolerance = int(args['--tolerance'])
            if (tolerance < 0) or (tolerance > constants.domain):
                raise Exception('Number out of range.')
        except:
            constants.print_error('<tolerance> must be a positive integer.')
            exit(1)

    # Processing sigma.
    sigma = 0.25
    if args['--sigma']:
        try:
            sigma = float(args['--sigma'])
            if (sigma < 0):
                raise Exception('<sigma> must be positive.')
        except:
            constants.print_error('<sigma> must be a positive number.')
            exit(1)

    # Processing sigma.
    iota = 0.0
    if args['--iota']:
        try:
            iota = float(args['--iota'])
            if (iota < 0):
                raise Exception('<iota> must be positive.')
        except:
            constants.print_error('<iota> must be a positive number.')
            exit(1)

    # Processing sigma.
    kappa = 0.0
    if args['--kappa']:
        try:
            kappa = float(args['--kappa'])
            if (kappa < 0):
                raise Exception('<kappa> must be positive.')
        except:
            constants.print_error('<kappa> must be a positive number.')
            exit(1)

    # Processing runpath.
    constants.run_path = args['--runpath']
    exp_set = constants.ExperimentSettings(
        stage, learned, extended, tolerance, sigma, iota, kappa)
    print(f'Working directory: {constants.run_path}')
    print(f'Experimental settings: {exp_set}')

    # PROCESSING OF MAIN OPTIONS.

    if args['-n']:
        create_and_train_classifiers(exp_set)
    elif args['-f']:
        produce_features_from_data(exp_set)
    elif args['-a']:
        create_and_train_autoencoder(exp_set)
    elif args['-c']:
        characterize_features(exp_set)
    elif args['-e']:
        run_evaluation(exp_set)
    elif args['-i']:
        extend_data(exp_set)
    elif args['-r']:
        recognize_sequences(exp_set)
