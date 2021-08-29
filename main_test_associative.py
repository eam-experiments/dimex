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

import csv
import sys
import gc
import argparse
import gettext

import numpy as np
from joblib import Parallel, delayed
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
import json
from numpy.core.defchararray import array
import seaborn

from tensorflow.python.framework.tensor_shape import unknown_shape
from tensorflow.python.ops.gen_parsing_ops import parse_single_sequence_example_eager_fallback

from associative import AssociativeMemory, AssociativeMemorySystem
import ciempiess
import constants
import dimex
import recnet

# Translation
gettext.install('ame', localedir=None, codeset=None, names=None)

def print_error(*s):
    print('Error:', *s, file = sys.stderr)


def plot_pre_graph (pre_mean, rec_mean, ent_mean, pre_std, rec_std, ent_std, \
    tag = '', xlabels = constants.memory_sizes, xtitle = None, \
        ytitle = None, action=None, occlusion = None, bars_type = None, tolerance = 0):

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

    plt.errorbar(x, pre_mean, fmt='r-o', yerr=pre_std, label=_('Precision'))
    plt.errorbar(x, rec_mean, fmt='b--s', yerr=rec_std, label=_('Recall'))

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
    graph_filename = constants.picture_filename(s, action, occlusion, bars_type, tolerance)
    plt.savefig(graph_filename, dpi=600)


def plot_size_graph (response_size, size_stdev, action=None, tolerance=0):
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

    graph_filename = constants.picture_filename('graph_size_MEAN' + _('-english'), action, tolerance=tolerance)
    plt.savefig(graph_filename, dpi=600)


def plot_behs_graph(no_response, no_correct, no_chosen, correct, action=None, tolerance=0):

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

    graph_filename = constants.picture_filename('graph_behaviours_MEAN' + _('-english'), action, tolerance=tolerance)
    plt.savefig(graph_filename, dpi=600)


def plot_features_graph(domain, means, stdevs, experiment, occlusion = None, bars_type = None):
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
        filename = constants.features_name(experiment, occlusion, bars_type) + '-' + str(i) + _('-english')
        plt.savefig(constants.picture_filename(filename), dpi=600)


def plot_conf_matrix(matrix, tags, prefix):
    plt.clf()
    plt.figure(figsize=(6.4,4.8))
    seaborn.heatmap(matrix, xticklabels=tags, yticklabels=tags, vmin = 0.0, vmax=1.0, annot=False, cmap='Blues')
    plt.xlabel(_('Prediction'))
    plt.ylabel(_('Label'))
    filename = constants.picture_filename(prefix)
    plt.savefig(filename, dpi=600)



def get_label(memories, entropies = None):
    # Random selection
    if entropies is None:
        i = random.atddrange(len(memories))
        return memories[i]
    else:
        i = memories[0] 
        entropy = entropies[i]

        for j in memories[1:]:
            if entropy > entropies[j]:
                i = j
                entropy = entropies[i]
    return i


def msize_features(features, msize, min_value, max_value):
    return np.round((msize-1)*(features-min_value) / (max_value-min_value)).astype(np.int16)


def conf_sum(cms, t):
    return np.sum([cms[i][t] for i in range(len(cms))])


def get_ams_results(midx, msize, domain, lpm, trf, tef, trl, tel, tolerance=0):
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
    n_mems = int(n_labels/lpm)

    measures = np.zeros(constants.n_measures, dtype=np.float64)
    entropy = np.zeros(n_mems, dtype=np.float64)
    behaviour = np.zeros(constants.n_behaviours, dtype=np.float64)

    # Confusion matrix for calculating precision and recall per memory.
    cms = np.zeros((n_mems, 2, 2), dtype='int')
    TP = (0,0)
    FP = (0,1)
    FN = (1,0)
    TN = (1,1)

    # Create the required associative memories.
    ams = dict.fromkeys(range(n_mems))
    for m in ams:
        ams[m] = AssociativeMemory(domain, msize, tolerance)

    # Registration
    for features, label in zip(trf_rounded, trl):
        m = int(label/lpm)
        ams[m].register(features)

    # Calculate entropies
    for m in ams:
        entropy[m] = ams[m].entropy

    # Recognition
    response_size = 0

    for features, label in zip(tef_rounded, tel):
        correct = int(label/lpm)

        memories = []
        for k in ams:
            recognized = ams[k].recognize(features)
            if recognized:
                memories.append(k)

            # For calculation of per memory precision and recall
            if (k == correct) and recognized:
                cms[k][TP] += 1
            elif k == correct:
                cms[k][FN] += 1
            elif recognized:
                cms[k][FP] += 1
            else:
                cms[k][TN] += 1
 
        response_size += len(memories)
        if len(memories) == 0:
            # Register empty case
            behaviour[constants.no_response_idx] += 1
        elif not (correct in memories):
            behaviour[constants.no_correct_response_idx] += 1
        else:
            l = get_label(memories, entropy)
            if l != correct:
                behaviour[constants.no_correct_chosen_idx] += 1
            else:
                behaviour[constants.correct_response_idx] += 1

    behaviour[constants.mean_responses_idx] = response_size /float(len(tef_rounded))
    all_responses = len(tef_rounded) - behaviour[constants.no_response_idx]
    all_precision = (behaviour[constants.correct_response_idx])/float(all_responses)
    all_recall = (behaviour[constants.correct_response_idx])/float(len(tef_rounded))

    behaviour[constants.precision_idx] = all_precision
    behaviour[constants.recall_idx] = all_recall

    positives = conf_sum(cms, TP) + conf_sum(cms, FP)
    details = True
    if positives == 0:
        print('No memory responded')
        measures[constants.precision_idx] = 1.0
        details = False
    else:
        measures[constants.precision_idx] = conf_sum(cms, TP)/positives
    measures[constants.recall_idx] = conf_sum(cms, TP)/(conf_sum(cms, TP) + conf_sum(cms, FN))
    measures[constants.entropy_avg_idx] = np.mean(entropy)
    measures[constants.entropy_std_idx] = np.std(entropy)
 
    if details:
        for i in range(n_mems):
            positives = cms[i][TP] + cms[i][FP]
            if positives == 0:
                print(f'Memory {i} of size {msize} in run did not respond.')
    return (midx, measures, behaviour)
    

def test_memories(domain, experiment, tolerance=0):
    average_entropy = []
    stdev_entropy = []
    precision = []
    recall = []
    all_precision = []
    all_recall = []

    no_response = []
    no_correct_response = []
    no_correct_chosen = []
    correct_chosen = []
    total_responses = []

    labels_x_memory = constants.labels_per_memory[experiment]
    n_memories = int(constants.n_labels/labels_x_memory)

    for i in range(constants.training_stages):
        gc.collect()

        suffix = constants.filling_suffix
        training_features_filename = constants.features_name(experiment) + suffix        
        training_features_filename = constants.data_filename(training_features_filename, i)
        training_labels_filename = constants.labels_name(experiment) + suffix        
        training_labels_filename = constants.data_filename(training_labels_filename, i)

        suffix = constants.testing_suffix
        testing_features_filename = constants.features_name(experiment) + suffix        
        testing_features_filename = constants.data_filename(testing_features_filename, i)
        testing_labels_filename = constants.labels_name(experiment) + suffix        
        testing_labels_filename = constants.data_filename(testing_labels_filename, i)

        training_features = np.load(training_features_filename)
        training_labels = np.load(training_labels_filename)
        testing_features = np.load(testing_features_filename)
        testing_labels = np.load(testing_labels_filename)

        measures_per_size = np.zeros((len(constants.memory_sizes), constants.n_measures), dtype=np.float64)
        behaviours = np.zeros((len(constants.memory_sizes), constants.n_behaviours))

        print('Train the different co-domain memories -- NxM: ',experiment,' run: ',i)
        # Processes running in parallel.
        list_measures = Parallel(n_jobs=constants.n_jobs, verbose=50)(
            delayed(get_ams_results)(midx, msize, domain, labels_x_memory, \
                training_features, testing_features, training_labels, testing_labels, tolerance) \
                    for midx, msize in enumerate(constants.memory_sizes))

        for j, measures, behaviour in list_measures:
            measures_per_size[j, :] = measures
            behaviours[j, :] = behaviour
        

        ###################################################################3##
        # Measures by memory size

        # Average entropy among al digits.
        average_entropy.append(measures_per_size[:,constants.entropy_avg_idx])
        stdev_entropy.append(measures_per_size[:,constants.entropy_std_idx])

        # Average precision and recall as percentage
        precision.append(measures_per_size[:,constants.precision_idx]*100)
        recall.append(measures_per_size[:,constants.recall_idx]*100)

        all_precision.append(behaviours[:, constants.precision_idx] * 100)
        all_recall.append(behaviours[:, constants.recall_idx] * 100)
        no_response.append(behaviours[:, constants.no_response_idx])
        no_correct_response.append(behaviours[:, constants.no_correct_response_idx])
        no_correct_chosen.append(behaviours[:, constants.no_correct_chosen_idx])
        correct_chosen.append(behaviours[:, constants.correct_response_idx])
        total_responses.append(behaviours[:, constants.mean_responses_idx])

    # Every row is training stage, and every column is a memory size.
    average_entropy=np.array(average_entropy)
    stdev_entropy=np.array(stdev_entropy)
    precision = np.array(precision)
    recall=np.array(recall)
    all_precision = np.array(all_precision)
    all_recall = np.array(all_recall)

    average_precision = np.mean(precision, axis=0)
    stdev_precision = np.std(precision, axis=0)
    average_recall = np.mean(recall, axis=0)
    stdev_recall = np.std(recall, axis=0)

    all_precision_average = []
    all_precision_stdev = []
    all_recall_average = []
    all_recall_stdev = []

    main_average_entropy=[]
    main_stdev_entropy=[]

    no_response = np.array(no_response)
    no_correct_response = np.array(no_correct_response)
    no_correct_chosen = np.array(no_correct_chosen)
    correct_chosen = np.array(correct_chosen)
    total_responses = np.array(total_responses)

    main_no_response = []
    main_no_correct_response = []
    main_no_correct_chosen = []
    main_correct_chosen = []
    main_total_responses = []
    main_total_responses_stdev = []


    for i in range(len(constants.memory_sizes)):
        main_average_entropy.append( average_entropy[:,i].mean() )
        main_stdev_entropy.append( stdev_entropy[:,i].mean() )

        all_precision_average.append(all_precision[:, i].mean())
        all_precision_stdev.append(all_precision[:, i].std())
        all_recall_average.append(all_recall[:, i].mean())
        all_recall_stdev.append(all_recall[:, i].std())

        main_no_response.append(no_response[:, i].mean())
        main_no_correct_response.append(no_correct_response[:, i].mean())
        main_no_correct_chosen.append(no_correct_chosen[:, i].mean())
        main_correct_chosen.append(correct_chosen[:, i].mean())
        main_total_responses.append(total_responses[:, i].mean())
        main_total_responses_stdev.append(total_responses[:, i].std())

    main_behaviours = [main_no_response, main_no_correct_response, \
        main_no_correct_chosen, main_correct_chosen, main_total_responses]

    np.savetxt(constants.csv_filename('memory_average_precision-{0}'.format(experiment),
        tolerance=tolerance), precision, delimiter=',')
    np.savetxt(constants.csv_filename('memory_average_recall-{0}'.format(experiment),
        tolerance=tolerance), recall, delimiter=',')
    np.savetxt(constants.csv_filename('memory_average_entropy-{0}'.format(experiment),
        tolerance=tolerance), average_entropy, delimiter=',')

    np.savetxt(constants.csv_filename('memory_stdev_precision-{0}'.format(experiment),
        tolerance=tolerance), stdev_precision, delimiter=',')
    np.savetxt(constants.csv_filename('memory_stdev_recall-{0}'.format(experiment),
        tolerance=tolerance), stdev_recall, delimiter=',')
    np.savetxt(constants.csv_filename('memory_stdev_entropy-{0}'.format(experiment),
        tolerance=tolerance), stdev_entropy, delimiter=',')

    np.savetxt(constants.csv_filename('all_precision-{0}'.format(experiment),
        tolerance=tolerance), all_precision, delimiter=',')
    np.savetxt(constants.csv_filename('all_recall-{0}'.format(experiment),
        tolerance=tolerance), all_recall, delimiter=',')
    np.savetxt(constants.csv_filename('main_behaviours-{0}'.format(experiment),
        tolerance=tolerance), main_behaviours, delimiter=',')

    plot_pre_graph(average_precision, average_recall, main_average_entropy,\
        stdev_precision, stdev_recall, main_stdev_entropy, action=experiment, \
        tolerance=tolerance)
    plot_pre_graph(all_precision_average, all_recall_average, \
        main_average_entropy, all_precision_stdev, all_recall_stdev,\
            main_stdev_entropy, 'overall', action=experiment, tolerance=tolerance)
    plot_size_graph(main_total_responses, main_total_responses_stdev, action=experiment, tolerance=tolerance)
    plot_behs_graph(main_no_response, main_no_correct_response, main_no_correct_chosen,\
        main_correct_chosen, action=experiment, tolerance=tolerance)
    print(f'Experiment {experiment} completed!')


def get_recalls(ams, msize, domain, min_value, max_value, trf, trl, tef, tel, idx, fill):

    n_mems = constants.n_labels

    # To store precisión, recall, and entropies
    measures = np.zeros(constants.n_measures, dtype=np.float64)
    entropy = np.zeros(n_mems, dtype=np.float64)

    # Confusion matrix for calculating precision and recall per memory.
    cms = np.zeros((n_mems, 2, 2))
    TP = (0,0)
    FP = (0,1)
    FN = (1,0)
    TN = (1,1)

    # Confusion matrix for calculating overall precision and recall.
    cmatrix = np.zeros((2,2))

    # Registration
    for features, label in zip(trf, trl):
        ams[label].register(features)

    # Calculate entropies
    for j in ams:
        entropy[j] = ams[j].entropy

    # The list of recalls recovered from memory.
    all_recalls = []

    # Total number of differences between features and memories.
    mismatches = 0

    # Recover memories
    for n, features, label in zip(range(len(tef)), tef, tel):
        memories = []
        recalls ={}

        # How much it was needed for the right memory to recognize
        # the features.
        mismatches += ams[label].mismatches(features)

        for k in ams:
            recall, recognized = ams[k].recall(features)

            # For calculation of per memory precision and recall
            if (k == label) and recognized:
                cms[k][TP] += 1
            elif k == label:
                cms[k][FN] += 1
            elif recognized:
                cms[k][FP] += 1
            else:
                cms[k][TN] += 1

            if recognized:
                memories.append(k)
                recalls[k] = recall

        if (len(memories) == 0):
            # Register empty case
            undefined = np.full(domain, ams[0].undefined)
            all_recalls.append((n, label, undefined))
            cmatrix[FN] += 1
        else:
            l = get_label(memories, entropy)
            features = recalls[l]*(max_value-min_value)*1.0/(msize-1) + min_value
            all_recalls.append((n, label, features))

            if l == label:
                cmatrix[TP] += 1
            else:
                cmatrix[FP] += 1

    positives = conf_sum(cms, TP) + conf_sum(cms, FP)
    details = True
    if positives == 0:
        print('No memory responded')
        measures[constants.precision_idx] = 1.0
        details = False
    else:
        measures[constants.precision_idx] = conf_sum(cms, TP)/positives
    measures[constants.recall_idx] = conf_sum(cms, TP)/(conf_sum(cms, TP) + conf_sum(cms, FN))
    measures[constants.entropy_avg_idx] = np.mean(entropy)
    measures[constants.entropy_std_idx] = np.std(entropy)
 
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
    total_recall = cmatrix[TP] / len(tef)
    mismatches /= len(tel)

    return all_recalls, measures, total_precision, total_recall, mismatches
    

def test_recalling_fold(n_memories, mem_size, domain, fold, experiment, occlusion = None, bars_type = None, tolerance = 0):
    # Create the required associative memories.
    ams = dict.fromkeys(range(n_memories))
    for j in ams:
        ams[j] = AssociativeMemory(domain, mem_size, tolerance)

    suffix = constants.filling_suffix
    filling_features_filename = constants.features_name(experiment) + suffix        
    filling_features_filename = constants.data_filename(filling_features_filename, fold)
    filling_labels_filename = constants.labels_name(experiment) + suffix        
    filling_labels_filename = constants.data_filename(filling_labels_filename, fold)

    suffix = constants.testing_suffix
    testing_features_filename = constants.features_name(experiment, occlusion, bars_type) + suffix        
    testing_features_filename = constants.data_filename(testing_features_filename, fold)
    testing_labels_filename = constants.labels_name(experiment) + suffix        
    testing_labels_filename = constants.data_filename(testing_labels_filename, fold)

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

    stage_recalls = []
    stage_avg_entropies = []
    stage_std_entropies = []
    stage_precision = []
    stage_recall = []
    total_precisions = []
    total_recalls = []
    mismatches = []

    start = 0
    for end in steps:
        features = filling_features[start:end]
        labels = filling_labels[start:end]

        recalls, measures, step_precision, step_recall, mis_count = get_recalls(ams, mem_size, domain, \
            minimum, maximum, features, labels, testing_features, testing_labels, fold, end)

        # A list of tuples (position, label, features)
        stage_recalls += recalls

        # An array with average entropy per step.
        stage_avg_entropies.append(measures[constants.entropy_avg_idx])
        # An array with standard deviation of entropy per step.
        stage_std_entropies.append(measures[constants.entropy_std_idx])
        # An array with precision per step
        stage_precision.append(measures[constants.precision_idx])
        # An array with recall per memory, per step
        stage_recall.append(measures[constants.recall_idx])
        # Overall recalls and precisions per step
        total_recalls.append(step_recall)
        total_precisions.append(step_precision)
        mismatches.append(mis_count)
        start = end

    stage_avg_entropies = np.array(stage_avg_entropies)
    stage_std_entropies = np.array(stage_std_entropies)
    stage_precision = np.array(stage_precision)
    stage_recall = np.array(stage_recall)
    total_precisions = np.array(total_precisions)
    total_recalls = np.array(total_recalls)
    mismatches = np.array(mismatches)

    return fold, stage_recalls, stage_avg_entropies, stage_std_entropies, stage_precision, \
        stage_recall, total_precisions, total_recalls, mismatches


def test_recalling(domain, mem_size, experiment, occlusion = None, bars_type = None, tolerance = 0):
    n_memories = constants.n_labels
    memory_fills = constants.memory_fills
    training_stages = constants.training_stages
    # All recalls, per memory fill and fold.
    all_memories = {}
    # All entropies, precision, and recall, per fold, and fill.
    total_avg_entropies = np.zeros((training_stages, len(memory_fills)))
    total_std_entropies = np.zeros((training_stages, len(memory_fills)))
    total_precisions = np.zeros((training_stages, len(memory_fills)))
    total_recalls = np.zeros((training_stages, len(memory_fills)))
    sys_precisions = np.zeros((training_stages, len(memory_fills)))
    sys_recalls = np.zeros((training_stages, len(memory_fills)))
    total_mismatches = np.zeros((training_stages, len(memory_fills)))

    list_results = Parallel(n_jobs=constants.n_jobs, verbose=50)(
        delayed(test_recalling_fold)(n_memories, mem_size, domain, fold, experiment, occlusion, bars_type, tolerance) \
            for fold in range(constants.training_stages))

    for fold, memories, avg_entropy, std_entropy, precision, recall,\
        sys_precision, sys_recall, mismatches in list_results:

        all_memories[fold] = memories
        total_precisions[fold] = precision
        total_recalls[fold] = recall
        total_mismatches[fold] = mismatches
        total_avg_entropies[fold] = avg_entropy
        total_std_entropies[fold] = std_entropy
        sys_precisions[fold] = sys_precision
        sys_recalls[fold] = sys_recall

    for fold in all_memories:
        list_tups = all_memories[fold]
        tags = []
        memories = []
        for (idx, label, features) in list_tups:
            tags.append((idx, label))
            memories.append(np.array(features))
        tags = np.array(tags)
        memories = np.array(memories)
        memories_filename = constants.memories_name(experiment, occlusion, bars_type, tolerance)
        memories_filename = constants.data_filename(memories_filename, fold)
        np.save(memories_filename, memories)
        tags_filename = constants.labels_name(experiment) + constants.memory_suffix
        tags_filename = constants.data_filename(tags_filename, fold)
        np.save(tags_filename, tags)
    
    main_avrge_entropies = np.mean(total_avg_entropies,axis=0)
    main_stdev_entropies = np.mean(total_std_entropies, axis=0)
    main_avrge_mprecision = np.mean(total_precisions,axis=0)
    main_stdev_mprecision = np.std(total_precisions,axis=0)
    main_avrge_mrecall = np.mean(total_recalls,axis=0)
    main_stdev_mrecall = np.std(total_recalls,axis=0)
    main_avrge_sys_precision = np.mean(sys_precisions,axis=0)
    main_stdev_sys_precision = np.std(sys_precisions,axis=0)
    main_avrge_sys_recall = np.mean(sys_recalls,axis=0)
    main_stdev_sys_recall = np.std(sys_recalls,axis=0)
    
    
    np.savetxt(constants.csv_filename('main_average_precision',experiment, occlusion, bars_type, tolerance), \
        main_avrge_mprecision, delimiter=',')
    np.savetxt(constants.csv_filename('main_average_recall',experiment, occlusion, bars_type, tolerance), \
        main_avrge_mrecall, delimiter=',')
    np.savetxt(constants.csv_filename('main_average_entropy',experiment, occlusion, bars_type, tolerance), \
        main_avrge_entropies, delimiter=',')
    np.savetxt(constants.csv_filename('main_stdev_precision',experiment, occlusion, bars_type, tolerance), \
        main_stdev_mprecision, delimiter=',')
    np.savetxt(constants.csv_filename('main_stdev_recall',experiment, occlusion, bars_type, tolerance), \
        main_stdev_mrecall, delimiter=',')
    np.savetxt(constants.csv_filename('main_stdev_entropy',experiment, occlusion, bars_type, tolerance), \
        main_stdev_entropies, delimiter=',')
    np.savetxt(constants.csv_filename('main_total_recalls',experiment, occlusion, bars_type, tolerance), \
        main_avrge_sys_recall, delimiter=',')
    np.savetxt(constants.csv_filename('main_total_precision',experiment, occlusion, bars_type, tolerance), \
        main_avrge_sys_precision, delimiter=',')
    np.savetxt(constants.csv_filename('main_total_mismatches',experiment, occlusion, bars_type, tolerance), \
        total_mismatches, delimiter=',')

    plot_pre_graph(main_avrge_mprecision*100, main_avrge_mrecall*100, main_avrge_entropies,\
        main_stdev_mprecision*100, main_stdev_mrecall*100, main_stdev_entropies, 'recall-', \
            xlabels = constants.memory_fills, xtitle = _('Percentage of memory corpus'), action = experiment,
            occlusion = occlusion, bars_type = bars_type, tolerance = tolerance)
    plot_pre_graph(main_avrge_sys_precision*100, main_avrge_sys_recall*100, main_avrge_entropies, \
        main_stdev_sys_precision*100, main_stdev_sys_recall*100, main_stdev_entropies, 'total_recall-', \
            xlabels = constants.memory_fills, xtitle = _('Percentage of memory corpus'), action=experiment,
            occlusion = occlusion, bars_type = bars_type, tolerance = tolerance)

    print(f'Experiment {experiment} completed!')


def get_all_data(prefix):
    data = None

    for stage in range(constants.training_stages):
        filename = constants.data_filename(prefix, stage)
        if data is None:
            data = np.load(filename)
        else:
            newdata = np.load(filename)
            data = np.concatenate((data, newdata), axis=0)

    return data

def characterize_features(domain, experiment, occlusion = None, bars_type = None):
    """ Produces a graph of features averages and standard deviations.
    """
    features_prefix = constants.features_name(experiment, occlusion, bars_type)
    tf_filename = features_prefix + constants.testing_suffix

    labels_prefix = constants.labels_name(experiment)
    tl_filename = labels_prefix + constants.testing_suffix

    features = get_all_data(tf_filename)
    labels = get_all_data(tl_filename)

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

    plot_features_graph(domain, means, stdevs, experiment, occlusion, bars_type)
    

def save_history(history, prefix):
    """ Saves the stats of neural networks.

    Neural networks stats may come either as a History object, that includes
    a History.history dictionary with stats, or directly as a dictionary.
    """
    stats = {}
    stats['history'] = []
    for h in history:
        if type(h) is dict:
            stats['history'].append(h)
        else:
            stats['history'].append(h.history)

    with open(constants.json_filename(prefix), 'w') as outfile:
        json.dump(stats, outfile)


def save_conf_matrix(matrix, prefix):
    prefix += constants.matrix_suffix
    plot_conf_matrix(matrix, dimex.phonemes, prefix)
    file_name = constants.data_filename(prefix)
    np.save(file_name, matrix)


def lev(a, b, m):
    if m[len(a), len(b)] >= 0:
        return m[len(a),len(b)]
    elif len(a) == 0:
        return len(b)
    elif len(b) == 0:
        return len(a)
    elif a[0] == b[0]:
        d = lev(a[1:], b[1:], m)
        m[len(a), len(b)] = d
        return d
    else:
        deletion = lev(a[1:], b, m)
        insertion = lev(a, b[1:], m)
        replacement = lev(a[1:], b[1:], m)
        d = min(deletion, insertion, replacement) + 1
        m[len(a), len(b)] = d
        return d


def levenshtein(a: list, b: list):
    m = np.full((len(a)+1, len(b)+1), -1, dtype=int)
    d = lev(a, b, m)
    return d


def save_recognitions(samples: list, dp: dimex.PostProcessor, experiment: int,
    fold: int, tolerance: int, counter: int):
    filename = constants.recog_filename(constants.recognition_prefix, experiment,
        fold, tolerance, counter)
    with open(filename, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['Id', 'Text', 'Correct', 'CorrSize', 'Network',
            'NetSize', 'Memories', 'MemSize', 'Cor2Net', 'Cor2Mem', 'Net2Mem'])
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
            net_mem = levenshtein(sample.net_labels, sample.ams_labels)
            row = [sample.id, sample.text, correct_phns, corr_size,
                nnet_phns, nnet_size, ams_phns, ams_size]
            row += [cor_net, cor_mem, net_mem]
            writer.writerow(row)


def recognition_on_dimex(samples, experiment, fold, tolerance, counter):
    dp = dimex.PostProcessor()
    for sample in samples:
        sample.net_labels = dp.process(sample.net_labels)
        sample.ams_labels = dp.process(sample.ams_labels)
    save_recognitions(samples, dp, experiment, fold, tolerance, counter)


def recognition_on_ciempiess(ams, experiment, fold, tolerance, counter):
    pass


def ams_process_samples(samples, ams, minimum, maximum):
    n = 0
    print('Processing samples with memories.')
    for sample in samples:
        features = msize_features(sample.features, ams.m, minimum, maximum)
        sample.ams_labels = [ams.recall(f)[0] for f in features]
        n += 1
        if (n % 100) == 0:
            print(f' {n} ', end = '', flush=True)
        elif (n % 10) == 0:
            print('.', end = '', flush=True)
    return samples


def test_recognition(domain, mem_size, experiment, tolerance = 0):
    ds = dimex.LearnedDataSet(tolerance)
    (data, labels), counter = ds.get_data()
    total = len(labels)
    step = total / constants.training_stages
    training_size = int(total*constants.nn_training_percent)
    truly_training = int(training_size*recnet.truly_training_percentage)
    histories = []
    model_prefix = constants.model_name(experiment)
    features_prefix = constants.features_name(experiment)
    labels_prefix = constants.labels_name(experiment)

    for fold in range(constants.training_stages):
        i = int(fold*step)
        j = (i + training_size) % total
        training_data = recnet.get_data_in_range(data, i, j)
        training_labels = recnet.get_data_in_range(labels, i, j)
        validation_data = training_data[truly_training:]
        validation_labels = training_labels[truly_training:]
        training_data = training_data[:truly_training]
        training_labels = training_labels[:truly_training]
        filling_data = recnet.get_data_in_range(data, j, i)
        filling_labels = recnet.get_data_in_range(labels, j, i)
        filling_features, history = recnet.train_next_network(training_data, training_labels,
            validation_data, validation_labels, filling_data, filling_labels,
            model_prefix, fold, tolerance, counter)
        histories.append(history)
        maximum = filling_features.max()
        minimum = filling_features.min()
        filling_features = msize_features(filling_features, mem_size, minimum, maximum)

        ams = AssociativeMemorySystem(constants.all_labels, domain, mem_size)
        for label, features in zip(filling_labels, filling_features):
            ams.register(label,features)
        tds = dimex.TestingDataSet()
        testing_data = tds.get_data()
        testing_data = recnet.process_samples(testing_data, model_prefix, fold, tolerance, counter)
        testing_data = ams_process_samples(testing_data, ams, minimum, maximum)
        recognition_on_dimex(testing_data, experiment, fold, tolerance, counter)

        nds = ciempiess.NextDataSet(counter)
        new_data = nds = nds.get_data()
        new_data = recnet.process_samples(new_data, model_prefix, fold, tolerance, counter)
        new_data = ams_process_samples(new_data, ams, minimum, maximum)
        recognition_on_ciempiess(ams, experiment, fold, tolerance, counter)

    stats_prefix = constants.stats_name(experiment)
    save_history(histories, stats_prefix)
    print(f'Experiment {experiment} round {counter} completed!')

        

##############################################################################
# Main section

def main(action, occlusion = None, bar_type= None, tolerance = 0):
    """ Distributes work.

    The main function distributes work according to the options chosen in the
    command line.
    """

    if (action == constants.TRAIN_CLASSIFIER):
        # Trains the classifier.
        training_percentage = constants.nn_training_percent
        model_prefix = constants.model_name
        stats_prefix = constants.stats_model_name + constants.classifier_suffix
        history, conf_matrix = recnet.train_classifier(training_percentage, model_prefix, action)
        save_history(history, stats_prefix)
        save_conf_matrix(conf_matrix, stats_prefix)
    elif (action == constants.TRAIN_AUTOENCODER):
        # Trains the autoencoder.
        model_prefix = constants.model_name
        stats_prefix = constants.stats_model_name + constants.decoder_suffix
        history = recnet.train_decoder(model_prefix, action)
        save_history(history, stats_prefix)
    elif (action == constants.GET_FEATURES):
        # Generates features for the memories using the previously generated
        # neural networks.
        training_percentage = constants.nn_training_percent
        am_filling_percentage = constants.am_filling_percent
        model_prefix = constants.model_name
        features_prefix = constants.features_name(action)
        labels_prefix = constants.labels_name(action)
        data_prefix = constants.data_name

        history = recnet.obtain_features(model_prefix, features_prefix, labels_prefix, data_prefix,
            training_percentage, am_filling_percentage, action)
        save_history(history, features_prefix)
    elif action == constants.CHARACTERIZE:
        # Generates graphs of mean and standard distributions of feature values,
        # per digit class.
        characterize_features(constants.domain, action)
    elif action == constants.EXP_1 :
        # The domain size, equal to the size of the output layer of the network.
        test_memories(constants.domain, action, tolerance)
    elif (action == constants.EXP_3):
        test_recalling(constants.domain, constants.ideal_memory_size, action, tolerance=tolerance)
    elif (action == constants.EXP_4):
        test_recognition(constants.domain, constants.ideal_memory_size, action, tolerance=tolerance)
    elif (constants.EXP_5 <= action) and (action <= constants.EXP_10):
        # Generates features for the data sections using the previously generate
        # neural network, introducing (background color) occlusion.
        training_percentage = constants.nn_training_percent
        am_filling_percentage = constants.am_filling_percent
        model_prefix = constants.model_name
        features_prefix = constants.features_name(action, occlusion, bar_type)
        labels_prefix = constants.labels_name(action)
        data_prefix = constants.data_name

        history = recnet.obtain_features(model_prefix, features_prefix, labels_prefix, data_prefix,
            training_percentage, am_filling_percentage, action, occlusion, bar_type)
        save_history(history, features_prefix)
        characterize_features(constants.domain, action, occlusion, bar_type)
        test_recalling(constants.domain, constants.ideal_memory_size,
            action, occlusion, bar_type, tolerance)
        recnet.remember(action, occlusion, bar_type, tolerance)



if __name__== "__main__" :
    """ Argument parsing.
    
    Basically, there is a parameter for choosing language (-l), one
    to train and save the neural networks (-n), one to create and save the features
    for all data (-f), one to characterize the initial features (-c), and one to run
    the experiments (-e).
    """
    parser = argparse.ArgumentParser(description='Associative Memory Experimenter.')
    parser.add_argument('-l', nargs='?', dest='lang', choices=['en', 'es'], default='en',
                        help='choose between English (en) or Spanish (es) labels for graphs.')
    parser.add_argument('-t', nargs='?', dest='tolerance', type=int,
                        help='run the experiment with the tolerance given (only experiments 5 to 12).')
    
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('-o', nargs='?', dest='occlusion', type=float, 
                        help='run the experiment with a given proportion of occlusion (only experiments 5 to 12).')
    group.add_argument('-b', nargs='?', dest='bars_type', type=int, 
                        help='run the experiment with chosen bars type (only experiments 5 to 12).')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-n', action='store_const', const=constants.TRAIN_CLASSIFIER, dest='action',
                        help='train the classifier, separating NN and AM training data (Separate Data NN).')
    group.add_argument('-f', action='store_const', const=constants.GET_FEATURES, dest='action',
                        help='get data features using the separate data neural networks.')
    group.add_argument('-a', action='store_const', const=constants.TRAIN_AUTOENCODER, dest='action',
                        help='train the autoencoder using the features generated previously as data.')
    group.add_argument('-c', action='store_const', const=constants.CHARACTERIZE, dest='action',
                        help='characterize the features from partial data neural networks by class.')
    group.add_argument('-e', nargs='?', dest='nexp', type=int, 
                        help='run the experiment with that number, using separate data neural networks.')

    args = parser.parse_args()
    lang = args.lang
    occlusion = args.occlusion
    bars_type = args.bars_type
    tolerance = args.tolerance
    action = args.action
    nexp = args.nexp

    
    if lang == 'es':
        es = gettext.translation('ame', localedir='locale', languages=['es'])
        es.install()

    if not (occlusion is None):
        if (occlusion < 0) or (1 < occlusion):
            print_error("Occlusion needs to be a value between 0 and 1")
            exit(1)
        elif (nexp is None) or (nexp < constants.EXP_5) or (constants.EXP_8 < nexp):
            print_error("Occlusion is only valid for experiments 5 to 8")
            exit(2)
    elif not (bars_type is None):
        if (bars_type < 0) or (constants.N_BARS <= bars_type):
            print_error("Bar type must be a number between 0 and {0}"\
                        .format(constants.N_BARS-1))
            exit(1)
        elif (nexp is None) or (nexp < constants.EXP_9):
            print_error("Bar type is only valid for experiments 9 to {0}"\
                        .format(constants.MAX_EXPERIMENT))
            exit(2)


    if tolerance is None:
        tolerance = 0
    elif (tolerance < 0) or (constants.domain < tolerance):
            print_error("tolerance needs to be a value between 0 and {0}."
                .format(constants.domain))
            exit(3)

    if action is None:
        # An experiment was chosen
        if (nexp < constants.MIN_EXPERIMENT) or (constants.MAX_EXPERIMENT < nexp):
            print_error("There are only {1} experiments available, numbered consecutively from {0}."
                .format(constants.MIN_EXPERIMENT, constants.MAX_EXPERIMENT))
            exit(1)
        main(nexp, occlusion, bars_type, tolerance)
    else:
        # Other action was chosen
        main(action)

    
    
