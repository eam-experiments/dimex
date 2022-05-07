import argparse
import csv
import numpy as np
import os.path
import pickle
from python_speech_features import mfcc
import random
import scipy.io.wavfile as wav
import scipy.signal

import constants
import dimex

_ALL_IDS = 'ids.csv'
_TRAINING_IDS = 'training.csv'
_ALL_DATA_PREFIX = 'all'


_BALANCED = 0
_SEED = 1
_FULL = 2

# milliseconds
left_padding = 35
right_padding = 25
min_for_crop = constants.phn_duration + 10

def create_balanced_data(cut_point, in_prefix, out_prefix, convert=False):
   # Load original data
    labels_filename = constants.data_filename(in_prefix + constants.labels_suffix)
    features_filename = constants.data_filename(in_prefix + constants.data_suffix)
    labels = np.load(labels_filename)
    features = np.load(features_filename)

    if convert:
        labels = [dimex.phns_to_labels[label] for label in labels]

    data = [(features[i], labels[i]) for i in range(0, len(labels))]

    frequencies = np.zeros(constants.n_labels, dtype='int')
    for label in labels:
        frequencies[label] += 1

    # Reduce the number of instances of over-represented classes
    pairs = []
    for f, l in data:
        if frequencies[l] > cut_point:
            frequencies[l] -= 1
        else:
            pairs.append((f, l))
    random.shuffle(pairs)

    features = [p[0] for p in pairs]
    labels = [p[1] for p in pairs]
    labels_filename = constants.data_filename(out_prefix + constants.labels_suffix)
    features_filename = constants.data_filename(out_prefix + constants.data_suffix)
    np.save(features_filename, features)
    np.save(labels_filename, labels)


def get_mods_ids(file_name):
    mods_ids = []
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        # Skip the header.
        next(reader, None)
        for row in reader:
            mods_ids.append(tuple(row))
    return mods_ids


def create_data_and_labels(id_filename, prefix, crop_pad=True):
    mods_ids = get_mods_ids(id_filename)

    data = []
    labels = []
    counter = 0
    phonemes = np.zeros(constants.n_labels, dtype=int)
    zeros = np.zeros(constants.n_labels, dtype=int)
    shorts = np.zeros(constants.n_labels, dtype=int)
    durations = []
    for i in range(constants.n_labels):
        durations.append([])

    for mod, id in mods_ids:
        audio_filename = dimex.get_audio_filename(mod, id)
        try:
            sample_rate, signal = wav.read(audio_filename)
        except:
            constants.print_warning(f'{audio_filename} has problems')
        phnms_filename = dimex.get_phonemes_filename(mod, id)

        with open(phnms_filename) as file:
            reader=csv.reader(file, delimiter=' ')
            row = next(reader)
            # skip the headers
            while (len(row) == 0) or (row[0] != 'END'):
                row = next(reader, None)

            for row in reader:
                if len(row) < 3:
                    continue
                phn = row[2]
                if (phn == '.sil') or (phn == '.bn'):
                    continue
                label = dimex.phns_to_labels[phn]
                phonemes[label] += 1
                start = float(row[0])
                end = float(row[1])
                # Duration in milliseconds
                duration = end - start
                durations[label].append(duration)
                if crop_pad and (duration < min_for_crop):
                    start -= left_padding
                    start = 0 if start < 0 else start
                    end += right_padding
                ns = signal[int(sample_rate*start/1000):int(sample_rate*end/1000)]
                if len(ns) == 0:
                    end -= right_padding
                    ns = signal[int(sample_rate*start/1000):int(sample_rate*end/1000)]
                    if len(ns) == 0:
                        zeros[label] += 1
                        continue
                if sample_rate!=dimex.IDEAL_SRATE:
                    duration = end - start
                    resampling = int(dimex.IDEAL_SRATE*duration/1000)
                    ns = scipy.signal.resample(ns,resampling)
                features = mfcc(ns,dimex.IDEAL_SRATE,numcep=26)
                if (len(features) < constants.n_frames):
                    shorts[label] += 1
                if crop_pad:
                    feat_list = constants.padding_cropping(features, constants.n_frames)
                    data += feat_list
                    labels += [label for i in range(len(feat_list))]
                else:
                    data.append(features)
                    labels.append(label)
        counter += 1
        constants.print_counter(counter,100,10)
    print(f'Total phonemes: {phonemes}')
    print(f'Total zeros: {zeros}')
    print(f'Total shorts: {shorts}')

    data, labels = dimex.shuffle(data, labels)
    frequencies = np.zeros(constants.n_labels, dtype=int)
    for label in labels:
        frequencies[label] += 1
    print(f'Frequencies: {frequencies}')
    means = np.zeros(constants.n_labels, dtype=float)
    stdvs = np.zeros(constants.n_labels, dtype=float)
    for i in range(constants.n_labels):
        means[i] = np.mean(durations[i])
        stdvs[i] = np.std(durations[i])
    print(f'Durations mean: {means}')
    print(f'Duraitons stdevs: {stdvs}')
    filename = constants.data_filename(prefix + constants.labels_suffix)
    np.save(filename,labels)
    if crop_pad:
        filename = constants.data_filename(prefix + constants.data_suffix)
        np.save(filename,data)
    else:
        filename = constants.pickle_filename(prefix + constants.data_suffix)
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
    return frequencies
     

def create_learning_seeds():
    frequencies = create_data_and_labels(_TRAINING_IDS, constants.seed_data)
    median = np.median(frequencies)
    print(f'Suggested cutpoint: {median}')


def create_full_data():
    create_data_and_labels(_ALL_IDS, _ALL_DATA_PREFIX)

def create_learned_data(fold, tolerance):
    filename = constants.data_filename(_ALL_DATA_PREFIX + constants.data_suffix)
    if os.path.exists(filename):
        constants.print_error(f'File/directory {filename} exists! Nothing is done.')
        exit(1)
    lds = dimex.LearnedDataSet(fold)
    data, labels = lds.get_seed_data()
    learned_data, learned_labels, _ = \
        lds.get_learned_data(fold, tolerance, None)
    if not ((learned_data is None) or (learned_labels is None)):
        data = np.concatenate((data, learned_data), axis=0)
        labels = np.concatenate((labels, learned_labels), axis=0)
        data, labels = dimex.shuffle(data, labels)
    np.save(filename, data)
    filename = constants.data_filename(_ALL_DATA_PREFIX + constants.labels_suffix)
    np.save(filename, labels)


if __name__== "__main__" :
    parser = argparse.ArgumentParser(description='Initial datasets creator.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-f', action='store_const', const=_FULL, dest='action',
        help='creates the initial data set for experiments 1 and 3.')
    group.add_argument('-b', nargs='?', dest='bcutpoint', type=float, 
        help='balances initial data considering a maximum frequency per class.')
    group.add_argument('-s', action='store_const', const=_SEED, dest='action',
        help='creates the initial data set for the learning process.')
    group.add_argument('-v', nargs='?', dest='vcutpoint', type=float, 
        help='balances the initial data set for the learning process.')
    group.add_argument('-l', nargs='?', dest='fold', type=int, 
        help='creates the initial data set from the learning process, given fold and tolerance.')
    parser.add_argument('-t', nargs='?', dest='tolerance', type=float, 
        help='tolerance for the -l option.')

    args = parser.parse_args()
    action = args.action
    if action is None:
        if not (args.bcutpoint is None):
            cutpoint = args.bcutpoint
            create_balanced_data(cutpoint, _ALL_DATA_PREFIX, constants.balanced_data)
        elif not (args.vcutpoint is None):
            cutpoint = args.vcutpoint
            create_balanced_data(cutpoint, constants.seed_data,
                constants.learning_data_seed)
        elif not (args.fold is None):
            if args.tolerance is None:
                constants.print_warning(f'Assuming tolerance = 0.')
                tolerance = 0
            else:
                tolerance = args.tolerance
            fold = args.fold
            create_learned_data(fold, tolerance)
    elif action == _SEED:
        create_learning_seeds()
    elif action == _FULL:
        create_full_data()

 
