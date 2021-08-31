import argparse
import csv
import numpy as np
import pickle
from python_speech_features import mfcc
import random
import scipy.io.wavfile as wav
import scipy.signal

import constants
import dimex

_FEATURES_DIR = 'Features'
_ALL_IDS = 'ids.csv'
_TRAINING_IDS = 'training.csv'
_ALL_DATA_PREFIX = 'all'
_BALANCED_DATA_PREFIX = 'rand'


_BALANCED = 0
_SEED = 1
_FULL = 2


def create_balanced_data(cut_point, in_prefix, out_prefix, convert=False):
   # Load original data
    labels_filename = in_prefix + constants.labels_suffix
    features_filename = in_prefix + constants.data_suffix
    labels = np.load(labels_filename)
    features = np.load(features_filename, allow_pickle=True)

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
    np.save(out_prefix + constants.data_suffix, features)
    np.save(out_prefix + constants.labels_suffix, labels)


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
    for mod, id in mods_ids:
        audio_filename = dimex.get_audio_filename(mod, id)
        try:
            sample_rate, signal = wav.read(audio_filename)
        except:
            constants.print_warning(f'{audio_filename} has problemas')
        phnms_filename = dimex.get_phonemes_filename(mod, id)

        with open(phnms_filename) as file:
            reader=csv.reader(file, delimiter=' ')
            anterior='-'
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
                start = float(row[0])
                end = float(row[1])
                # Duration in milliseconds
                duration = float(end)-float(start)
                ns = signal[int(float(start)/1000 * sample_rate):int(float(end)/1000 * sample_rate)]
                if len(ns) == 0:
                    continue
                if sample_rate!=dimex.IDEAL_SRATE:
                    resampling = int(duration/1000*dimex.IDEAL_SRATE)
                    ns = scipy.signal.resample(ns,resampling)
                features = mfcc(ns,dimex.IDEAL_SRATE,numcep=26)
                if crop_pad:
                    features = constants.padding_cropping(features, constants.n_frames)
                label = dimex.phns_to_labels[phn]
                data.append(features)
                labels.append(label)
        counter += 1
        constants.print_counter(counter,100,10)    
    data, labels = dimex.shuffle(data, labels)
    filename = constants.data_filename(prefix + constants.labels_suffix)
    np.save(filename,labels)
    if crop_pad:
        filename = constants.data_filename(prefix + constants.data_suffix)
        np.save(filename,data)
    else:
        filename = constants.pickle_filename(prefix + constants.data_suffix)
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
 
    

def create_learning_seeds():
    create_data_and_labels(_TRAINING_IDS, constants.learning_data_seed)

def create_full_data():
    create_data_and_labels(_ALL_IDS, _ALL_DATA_PREFIX)

if __name__== "__main__" :
    parser = argparse.ArgumentParser(description='Initial datasets creator.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-b', nargs='?', dest='cutpoint', type=float, 
        help='balances initial data considering a maximum frequency per class.')
    group.add_argument('-s', action='store_const', const=_SEED, dest='action',
        help='creates the initial data set for the learning process.')
    group.add_argument('-f', action='store_const', const=_FULL, dest='action',
        help='creates the initial data set for experiments 1 and 3.')

    args = parser.parse_args()
    action = args.action
    if action is None:
        cutpoint = args.cutpoint
        create_balanced_data(cutpoint, _ALL_DATA_PREFIX, _BALANCED_DATA_PREFIX)
    elif action == _SEED:
        create_learning_seeds()
    else:
        create_full_data()
 