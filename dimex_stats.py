import argparse
import csv
import numpy as np
from python_speech_features import mfcc
import random
import scipy.io.wavfile as wav
import scipy.signal

import constants
import dimex

_FEATURES_DIR = 'Features'
_TRAINING_DATA = 'training.csv'

_BALANCED = 0
_SEED = 1
_LABELS_SUFFIX = '_Y.npy'
_DATA_SUFFIX = '_X.npy'


def create_balanced_data(cut_point, in_prefix, out_prefix, convert=False):
   # Load original data
    labels_filename = in_prefix + _LABELS_SUFFIX
    features_filename = in_prefix + _DATA_SUFFIX
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
    np.save(out_prefix + _DATA_SUFFIX, features)
    np.save(out_prefix + _LABELS_SUFFIX, labels)


def get_mods_ids(file_name):
    mods_ids = []
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        # Skip the header.
        next(reader, None)
        for row in reader:
            mods_ids.append(tuple(row))
    return mods_ids


def create_learning_seeds():
    mods_ids = get_mods_ids(_TRAINING_DATA)

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
                features = constants.padding_cropping(features, constants.n_frames)
                label = dimex.phns_to_labels[phn]
                data.append(features)
                labels.append(label)

            
        counter += 1
        if (counter % 100) == 0:
            print(f' {counter} ', end = '', flush=True)
        elif (counter % 10) == 0:
            print('.', end = '', flush=True)
    
    data, labels = dimex.shuffle(data, labels)
    filename = constants.seed_data_filename()
    np.save(filename,data)
    filename = constants.seed_labels_filename()
    np.save(filename,labels)
    


if __name__== "__main__" :
    parser = argparse.ArgumentParser(description='Initial datasets creator.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-b', nargs='?', dest='cutpoint', type=float, 
        help='balances initial data considering a maximum frequency per class.')
    group.add_argument('-s', action='store_const', const=_SEED, dest='action',
        help='creates the initial data set for the learning process.')

    args = parser.parse_args()
    action = args.action
    if action is None:
        cutpoint = args.cutpoint
        create_balanced_data(cutpoint)
    else:
        create_learning_seeds()
 