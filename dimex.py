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

import copy
import csv
import numpy as np
import pickle
from python_speech_features import mfcc
import random
import re
import scipy.io.wavfile as wav
import scipy.signal
import constants

phns_to_labels = {
    'g': 0,
    'n~': 1,
    'f': 2,
    'd': 3,
    'n': 4,
    'm': 5,
    'r(': 6,
    's': 7,
    'e': 8,
    'tS': 9,
    'p': 10,
    'l': 11,
    'k': 12,
    't': 13,
    'b': 14,
    'Z': 15,
    'i': 16,
    'x': 17,
    'o': 18,
    'a': 19,
    'r': 20,
    'u': 21
}
labels_to_phns = [
    'g', 'n~', 'f', 'd', 'n', 'm', 'r(', 's', 'e', 'tS', 'p', 'l', 'k', 't',
    'b', 'Z', 'i', 'x', 'o', 'a', 'r', 'u'
]
phonemes = labels_to_phns
unknown_phn = '-'

_CORPUS_DIR = 'Corpus'
_AUDIO_DIR = 'audio_editado'
_PHONEMES_DIR = 'T22'
_TEXT_DIR = 'texto'
_TEXT_EXT = '.txt'
_AUDIO_EXT = '.wav'
_PHN_EXT = '.phn'

IDEAL_SRATE = 16000


def get_text_filename(modifier, id):
    return _get_file_name(modifier, id, _TEXT_DIR, _TEXT_EXT)


def get_audio_filename(modifier, id):
    return _get_file_name(modifier, id, _AUDIO_DIR, _AUDIO_EXT)


def get_phonemes_filename(modifier, id):
    return _get_file_name(modifier, id, _PHONEMES_DIR, _PHN_EXT)


def shuffle(data, labels):
    pairs = [(data[i], labels[i]) for i in range(len(labels))]
    random.shuffle(pairs)
    data = np.array([p[0] for p in pairs])
    labels = np.array([p[1] for p in pairs], dtype='int')
    return data, labels

def balance_pairs(pairs, cutpoint):
    freqs = np.zeros(constants.n_labels, dtype=int)
    balanced = []
    for d, l in pairs:
        if freqs[l] < cutpoint:
            balanced.append((d,l))
            freqs[l] += 1
    return balanced


def frequencies(labels):
    freqs = np.zeros(constants.n_labels, dtype=int)
    for label in labels:
        freqs[label] += 1
    return freqs

def cutpoint(labels):
    freqs = frequencies(labels)
    return np.median(freqs)

def shuffle_and_balance(data, labels):
    print(f'Shuffling and balancing classes with cutpoint of {cp}.')
    pairs = [(data[i], labels[i]) for i in range(len(labels))]
    random.shuffle(pairs)
    cp = cutpoint(labels)
    pairs = balance_pairs(pairs, cp)
    balanced_data = np.array([p[0] for p in pairs], dtype = 'float')
    balanced_labels = np.array([p[1] for p in pairs], dtype = 'int')
    return balanced_data, balanced_labels

def balance(data, labels):
    pairs = [(data[i], labels[i]) for i in range(len(labels))]
    cp = cutpoint(labels)
    print(f'Balancing classes with cutpoint of {cp}.')
    pairs = balance_pairs(pairs, cp)
    balanced_data = np.array([p[0] for p in pairs], dtype = 'float')
    balanced_labels = np.array([p[1] for p in pairs], dtype = 'int')
    return balanced_data, balanced_labels

def _get_file_name(modifier, id, cls, extension):
    sdir = id[:4]
    file_name = _CORPUS_DIR + '/' + sdir + '/' + cls + '/'
    file_name += modifier + '/' + id + extension
    return file_name


class TaggedAudio:
    def __init__(self, id):
        self.id = id  # Audio id
        self.text = ''  # Textual transcription of audio.
        self.labels = []  # Phonemes as integers.
        self.segments = []  # MFCC of phoneme audio segments
        self.features = []  # Features of MFCC segments.
        self.net_labels = []  # Classification of segments by neural network.
        self.ams_labels = []  # Classification of segments by memories.
        self.ams_features = [
        ]  # Features of MFCC segments as recalled by memories.
        self.net_segments = []  # MFCC generated by decoder from segments.
        self.ams_segments = []  # MFCC generated by decoder from remembrances.


class Sampler:
    _IDS_FILE = 'ids.csv'
    _PHNS_FILE = 't22-phonemes.csv'
    _SEGMENT_MILLISECONDS = 90

    def __init__(self):
        """ Creates the random sampler by reading the identifiers file and storing it
            in memory.

            It also reads the phonemes and does the same.
        """
        self._ids = []
        file_name = self._IDS_FILE
        with open(file_name, 'r') as file:
            reader = csv.reader(file)
            next(reader, None)
            for row in reader:
                self._ids.append(tuple(row))

    def get_sample(self, n=1):
        sample = random.sample(self._ids, n)
        audios = []
        for s in sample:
            modifier = s[0]
            id = s[1]
            audio = TaggedAudio(id)
            audio.text = self._get_text(modifier, id)
            audio.labels = self._get_labels(modifier, id)
            audio.segments = self._get_segments(modifier, id)
            audios.append(audio)
        if n == 1:
            return audios[0]
        else:
            return audios

    def _get_text(self, modifier, id):
        file_name = get_text_filename(modifier, id)
        text = ''
        try:
            with open(file_name, 'r', encoding="ISO-8859-1") as file:
                text = file.readline()
                text = re.sub(' *\.\n', '', text)
        except:
            pass
        return text

    def _get_labels(self, modifier, id):
        labels = []
        file_name = get_phonemes_filename(modifier, id)
        with open(file_name, 'r') as file:
            reader = csv.reader(file, delimiter=' ')
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
                label = phns_to_labels[phn]
                labels.append(label)
        return labels

    def _get_segments(self, modifier, id):
        audio_fname = get_audio_filename(modifier, id)
        segments = self._get_mfcc(audio_fname)
        return segments

    def _get_mfcc(self, audio_fname):
        sample_rate, signal = wav.read(audio_fname)
        if sample_rate != IDEAL_SRATE:
            new_length = int(IDEAL_SRATE * len(signal) / sample_rate)
            new_signal = scipy.signal.resample(signal, new_length)

        segments = self._scan_audio(IDEAL_SRATE, new_signal)
        return np.array(segments)

    def _scan_audio(self, sample_rate, signal):
        segments = []
        seg_len = int(self._SEGMENT_MILLISECONDS * sample_rate / 1000)
        step = int(sample_rate / 100)
        i = 0
        end = len(signal)
        stop = False
        while not stop:
            j = i + seg_len
            if j > end:
                j = end
                stop = True
            segment = signal[i:j]
            features = mfcc(segment,
                            sample_rate,
                            numcep=constants.mfcc_numceps)
            features = constants.padding_cropping(features, constants.n_frames)
            segments.append(features)
            i += step
        return segments


class TestingDataSet:
    _INDIVIDUALS_FILE = 'testing-individuals.csv'
    _COMMONS_FILE = 'testing-commons.csv'
    _TESTING_PREFIX = 'testing-data'
    _SEGMENT_MILLISECONDS = 90

    def __init__(self):
        """ Creates the random sampler by reading the identifiers file and storing it
            in memory.

            It also reads the phonemes and does the same.
        """
        testing_filename = constants.pickle_filename(self._TESTING_PREFIX)
        try:
            with open(testing_filename, 'rb') as f:
                self.testing_data = pickle.load(f)
        except:
            self.testing_data = None

        if self.testing_data is None:
            ids = []
            ids += self._get_ids(self._INDIVIDUALS_FILE)
            ids += self._get_ids(self._COMMONS_FILE)
            self.testing_data = self._get_data(ids)
            with open(testing_filename, 'wb') as f:
                pickle.dump(self.testing_data, f)

    def get_data(self):
        return self.testing_data

    def _get_ids(self, filename):
        ids = []
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            next(reader, None)
            for row in reader:
                ids.append(tuple(row))
        return ids

    def _get_data(self, ids):
        audios = []
        for i in ids:
            modifier = i[0]
            id = i[1]
            audio = TaggedAudio(id)
            audio.text = self._get_text(modifier, id)
            audio.labels = self._get_labels(modifier, id)
            audio.segments = self._get_segments(modifier, id)
            audios.append(audio)
        return audios

    def _get_text(self, modifier, id):
        file_name = get_text_filename(modifier, id)
        text = ''
        try:
            with open(file_name, 'r', encoding="ISO-8859-1") as file:
                text = file.readline()
                text = re.sub(' *\.\n', '', text)
        except:
            pass
        return text

    def _get_labels(self, modifier, id):
        labels = []
        file_name = get_phonemes_filename(modifier, id)
        with open(file_name, 'r') as file:
            reader = csv.reader(file, delimiter=' ')
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
                label = phns_to_labels[phn]
                labels.append(label)
        return labels

    def _get_segments(self, modifier, id):
        audio_fname = get_audio_filename(modifier, id)
        segments = self._get_mfcc(audio_fname)
        return segments

    def _get_mfcc(self, audio_fname):
        sample_rate, signal = wav.read(audio_fname)
        if sample_rate != IDEAL_SRATE:
            new_length = int(IDEAL_SRATE * len(signal) / sample_rate)
            new_signal = scipy.signal.resample(signal, new_length)

        segments = self._scan_audio(IDEAL_SRATE, new_signal)
        return np.array(segments)

    def _scan_audio(self, sample_rate, signal):
        segments = []
        seg_len = int(self._SEGMENT_MILLISECONDS * sample_rate / 1000)
        step = int(sample_rate / 100)
        i = 0
        end = len(signal)
        stop = False
        while not stop:
            j = i + seg_len
            if j > end:
                j = end
                stop = True
            segment = signal[i:j]
            features = mfcc(segment,
                            sample_rate,
                            numcep=constants.mfcc_numceps)
            features = constants.padding_cropping(features, constants.n_frames)
            segments.append(features)
            i += step
        return segments


class LearnedDataSet:
    _TRAINING_PREFIX = 'seed'
    _LEARNED_PREFIX = 'learned'
    _RECOG_SUFFIXES = constants.learning_suffixes
    _TRAINING_SEGMENT = 0
    _FILLING_SEGMENT = 1
    _TESTING_SEGMENT = 2

    def __init__(self, es, fold):
        self.es = es
        self.fold = fold
        self.seed_data, self.seed_labels = self._get_seed_data()
        self.learned_data, self.learned_labels = self._get_learned_data(es, fold)
        if not ((self.learned_data is None) or (self.learned_labels is None)):
            self.learned_data, self.learned_labels  = balance(self.learned_data, self.learned_labels)

    def _get_data_segment(self, data, labels, segment, fold):
        total = len(labels)
        training = total*constants.nn_training_percent
        filling = total*constants.am_filling_percent
        testing = total*constants.am_testing_percent
        step = total / constants.n_folds
        i = fold * step
        j = i + training
        k = j + filling
        l = k + testing
        i = int(i)
        j = int(j) % total
        k = int(k) % total
        l = int(l) % total
        n, m = None, None
        if segment == self._TRAINING_SEGMENT:
            n, m = i, j
        elif segment == self._FILLING_SEGMENT:
            n, m = j, k
        elif segment == self._TESTING_SEGMENT:
            n, m = k, l
        return constants.get_data_in_range(data, n, m), \
                constants.get_data_in_range(labels, n, m)

    def get_data_segment(self, segment, fold):
        seed_data, seed_labels = \
            self._get_data_segment(self.seed_data, self.seed_labels, segment, fold)
        if (self.learned_data is None) or (self.learned_labels is None):
            return seed_data, seed_labels
        elif (segment == self._TESTING_SEGMENT) and not self.es.extended:
            return seed_data, seed_labels
        else:
            learned_data, learned_labels = \
                self._get_data_segment(self.learned_data, self.learned_labels,
                    segment, 0)
            data = np.concatenate((seed_data, learned_data), axis=0)
            labels = np.concatenate((seed_labels, learned_labels), axis=0)
            data, labels = shuffle(data, labels)
            return data, labels

    def get_training_data(self):
        return self.get_data_segment(self._TRAINING_SEGMENT, self.fold)

    def get_filling_data(self):
        return self.get_data_segment(self._FILLING_SEGMENT, self.fold)

    def get_testing_data(self):
        return self.get_data_segment(self._TESTING_SEGMENT, self.fold)

    def _get_seed_data(self):
        data_filename = constants.seed_data_filename()
        labels_filename = constants.seed_labels_filename()
        data = np.load(data_filename)
        labels = np.load(labels_filename)
        return data, labels

    def _get_learned_data(self, es, fold):
        les = None
        # Extended mode can start in any stage.
        if es.extended:
            les = copy.copy(es)
            les.extended = False
        have_been_data = True
        stage = 0
        data = []
        labels = []
        suffixes = self._RECOG_SUFFIXES[es.learned]
        while have_been_data and (stage < es.stage):
            new_data, new_labels = self._get_stage_learned_data(suffixes, stage, es, fold)
            have_been_data = not ((new_data is None) or (new_labels is None))
            if (not have_been_data) and es.extended:
                new_data, new_labels = self._get_stage_learned_data(suffixes, stage, les, fold)
                have_been_data = not ((new_data is None) or (new_labels is None))
            if have_been_data:
                data.append(new_data)
                labels.append(new_labels)
                stage += 1
        if (stage < es.stage):
            constants.print_error(f'Last {es.stage - stage} stage(s) missing.')
            raise ValueError
        if data and labels:
            data = np.concatenate(data, axis=0)
            labels = np.concatenate(labels, axis=0)
        else:
            data = None
            labels = None
        return data, labels

    def _get_stage_learned_data(self, suffixes, stage, es, fold):
        les = copy.copy(es)
        les.stage = stage
        data = []
        labels = []
        for s in suffixes:
            data_filename = constants.learned_data_filename(s, les, fold)
            labels_filename = constants.learned_labels_filename(s, les, fold)
            try:
                print(f'Getting learned data from {data_filename} and {labels_filename}')
                new_data = np.load(data_filename)
                new_labels = np.load(labels_filename)
            except:
                return None, None
            data.append(new_data)
            labels.append(new_labels)
        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)
        return data, labels


class PostProcessor:
    def process(self, labels):
        return labels

    def get_phonemes(self, labels):
        phonemes = ''
        for label in labels:
            if (label is None) or (label == constants.n_labels):
                continue
            else:
                phonemes += labels_to_phns[label]
        return phonemes
