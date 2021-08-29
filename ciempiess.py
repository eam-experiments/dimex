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
import numpy as np
import pickle
from python_speech_features import mfcc
import random
import re
import scipy.signal
import soundfile as sf  
import constants
from dimex import TaggedAudio

_CORPUS_DIR = 'CIEMPIESS'
_SPEECH_DIR = 'data/speech'
_AUDIO_EXT = 'flac'

# CIEMPIESS uses the IDEAL_SRATE.
IDEAL_SRATE = 16000

def get_full_filename(fn):
    filename = _CORPUS_DIR + '/' + _SPEECH_DIR + '/' + fn
    return filename


class NextDataSet:
    _AUDIO_LIST_FILENAME = 'ciempiess.txt'
    _SEGMENT_MILLISECONDS = 90
    _NEXTDATA_PREFIX = 'ciempiess_data'
    def __init__(self, n):
        self._n = n
        nextdata_filename = constants.pickle_filename(self._NEXTDATA_PREFIX, n)
        try:
            with open(nextdata_filename, 'rb') as f:
                self._next_data = pickle.load(f)
            print(f'File {nextdata_filename} exists.')
        except:
            self._next_data = None
            print(f'File {nextdata_filename} does not exists.')

        if self._next_data is None:
            audios_filenames = []
            with open(self._AUDIO_LIST_FILENAME, 'rt') as f:
                audios_filenames = [filename.rstrip() for filename in f]
            start = self._n*constants.ciempiess_segment_size
            end = start + constants.ciempiess_segment_size
            self._next_data = self._get_data(audios_filenames[start:end])
            with open(nextdata_filename, 'wb') as f:
                pickle.dump(self._next_data, f)
    
    def get_data(self):
        return self._next_data
 
    def _get_data(self, filenames):
        audios = []
        i = 0
        for fn in filenames:
            id = re.sub('\.' + _AUDIO_EXT + '$', '', fn)
            audio = TaggedAudio(id)
            audio.segments = self._get_segments(fn)
            audios.append(audio)
            i += 1
            constants.print_counter(i, 100, 10)
        return audios

    def _get_segments(self, filename):
        audio_fname = get_full_filename(filename)
        signal, sample_rate = sf.read(audio_fname)
        segments = self._get_mfcc(signal, sample_rate)
        return segments

    def _get_mfcc(self, signal, sample_rate):
        if sample_rate != IDEAL_SRATE:
            new_length = int(IDEAL_SRATE*len(signal)/sample_rate)
            signal = scipy.signal.resample(signal, new_length)

        segments = self._scan_audio(IDEAL_SRATE, signal)
        return np.array(segments)

    def _scan_audio(self, sample_rate, signal):
        segments = []
        seg_len = int(self._SEGMENT_MILLISECONDS*sample_rate/1000)
        step = int(sample_rate/100)
        i = 0
        end = len(signal)
        stop = False
        while not stop:
            j = i + seg_len
            if j > end:
                j = end
                stop = True
            segment = signal[i:j]
            features = mfcc(segment, sample_rate, numcep=constants.mfcc_numceps)
            features = constants.padding_cropping(features, constants.n_frames)
            segments.append(features)
            i += step
        return segments


class LearnedDataSet:
    _TRAINING_PREFIX = 'seed'
    _LEARNED_PREFIX = 'learned'
    _RECOG_SUFFIXES = [['-agr'],['-agr','-ams'], ['-agr','-rnn'], ['-agr', '-ams', '-rnn'],
        ['-agr', '-ams', '-rnn','-ori']]

    def __init__(self, tolerance):
        """ Creates learned data set for given 'tolerance'.

            Tolerance:
                0: Only agreed averaged learned features.
                1: Agreed averaged and averaged accepted by memory.
                2: Agreed averaged and averaged accepted by neural network.
                3: Agreed averaged and averaged accepted by any.
                4: Agreed averaged, averaged accepted, and original.
        """
        if (tolerance < 0) or (len(self._RECOG_SUFFIXES) <= tolerance):
            constants.print_error(f'Tolerance {tolerance} is out of range.')
            exit(1)
        self._tolerance = tolerance

    def get_data(self):
        data, labels = self.get_seed_data()
        learned_data, learned_labels, counter = self.get_learned_data(self._tolerance)
        if not ((learned_data is None) or (learned_labels is None)):
            data = np.concatenate((data, learned_data), axis=0)
            labels = np.concatenate((labels, learned_labels), axis=0)
        return shuffle(data, labels), counter

    def get_seed_data(self):
        data_filename = constants.seed_data_filename()
        labels_filename = constants.seed_labels_filename()
        data = np.load(data_filename)
        labels = np.load(labels_filename)
        return data, labels    

    def get_learned_data(self, tolerance):
        have_been_data = True
        n = 0
        data = []
        labels = []
        suffixes = self._RECOG_SUFFIXES[tolerance]
        while have_been_data:
            new_data, new_labels = self._get_stage_learned_data(suffixes, n)
            have_been_data = not ((new_data is None) or (new_labels is None))
            if have_been_data:
                data.append(new_data)
                labels.append(new_labels)
                n += 1
        if data and labels:
            data = np.concatenate(data, axis= 0)
            labels = np.concatenate(labels, axis=0)
        else:
            data = None
            labels = None
        return data, labels, n

    def _get_stage_learned_data(self, suffixes, stage):
        data = []
        labels = []
        for s in suffixes:
            data_filename = constants.learned_data_filename(s, stage)
            labels_filename = constants.learned_data_filename(s, stage)
            try:
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
    _FEATURES_DIR_NAME = 'Features'
    _MEANS_FILE_NAME = _FEATURES_DIR_NAME + '/media.npy'
    _STDEV_FILE_NAME = _FEATURES_DIR_NAME + '/std.npy'
    _INITIAL_HITS = 3
    _MAX_MISSES = 3

    class Parameters:
        def __init__(self):
            self.mean = None
            self.stdev = None

    class Counter:
        def __init__(self):
            self.minimum = None
            self.maximum = None
            self.hits = 0
            self.misses = 0
            self.prints = 0

    def __init__(self):
        # The list of parameters include one extra element to deal with unknown (None).
        self.params = [ self.Parameters() for i in range(constants.n_labels+1)]

        # Get means per phoneme as a dictionary.
        phn_means = np.load(self._MEANS_FILE_NAME, allow_pickle=True).item()
        unknown_mean = 0
        for phn in phn_means:
            i = phns_to_labels[phn]
            p = self.params[i]
            p.mean = phn_means[phn]
            unknown_mean += p.mean

        phn_stdevs = np.load(self._STDEV_FILE_NAME, allow_pickle=True).item()
        unknown_stdev = 0
        for phn in phn_stdevs:
            i = phns_to_labels[phn]
            p = self.params[i]
            p.stdev = phn_stdevs[phn]
            unknown_stdev += p.stdev

        unknown_mean /= constants.n_labels
        unknown_stdev /= constants.n_labels
        p = self.params[constants.n_labels]
        p.mean = unknown_mean
        p.stdev = unknown_stdev


    def _get_counters(self):
        """ Constructs the list of counters for processing the labels.
        
            The list of counters includes one extra element to deal with
            None (unrecognized).
        """
        counters = [ self.Counter() for i in range(constants.n_labels+1)]
        for i in range(constants.n_labels+1):
            minimum = (self.params[i].mean - self.params[i].stdev)/10.0
            minimum = int(minimum) if minimum > 0.0 else 0
            counters[i].minimum = minimum
            counters[i].maximum = int((self.params[i].mean + self.params[i].stdev)/10.0)
        return counters


    def process(self, labels):
        lbls = [constants.n_labels if label is None else label for label in labels]
        phonemes = []
        c = self._get_counters()
        for label in lbls:
            if c[label].hits == 0:
                c[label].hits = self._INITIAL_HITS
            else:
                c[label].hits += 1
            c[label].misses = 0
            if c[label].hits > c[label].maximum:
                c[label].hits = 0
                c[label].prints = 0
            elif (c[label].hits > c[label].minimum) and (c[label].prints < 1):
                phonemes.append(label)   
                c[label].prints += 1
            # Processes other labels (miss).
            for other in range(constants.n_labels+1):
                if other == label:
                    continue
                c[other].misses += 1
                if c[other].misses == self._MAX_MISSES:
                    c[other].hits = 0
                    c[other].misses = 0
                    c[other].prints = 0
        return phonemes

    def get_phonemes(self, labels):
        phonemes = ''
        for label in labels:
            if (label is None) or (label == constants.n_labels):
                phonemes += unknown_phn
            else:
                phonemes += labels_to_phns[label]
        return phonemes
  
