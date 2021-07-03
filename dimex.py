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

from abc import abstractclassmethod
import csv
import numpy as np
import random
import re
import scipy.io.wavfile as wav
import scipy.signal
from python_speech_features import mfcc
import constants


phns_to_labels = {'g': 0, 'n~': 1, 'f': 2, 'd': 3, 'n': 4, 'm': 5, 
    'r(': 6, 's': 7, 'e': 8, 'tS': 9, 'p': 10, 'l': 11, 'k': 12,
    't': 13, 'b': 14, 'Z': 15, 'i': 16, 'x': 17, 'o': 18, 'a': 19,
    'r': 20, 'u': 21}
labels_to_phns = ['g', 'n~', 'f', 'd', 'n', 'm', 
    'r(', 's', 'e', 'tS', 'p', 'l', 'k',
    't', 'b', 'Z', 'i', 'x', 'o', 'a',
    'r', 'u']
phonemes = labels_to_phns
unknown_phn = '-'



class TaggedAudio:
    def __init__(self, id):
        self.id = id
        self.text = ''
        self.segments = []
        self.features = []
        self.labels = []
        self.net_labels = []
        self.ams_labels = []

class Sampler:
    _CORPUS_DIR = 'Corpus'
    _AUDIO_DIR = 'audio_editado'
    _PHONEMES_DIR = 'T22'
    _TEXT_DIR = 'texto'
    _IDS_FILE = 'ids.csv'
    _PHNS_FILE = 't22-phonemes.csv'
    _TEXT_EXT = '.txt'
    _AUDIO_EXT = '.wav'
    _PHN_EXT = '.phn'
    _IDEAL_SRATE = 16000
    _SEGMENT_MILLISECONDS = 80

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

    def get_sample(self, n = 1):
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
        file_name = self._get_text_filename(modifier, id)
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
        file_name = self._get_phonemes_filename(modifier, id)
        with open(file_name, 'r') as file:
            reader = csv.reader(file, delimiter = ' ')
            row = next(reader)
            # skip the headers
            while (row[0] != 'END'):
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
        audio_fname = self._get_audio_filename(modifier, id)
        segments = self._get_mfcc(audio_fname)
        return segments

    def _get_mfcc(self, audio_fname):
        sample_rate, signal = wav.read(audio_fname)
        if sample_rate != self._IDEAL_SRATE:
            new_length = int(self._IDEAL_SRATE*len(signal)/sample_rate)
            new_signal = scipy.signal.resample(signal, new_length)

        segments = self._scan_audio(self._IDEAL_SRATE, new_signal)
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

    def _get_text_filename(self, modifier, id):
        return self._get_file_name(modifier, id, self._TEXT_DIR, self._TEXT_EXT)

    def _get_audio_filename(self, modifier, id):
        return self._get_file_name(modifier, id, self._AUDIO_DIR, self._AUDIO_EXT)

    def _get_phonemes_filename(self, modifier, id):
        return self._get_file_name(modifier, id, self._PHONEMES_DIR, self._PHN_EXT)

    def _get_file_name(self, modifier, id, cls, extension):
        sdir = id[:4]
        file_name = self._CORPUS_DIR + '/' + sdir + '/' + cls + '/'
        file_name += modifier + '/' + id + extension
        return file_name


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
            if (label is None) or (labels == constants.n_labels):
                phonemes += unknown_phn
            else:
                phonemes += labels_to_phns[label]
        return phonemes
  