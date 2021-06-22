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
import random
import re
import scipy.io.wavfile as wav
import scipy.signal
from python_speech_features import mfcc
import constants


class TaggedAudio:
    def __init__(self):
        self.text = ''
        self.segments = []


class DimexSampler:
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
    _NUMCEP = 26

    def __init__(self):
        """ Creates the random sampler by reading the identifiers file and storing it
            in memory.

            It also reads the phonemes and does the same.
        """
        self._ids = []
        file_name = self._CORPUS_DIR + '/' + self._IDS_FILE
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
            audio = TaggedAudio()
            audio.text = self._get_text(modifier, id)
            audio.segments = self._get_segments(modifier, id)
            audios.append(audio)
        if n == 1:
            return audios[0]
        else:
            return audios

    def _get_text(self, modifier, id):
        file_name = self._get_text_filename(modifier, id)

        text = None
        with open(file_name, 'r', encoding="ISO-8859-1") as file:
            text = file.readline()
            text = re.sub(' *\.\n', '', text)
        return text

    def _get_segments(self, modifier, id):
        audio_fname = self._get_audio_filename(modifier, id)
        phn_fname = self._get_phonemes_filename(modifier, id)
        segments = self._get_features_phonemes(audio_fname, phn_fname)
        return segments

    def _get_features_phonemes(self, audio_fname, phn_fname):
        sample_rate, signal = wav.read(audio_fname)
        with open(phn_fname, 'r') as file:
            reader = csv.reader(file, delimiter = ' ')
            row = next(reader)
            while (row[0] != 'END'):
                row = next(reader, None)  # skip the headers
            segments = self._generate_feats_per_phoneme(sample_rate, signal, reader)
            return(segments)

    def _generate_feats_per_phoneme(self, sample_rate, signal, reader):
        segments = []
        start = 0.0
        for row in reader:
            phn = row[2]
            if (phn == '.sil') or (phn == '.bn'):
                continue
            phn_start = float(row[0])
            phn_end = float(row[1])
            duration = phn_end - phn_start
            segment = signal[int(sample_rate*phn_start/1000)
                                 :int(sample_rate*phn_end/1000)]
            if sample_rate != self._IDEAL_SRATE and len(segment) != 0:
                sampls = int(self._IDEAL_SRATE*duration/1000)
                segment = scipy.signal.resample(segment, sampls)
            features = mfcc(segment, self._IDEAL_SRATE, numcep=self._NUMCEP)
            features = constants.padding_cropping(features, constants.n_frames)
            segments.append((phn, features))
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
