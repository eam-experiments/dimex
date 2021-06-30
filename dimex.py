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
    def __init__(self, id):
        self.id = id
        self.text = ''
        self.segments = []
        self.features = []
        self.net_labels = []
        self.ams_labels = []

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
