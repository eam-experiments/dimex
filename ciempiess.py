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
    def __init__(self, stage):
        self._stage = stage
        nextdata_filename = constants.data_filename(self._NEXTDATA_PREFIX, stage=stage)
        try:
            self._next_data = np.load(nextdata_filename)
            print(f'File {nextdata_filename} exists.')
        except:
            self._next_data = None
            print(f'File {nextdata_filename} does not exists.')

        if self._next_data is None:
            audios_filenames = []
            with open(self._AUDIO_LIST_FILENAME, 'rt') as f:
                audios_filenames = [filename.rstrip() for filename in f]
            start = self._stage*constants.ciempiess_segment_size
            end = start + constants.ciempiess_segment_size
            self._next_data = self._get_data(audios_filenames[start:end])
            np.save(nextdata_filename, self._next_data)
    
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

