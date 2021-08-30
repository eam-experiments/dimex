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

import os
import sys
import numpy as np

# Directory where all results are stored.
run_path = './runs'
idx_digits = 3

testing_path = 'test'
memories_path = 'memories'

labels_prefix = 'labels'
features_prefix = 'features'
memories_prefix = 'memories'
model_prefix = 'model'
recognition_prefix = 'recognition'
stats_prefix = 'model_stats'

learning_data_seed = 'seed'
learning_data_learnt = 'learned'

# Categories prefixes.
stats_model_name = 'model_stats'
data_name = 'data'

original_suffix = '-original'

# Categories suffixes.
training_suffix = '-training'
filling_suffix = '-filling'
testing_suffix = '-testing'
memory_suffix = '-memories'

# Model suffixes.
classifier_suffix = '-classifier'
decoder_suffix = '-autoencoder'

# Other suffixes.
matrix_suffix = '-matrix'
data_suffix = '-X'
labels_suffix = '-Y'

agreed_suffix = '-agr'
original_suffix = '-ori'
amsystem_suffix = '-ams'
nnetwork_suffix = '-rnn'
recognition_suffixes = [[agreed_suffix], [agreed_suffix, original_suffix],
    [agreed_suffix, original_suffix, amsystem_suffix],
    [agreed_suffix, original_suffix, nnetwork_suffix],
    [agreed_suffix, original_suffix, amsystem_suffix, nnetwork_suffix]]


mfcc_numceps = 26
training_stages = 10 
domain = 64
n_frames = 8
n_jobs = 4

nn_training_percent = 0.69
am_filling_percent = 0.21
am_testing_percent = 0.10

n_labels = 22
labels_per_memory = [0, 1, 2]
all_labels = list(range(n_labels))
label_formats = ['r:v', 'y--d', 'g-.4', 'y-.3', 'k-.8', 'y--^',
    'c-..', 'm:*', 'c-1', 'b-p', 'm-.D', 'c:D', 'r--s', 'g:d',
    'm:+', 'y-._', 'm:_', 'y--h', 'g--*', 'm:_', 'g-_', 'm:d']

precision_idx = 0
recall_idx = 1
entropy_avg_idx = 2
entropy_std_idx = 3
n_measures = 4

no_response_idx = 2
no_correct_response_idx = 3
no_correct_chosen_idx = 4
correct_response_idx = 5
mean_responses_idx = 6

n_behaviours = 7

memory_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
memory_fills = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 100.0]
ideal_memory_size = 256

n_samples = 10
ciempiess_segment_size = 250

CHARACTERIZE = -3
TRAIN_AUTOENCODER = -2
TRAIN_CLASSIFIER = -1
GET_FEATURES = 0
EXP_1 = 1
EXP_2 = 2
EXP_3 = 3
EXP_4 = 4
EXP_5 = 5
EXP_6 = 6
EXP_7 = 7
EXP_8 = 8
EXP_9 = 9
EXP_10 = 10

MIN_EXPERIMENT = 1
MAX_EXPERIMENT = 10

bar_patterns = [[1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0],
            [1,1,1,0,0,0,0,1,1,1,0,0,0,0,1,1,1,0,0,0,0,1,1,1,0,0,0,0],
            [1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1],
            [1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1]]
N_BARS = len(bar_patterns)


def print_warning(*s):
    print('WARNING:', *s, file = sys.stderr)

def print_error(*s):
    print('ERROR:', *s, file = sys.stderr)

def tolerance_suffix(tolerance):
    return '' if not tolerance else '-tol_' + str(tolerance).zfill(3)

def experiment_suffix(experiment):
    return '' if not experiment else '-exp_' + str(experiment).zfill(3)

def counter_suffix(counter):
    return '' if not counter else '-cnt_' + str(counter).zfill(3)    

def fold_suffix(fold):
    return '' if not fold else '-fold_' + str(fold).zfill(3)    

def filename(s, idx = None, tolerance = 0, extension = '',
    experiment = None, counter = 0):
    """ Returns a file name in run_path directory with a given extension and an index
    """
    # Create target directory & all intermediate directories if don't exists
    try:
        os.makedirs(run_path)
        print("Directory " , run_path ,  " created ")
    except FileExistsError:
        pass

    if idx is None:
        return run_path + '/' + s + extension
    else:
        return run_path + '/' + s + '-' + str(idx).zfill(3) \
            + experiment_suffix(experiment) \
            + tolerance_suffix(tolerance) \
            + counter_suffix(counter) + extension


def json_filename(s):
    """ Returns a file name for a JSON file in run_path directory
    """
    return filename(s,  extension = '.json')


def csv_filename(s, idx = None, tolerance = 0, experiment = None, counter = 0):
    """ Returns a file name for csv(i) in run_path directory
    """
    return filename(s, idx, tolerance, '.csv', experiment, counter)


def data_filename(s, idx = None):
    return filename(s, idx, extension='.npy')

def pickle_filename(s, idx = None):
    return filename(s, idx, extension='.pkl')


def picture_filename(s, idx = None, tolerance = 0):

    """ Returns a file name for a graph.
    """
    return filename(s, idx, tolerance, '.svg')

def classifier_filename(s, idx = None, tolerance=0, counter = None):
    return filename(s + classifier_suffix, idx, tolerance, counter = counter)

def decoder_filename(s, idx = None, tolerance=0, counter = None):
    return filename(s + decoder_suffix, idx, tolerance, counter = counter)

def recog_filename(s, experiment = None, idx = None, tolerance = None, counter = 0):
    return csv_filename(s, idx, tolerance, experiment, counter)


def image_filename(dir, stage, idx, label, suffix = ''):
    image_path = run_path + '/images/' + dir + '/' + 'stage_' + str(stage) + '/'

    try:
        os.makedirs(image_path)
    except FileExistsError:
        pass

    image_path += str(label) + '_' + str(idx).zfill(5)  + suffix + '.png'
    return image_path


def testing_directory(i):
    return testing_path + '-' + str(i).zfill(3)


def memories_directory(i, tolerance = 0):
    return memories_path + '-' + str(i).zfill(3) \
        + tolerance_suffix(tolerance)


def memory_filename(dir, msize, stage, idx, label):
    # Remove '-'
    image_path = run_path + '/images/' + dir + '/' + 'stage_' + str(stage) + '/'
    image_path += 'msize_' + str(msize) + '/'

    try:
        os.makedirs(image_path)
        print("Directory " , image_path ,  " created ")
    except FileExistsError:
        pass

    image_path += str(label) + '_' + str(idx).zfill(5) + '.png'
    return image_path

def original_image_filename(dir, stage, idx, label):
    return image_filename(dir, stage, idx, label, original_suffix)

def produced_image_filename(dir, stage, idx, label):
    return image_filename(dir, stage, idx, label)

def produced_memory_filename(dir, msize, stage, idx, label):
    return memory_filename(dir, msize, stage, idx, label)

def seed_data_filename():
    return data_filename(learning_data_seed + data_suffix)

def seed_labels_filename():
    return data_filename(learning_data_seed + labels_suffix)

def learned_data_filename(suffix, fold, n):
    prefix = learning_data_learnt + suffix + fold_suffix(fold) + data_suffix
    return data_filename(prefix, n)

def learned_labels_filename(suffix, fold, n):
    prefix = learning_data_learnt + suffix + fold_suffix(fold) + labels_suffix
    return data_filename(prefix, n)

def get_name_w_suffix(prefix, n, v, sf):
    suffix = ''
    if n > 0:
        suffix = sf(v)
    return prefix + suffix 

def features_name(n = -1):
    return get_name_w_suffix(features_prefix, n, n, experiment_suffix)

def labels_name(n = -1):
    return get_name_w_suffix(labels_prefix, n, n, experiment_suffix)

def memories_name(n = -1, tolerance = 0):
    return get_name_w_suffix(memories_prefix, n, tolerance, tolerance_suffix)

def model_name(n = -1):
    return get_name_w_suffix(model_prefix, n, n, experiment_suffix)

def stats_name(n = -1):
    return get_name_w_suffix(stats_prefix, n, n, experiment_suffix)

def mean_idx(m):
    return m

def std_idx(m):
    return m+1


def padding_cropping(data, n_frames):

    frames, _  = data.shape
    df = n_frames - frames
    if df == 0:
        return data
    elif df < 0:
        return data[:n_frames]
    else:
        top_padding = df // 2
        bottom_padding = df - top_padding
        return np.pad(data, ((top_padding, bottom_padding),(0,0)),
            'constant', constant_values=((0,0),(0,0)))

def print_counter(n, every, step = 1, symbol = '.'):
    e = n % every
    s = n % step
    if e and s:
        return
    counter = symbol
    if not e:
        counter =  ' ' + str(n) + ' '
    print(counter, end = '', flush=True)

