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
from signal import Sigmasks
import sys
import numpy as np

# Directory where all results are stored.
run_path = './runs'
idx_digits = 3

testing_path = 'test'
memories_path = 'memories'

data_prefix = 'data'
labels_prefix = 'labels'
features_prefix = 'features'
memories_prefix = 'memories'
model_prefix = 'model'
recognition_prefix = 'recognition'
stats_prefix = 'model_stats'
learn_params_prefix ='learn_params'

balanced_data = 'balanced'
seed_data = 'seed'
learning_data_seed = 'seed_balanced'
learning_data_learned = 'learned'

# Categories suffixes.
training_suffix = '-training'
filling_suffix = '-filling'
testing_suffix = '-testing'
memory_suffix = '-memories'

# Model suffixes.
classifier_suffix = '-classifier'
decoder_suffix = '-autoencoder'

# Other suffixes.
original_suffix = '-original'
data_suffix = '_X'
labels_suffix = '_Y'
matrix_suffix = '-confrix'

agreed_suffix = '-agr'
original_suffix = '-ori'
amsystem_suffix = '-ams'
nnetwork_suffix = '-rnn'
learning_suffixes = [[original_suffix], [agreed_suffix], [amsystem_suffix],
    [nnetwork_suffix], [original_suffix, amsystem_suffix]]


mfcc_numceps = 26
n_folds = 10
domain = 32
n_frames = 8
phn_duration = n_frames*10 + 15
n_jobs = 22

nn_training_percent = 0.70
am_filling_percent = 0.20
am_testing_percent = 0.10

n_labels = 22
labels_per_memory = 1
all_labels = list(range(n_labels))
label_formats = ['r:v', 'y--d', 'g-.4', 'y-.3', 'k-.8', 'y--^',
    'c-..', 'm:*', 'c-1', 'b-p', 'm-.D', 'c:D', 'r--s', 'g:d',
    'm:+', 'y-._', 'm:_', 'y--h', 'g--*', 'm:_', 'g-_', 'm:d']

precision_idx = 0
recall_idx = 1
accuracy_idx = 2
entropy_idx = 3
n_measures = 4

no_response_idx = 2
no_correct_response_idx = 3
no_correct_chosen_idx = 4
correct_response_idx = 5
response_size_idx = 6
n_behaviours = 7

memory_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
memory_fills = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 100.0]
ideal_memory_size = 16

n_samples = 10
ciempiess_segment_size = 250
learned_data_groups = 6

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

class ExperimentSettings:
    def __init__(self, stage = 0, learned = 0, extended = False,
        tolerance = 0, sigma = 0.5, iota = 0.0, kappa = 0.0):
        self.stage = stage
        self.learned = learned
        self.extended = extended
        self.tolerance = tolerance
        self.sigma = sigma
        self.iota = iota
        self.kappa = kappa

    def __str__(self):
        s = '{Stage: ' + str(self.stage) + \
            ', Learned: ' + str(self.learned) + \
            ', Extended: ' + str(self.extended) + \
            ', Tolerance: ' + str(self.tolerance) + \
            ', Sigma: ' + str(self.sigma) + \
            ', Iota: ' + str(self.iota) + \
            ', Kappa: ' + str(self.kappa) + '}'
        return s


def print_warning(*s):
    print('WARNING:', *s, file = sys.stderr)

def print_error(*s):
    print('ERROR:', *s, file = sys.stderr)

def print_counter(n, every, step = 1, symbol = '.', prefix = ''):
    if n == 0:
        return
    e = n % every
    s = n % step
    if (e != 0) and (s != 0):
        return
    counter = symbol
    if e == 0:
        counter =  ' ' + prefix + str(n) + ' '
    print(counter, end = '', flush=True)

def extended_suffix(extended):
    return '-ext' if extended else ''    

def fold_suffix(fold):
    return '' if fold is None else '-fld_' + str(fold).zfill(3)

def learned_suffix(learned):
    return '-lrn_' + str(learned).zfill(3)    

def stage_suffix(stage):
    return '-stg_' + str(stage).zfill(3)    

def tolerance_suffix(tolerance):
    return '-tol_' + str(tolerance).zfill(3)

def experiment_suffix(experiment):
    return '' if (experiment is None) or experiment < EXP_1 \
        else '-exp_' + str(experiment).zfill(3)

def get_name_w_suffix(prefix, add_suffix, value, sf):
    suffix = ''
    if add_suffix:
        suffix = sf(value)
    return prefix + suffix 

def get_full_name(prefix, es):
    if es is None:
        return prefix
    name = get_name_w_suffix(prefix, True, es.stage, stage_suffix)
    name = get_name_w_suffix(name, True, es.learned, learned_suffix)
    name = get_name_w_suffix(name, True, es.extended, extended_suffix)
    name = get_name_w_suffix(name, True, es.tolerance, tolerance_suffix)
    return name

# Currently, names include nothing about experiment settings.
def model_name(es):
    return model_prefix

def stats_model_name(es):
    return stats_prefix

def data_name(es):
    return data_prefix

def features_name(es):
    return features_prefix

def labels_name(es):
    return labels_prefix

def memories_name(es):
    return memories_prefix

def learn_params_name(es):
    return learn_params_prefix

def filename(name_prefix, es = None, fold = None, extension = ''):
    """ Returns a file name in run_path directory with a given extension and an index
    """
    # Create target directory & all intermediate directories if don't exists
    try:
        os.makedirs(run_path)
        print("Directory " , run_path ,  " created ")
    except FileExistsError:
        pass
    return run_path + '/' + get_full_name(name_prefix,es) \
        + fold_suffix(fold) + extension 

def csv_filename(name_prefix, es = None, fold = None):
    return filename(name_prefix, es, fold, '.csv')

def data_filename(name_prefix, es = None, fold = None):
    return filename(name_prefix, es, fold, '.npy')

def json_filename(name_prefix, es):
    return filename(name_prefix, es, extension='.json')

def pickle_filename(name_prefix, es = None, fold = None):
    return filename(name_prefix, es, fold, '.pkl')

def picture_filename(name_prefix, es):
    return filename(name_prefix, es, extension='.svg')

def learned_data_filename(suffix, es, fold):
    prefix = learning_data_learned + suffix + data_suffix
    return data_filename(prefix, es, fold)

def learned_labels_filename(suffix, es, fold):
    prefix = learning_data_learned + suffix + labels_suffix
    return data_filename(prefix, es, fold)

def seed_data_filename():
    return data_filename(learning_data_seed + data_suffix)

def seed_labels_filename():
    return data_filename(learning_data_seed + labels_suffix)

def classifier_filename(name_prefix, es, fold):
    return filename(name_prefix + classifier_suffix, es, fold)

def decoder_filename(name_prefix, es, fold):
    return filename(name_prefix + decoder_suffix, es, fold)



###### TO BE MODIFIED #####

def stats_name(experiment = -1):
    return get_name_w_suffix(stats_prefix, experiment >= EXP_1, experiment, experiment_suffix)

def recog_filename(name_prefix, es, fold):
    return csv_filename(name_prefix, es, fold)

def mean_idx(m):
    return m

def std_idx(m):
    return m+1

def padding_cropping(data, n_frames):
    frames, _  = data.shape
    df = frames - n_frames
    if df < 0:
        return []
    elif df == 0:
        return [data]
    else:
        features = []
        for i in range(df+1):
            features.append(data[i:i+n_frames,:])
        return features

def get_data_in_range(data, i, j):
    if j >= i:
        return data[i:j]
    else:
        return np.concatenate((data[i:], data[:j]), axis=0)
