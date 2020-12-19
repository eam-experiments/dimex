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

# Directory where all results are stored.
run_path = './runs'
idx_digits = 3

def filename(s, idx = None, extension = ''):
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
        return run_path + '/' + s + '-' + str(idx).zfill(3) + extension


def csv_filename(s, idx = None):
    """ Returns a file name for csv(i) in run_path directory
    """
    return filename(s, idx, '.csv')


def data_filename(s, idx = None):
    """ Returns a file name for csv(i) in run_path directory
    """
    return filename(s, idx, '.npy')


def picture_filename(s, idx = None):
    """ Returns a file name for csv(i) in run_path directory
    """
    return filename(s, idx, '.svg')


def model_filename(s, idx = None):
    return filename(s, idx)


def image_filename(dir, stage, idx, label, suffix = ''):
    image_path = run_path + '/images/' + dir + '/' + 'stage_' + str(stage) + '/'

    try:
        os.makedirs(image_path)
        print("Directory " , image_path ,  " created ")
    except FileExistsError:
        pass

    image_path += str(label) + '_' + str(idx).zfill(5)  + suffix + '.png'
    return image_path


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


original_suffix = '-original'


def original_image_filename(dir, stage, idx, label):
    return image_filename(dir, stage, idx, label, original_suffix)


def produced_image_filename(dir, stage, idx, label):
    return image_filename(dir, stage, idx, label)


def produced_memory_filename(dir, msize, stage, idx, label):
    return memory_filename(dir, msize, stage, idx, label)


# Categories prefixes.
model_name = 'model'
stats_model_name = 'model_stats'
data_name = 'data'
features_name = 'features'
labels_name = 'labels'
memories_name = 'memories'

# Categories suffixes.
training_suffix = '-training'
filling_suffix = '-filling'
testing_suffix = '-testing'
memory_suffix = '-memories'

testing_directory = 'test'
memories_directory = 'memories'

training_stages = 10        # 0.10 of data.
model_epochs = 10

nn_training_percent = 0.57  # 0.10 + 0.57 = 0.67
am_filling_percent = 0.33   # 0.67 + 0.33 = 1.0

domain = 64

n_jobs = 4
n_labels = 10
labels_per_memory = [0, 1, 2]

all_labels = list(range(n_labels))

precision_idx = 0
recall_idx = 1
n_measures = 2
def mean_idx(m):
    return m

def std_idx(m):
    return m+1

no_response_idx = 2
no_correct_response_idx = 3
no_correct_chosen_idx = 4
correct_response_idx = 5
mean_responses_idx = 6

n_behaviours = 7

memory_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
memory_fills = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 100.0]
partial_ideal_memory_size = 32
full_ideal_memory_size = 64
