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

import sys
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, GRU, Dropout, Flatten, Dense, \
    Bidirectional, LayerNormalization, Reshape, Conv2DTranspose
from tensorflow.keras.utils import to_categorical
from joblib import Parallel, delayed
import png

import constants

n_frames = 0
n_mfcc = 26

TOP_SIDE = 0
BOTTOM_SIDE = 1
LEFT_SIDE = 2
RIGHT_SIDE = 3
VERTICAL_BARS = 4
HORIZONTAL_BARS = 5

def print_error(*s):
    print('Error:', *s, file = sys.stderr)


#######################################################################
# Noise related code.

def add_side_occlusion(data, side_hidden, occlusion):
    noise_value = 0
    mid_row = int(round(n_frames*occlusion))
    mid_col = int(round(n_mfcc*occlusion))
    origin = (0, 0)
    end = (0, 0)

    if side_hidden == TOP_SIDE:
        origin = (0, 0)
        end = (mid_row, n_mfcc)
    elif side_hidden ==  BOTTOM_SIDE:
        origin = (mid_row, 0)
        end = (n_frames, n_mfcc)
    elif side_hidden == LEFT_SIDE:
        origin = (0, 0)
        end = (n_frames, mid_col)
    elif side_hidden == RIGHT_SIDE:
        origin = (0, mid_col)
        end = (n_frames, n_mfcc)

    for image in data:
        n, m = origin
        end_n, end_m = end

        for i in range(n, end_n):
            for j in range(m, end_m):
                image[i,j] = noise_value

    return data


def add_bars_occlusion(data, bars, n):
    pattern = constants.bar_patterns[n]

    if bars == VERTICAL_BARS:
        for image in data:
            for j in range(n_mfcc):
                image[:,j] *= pattern[j]     
    else:
        for image in data:
            for i in range(n_frames):
                image[i,:] *= pattern[i]

    return data


def add_noise(data, experiment, occlusion = 0, bars_type = None):
    # data is assumed to be a numpy array of shape (N, img_rows, img_columns)

    if experiment < constants.EXP_5:
        return data
    elif experiment < constants.EXP_9:
        sides = {constants.EXP_5: TOP_SIDE,  constants.EXP_6: BOTTOM_SIDE,
                 constants.EXP_7: LEFT_SIDE, constants.EXP_8: RIGHT_SIDE }
        return add_side_occlusion(data, sides[experiment], occlusion)
    else:
        bars = {constants.EXP_9: VERTICAL_BARS,  constants.EXP_10: HORIZONTAL_BARS}
        return add_bars_occlusion(data, bars[experiment], bars_type)


#######################################################################
# Getting data code

def max_frames(data):
    """ Calculates maximum number of feature frames.

    Assumes features come as a flatten matrix of (frames, n_mfcc).
    """
    maximum = 0
    for d in data:
        s = d.size // n_mfcc
        if s > maximum:
            maximum = s
    return maximum


def padding(data, max_frames):

    frames, _  = data.shape
    df = max_frames - frames
    if df == 0:
        return data
    else:
        top_padding = df // 2
        bottom_padding = df - top_padding
        return np.pad(data, ((top_padding, bottom_padding),(0,0)),
            'constant', constant_values=((0,0),(0,0)))


def reshape(data, max_frames):
    """ Restores the flatten matrices (frames, n_mfcc) and pads them vertically.
    """

    reshaped = []
    for d in data:
        frames = d.size // n_mfcc
        d = d.reshape((frames, n_mfcc))
        d = padding(d,max_frames)
        reshaped.append(d)

    return np.array(reshaped, dtype=np.float32)



def get_data(experiment, occlusion = None, bars_type = None, one_hot = False):

    global n_frames

    # Load dictionary with labels as keys (structured array) 
    label_idx = np.load('Features/media.npy', allow_pickle=True).item()

    if constants.n_labels != len(label_idx):
        print_error("Inconsistent number of labels: ", n)
        exit(1)
    
    # Load DIMEX-100 labels
    labels = np.load('Features/feat_Y.npy')

    # Replaces actual labels (letter codes for sounds) by
    # numbers from 0 to N-1, where N is the number of labels.
    idx = 0
    for label in label_idx:
        label_idx[label] = idx
        idx += 1

    all_labels = np.zeros(labels.shape)
    for i in range(all_labels.size):
        label = labels[i]
        idx = label_idx[label]
        all_labels[i] = idx

    # Load DIMEX-100 features and labels
    all_data = np.load('Features/feat_X.npy', allow_pickle=True) 

    n_frames = max_frames(all_data) 
    all_data = reshape(all_data, n_frames)
    # all_data = add_noise(all_data, experiment, occlusion, bars_type)
    minimum = all_data.min()
    maximum = all_data.max()
    all_data = (all_data - minimum)/ maximum

    # if one_hot:
    #     # Changes labels to binary rows. Each label correspond to a column, and only
    #     # the column for the corresponding label is set to one.
    #     all_labels = to_categorical(all_labels)

    return (all_data, all_labels)


def get_weights_bias(labels):
    frequency = {}
    for label in labels:
        if label in frequency:
            frequency[label] += 1
        else:
            frequency[label] = 1

    total = len(labels)
    maximum = 0
    for label in frequency:
        if maximum < frequency[label]:
            maximum = frequency[label]

    weights = {}
    bias = np.zeros(len(frequency))
    for label in frequency:
        weights[label] = maximum*(1.0/frequency[label])
        bias[int(label)] = math.log(frequency[label]/total)


    return weights, bias


def get_encoder(input_data):

    # Recurrent encoder
    gru_1 = Bidirectional(GRU(constants.domain, return_sequences=True))(input_data)
    drop_1 = Dropout(0.4)(gru_1)
    gru_2 = Bidirectional(GRU(constants.domain // 2))(drop_1) 
    norm = LayerNormalization()(gru_2)

    # Produces an array of size equal to constants.domain.
    code = norm

    return code


def get_decoder(encoded):
    dense = Dense(units=7*7*32, activation='relu', input_shape=(64, ))(encoded)
    reshape = Reshape((7, 7, 32))(dense)
    trans_1 = Conv2DTranspose(64, kernel_size=3, strides=2,
        padding='same', activation='relu')(reshape)
    drop_1 = Dropout(0.4)(trans_1)
    trans_2 = Conv2DTranspose(32, kernel_size=3, strides=2,
        padding='same', activation='relu')(drop_1)
    drop_2 = Dropout(0.4)(trans_2)
    output_img = Conv2D(1, kernel_size=3, strides=1,
        activation='sigmoid', padding='same', name='autoencoder')(drop_2)

    # Produces an image of same size and channels as originals.
    return output_img


def get_classifier(encoded, output_bias = None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    dense_1 = Dense(constants.domain*2, activation='relu')(encoded)
    drop = Dropout(0.4)(dense_1)
    classification = Dense(constants.n_labels, activation='softmax',
         bias_initializer=output_bias, name='classification')(drop)

    return classification


def train_networks(training_percentage, filename, experiment):

    EPOCHS = constants.model_epochs
    stages = constants.training_stages

    (data, labels) = get_data(experiment)

    total = len(data)
    step = int(total/stages)

    # Amount of testing data
    atd = total - int(total*training_percentage)

    n = 0
    histories = []
    for i in range(0, total, step):
        j = (i + atd) % total

        if j > i:
            testing_data = data[i:j]
            testing_labels = labels[i:j]

            training_data = np.concatenate((data[0:i], data[j:total]), axis=0)
            training_labels = np.concatenate((labels[0:i], labels[j:total]), axis=0)
        else:
            testing_data = np.concatenate((data[i:total], data[0:j]), axis=0)
            testing_labels = np.concatenate((labels[i:total], labels[0:j]), axis=0)
            training_data = data[j:i]
            training_labels = labels[j:i]

        weights, bias = get_weights_bias(training_labels)
        input_data = Input(shape=(n_frames, n_mfcc))
        encoded = get_encoder(input_data)
        classified = get_classifier(encoded, bias)
        # decoded = get_decoder(encoded)
        # model = Model(inputs=input_data, outputs=[classified, decoded])
        model = Model(inputs=input_data, outputs=classified)

        # model.compile(loss=['categorical_crossentropy', 'binary_crossentropy'],
        #             optimizer='adam',
        #             metrics='accuracy')
        model.compile(loss='SparseCategoricalCrossentropy', optimizer='adam',
                    metrics='accuracy')

        model.summary()

        # history = model.fit(training_data,
        #         (training_labels, training_data),
        #         batch_size=100,
        #         epochs=EPOCHS,
        #         validation_data= (testing_data,
        #             {'classification': testing_labels, 'autoencoder': testing_data}),
        #         verbose=2)
        history = model.fit(training_data, training_labels,
                batch_size=2048, epochs=EPOCHS,
                validation_data= (testing_data,testing_labels),
                class_weight=weights, verbose=2)

        histories.append(history)
        model.save(constants.model_filename(filename, n))
        n += 1

    return histories


def store_images(original, produced, directory, stage, idx, label):
    original_filename = constants.original_image_filename(directory, stage, idx, label)
    produced_filename = constants.produced_image_filename(directory, stage, idx, label)

    pixels = original.reshape(28,28) * 255
    pixels = pixels.round().astype(np.uint8)
    png.from_array(pixels, 'L;8').save(original_filename)
    pixels = produced.reshape(28,28) * 255
    pixels = pixels.round().astype(np.uint8)
    png.from_array(pixels, 'L;8').save(produced_filename)


def store_memories(labels, produced, features, directory, stage, msize):
    (idx, label) = labels
    produced_filename = constants.produced_memory_filename(directory, msize, stage, idx, label)

    if np.isnan(np.sum(features)):
        pixels = np.full((28,28), 255)
    else:
        pixels = produced.reshape(28,28) * 255
    pixels = pixels.round().astype(np.uint8)
    png.from_array(pixels, 'L;8').save(produced_filename)


def obtain_features(model_prefix, features_prefix, labels_prefix, data_prefix,
            training_percentage, am_filling_percentage, experiment,
            occlusion = None, bars_type = None):
    """ Generate features for images.
    
    Uses the previously trained neural networks for generating the features corresponding
    to the images. It may introduce occlusions.
    """
    (data, labels) = get_data(experiment, occlusion, bars_type)

    total = len(data)
    step = int(total/constants.training_stages)

    # Amount of data used for training the networks
    trdata = int(total*training_percentage)

    # Amount of data used for testing memories
    tedata = step

    n = 0
    histories = []
    for i in range(0, total, step):
        j = (i + tedata) % total

        if j > i:
            testing_data = data[i:j]
            testing_labels = labels[i:j]
            other_data = np.concatenate((data[0:i], data[j:total]), axis=0)
            other_labels = np.concatenate((labels[0:i], labels[j:total]), axis=0)
            training_data = other_data[:trdata]
            training_labels = other_labels[:trdata]
            filling_data = other_data[trdata:]
            filling_labels = other_labels[trdata:]
        else:
            testing_data = np.concatenate((data[0:j], data[i:total]), axis=0)
            testing_labels = np.concatenate((labels[0:j], labels[i:total]), axis=0)
            training_data = data[j:j+trdata]
            training_labels = labels[j:j+trdata]
            filling_data = data[j+trdata:i]
            filling_labels = labels[j+trdata:i]

        # Recreate the exact same model, including its weights and the optimizer
        model = tf.keras.models.load_model(constants.model_filename(model_prefix, n))

        # Drop the autoencoder and the last layers of the full connected neural network part.
        # classifier = Model(model.input, model.output[0])
        classifier = model
        no_hot = to_categorical(testing_labels)
        classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
        history = classifier.evaluate(testing_data, no_hot, batch_size=100, verbose=1, return_dict=True)
        print(history)
        histories.append(history)
        model = Model(classifier.input, classifier.layers[-4].output)
        model.summary()

        training_features = model.predict(training_data)
        if len(filling_data) > 0:
            filling_features = model.predict(filling_data)
        else:
            r, c = training_features.shape
            filling_features = np.zeros((0, c))
        testing_features = model.predict(testing_data)

        dict = {
            constants.training_suffix: (training_data, training_features, training_labels),
            constants.filling_suffix : (filling_data, filling_features, filling_labels),
            constants.testing_suffix : (testing_data, testing_features, testing_labels)
            }

        for suffix in dict:
            data_fn = constants.data_filename(data_prefix+suffix, n)
            features_fn = constants.data_filename(features_prefix+suffix, n)
            labels_fn = constants.data_filename(labels_prefix+suffix, n)

            d, f, l = dict[suffix]
            np.save(data_fn, d)
            np.save(features_fn, f)
            np.save(labels_fn, l)

        n += 1
    
    return histories


def remember(experiment, occlusion = None, bars_type = None, tolerance = 0):
    """ Creates images from features.
    
    Uses the decoder part of the neural networks to (re)create images from features.

    Parameters
    ----------
    experiment : TYPE
        DESCRIPTION.
    occlusion : TYPE, optional
        DESCRIPTION. The default is None.
    tolerance : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    None.

    """

    for i in range(constants.training_stages):
        testing_data_filename = constants.data_name + constants.testing_suffix
        testing_data_filename = constants.data_filename(testing_data_filename, i)
        testing_features_filename = constants.features_name(experiment, occlusion, bars_type) + constants.testing_suffix
        testing_features_filename = constants.data_filename(testing_features_filename, i)
        testing_labels_filename = constants.labels_name + constants.testing_suffix
        testing_labels_filename = constants.data_filename(testing_labels_filename, i)
        memories_filename = constants.memories_name(experiment, occlusion, bars_type, tolerance)
        memories_filename = constants.data_filename(memories_filename, i)
        labels_filename = constants.labels_name + constants.memory_suffix
        labels_filename = constants.data_filename(labels_filename, i)
        model_filename = constants.model_filename(constants.model_name, i)

        testing_data = np.load(testing_data_filename)
        testing_features = np.load(testing_features_filename)
        testing_labels = np.load(testing_labels_filename)
        memories = np.load(memories_filename)
        labels = np.load(labels_filename)
        model = tf.keras.models.load_model(model_filename)

        # Drop the classifier.
        autoencoder = Model(model.input, model.output[1])
        autoencoder.summary()

        # Drop the encoder
        input_mem = Input(shape=(constants.domain, ))
        decoded = get_decoder(input_mem)
        decoder = Model(inputs=input_mem, outputs=decoded)
        decoder.summary()

        for dlayer, alayer in zip(decoder.layers[1:], autoencoder.layers[11:]):
            dlayer.set_weights(alayer.get_weights())

        produced_images = decoder.predict(testing_features)
        n = len(testing_labels)

        Parallel(n_jobs=constants.n_jobs, verbose=5)( \
            delayed(store_images)(original, produced, constants.testing_directory(experiment, occlusion, bars_type), i, j, label) \
                for (j, original, produced, label) in \
                    zip(range(n), testing_data, produced_images, testing_labels))

        total = len(memories)
        steps = len(constants.memory_fills)
        step_size = int(total/steps)

        for j in range(steps):
            print('Decoding memory size ' + str(j) + ' and stage ' + str(i))
            start = j*step_size
            end = start + step_size
            mem_data = memories[start:end]
            mem_labels = labels[start:end]
            produced_images = decoder.predict(mem_data)

            Parallel(n_jobs=constants.n_jobs, verbose=5)( \
                delayed(store_memories)(label, produced, features, constants.memories_directory(experiment, occlusion, bars_type, tolerance), i, j) \
                    for (produced, features, label) in zip(produced_images, mem_data, mem_labels))
