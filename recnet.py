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
from python_speech_features.base import mfcc
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, GRU, Dropout, Dense, AveragePooling1D, \
    MaxPool1D, Bidirectional, LayerNormalization, Reshape, RepeatVector, TimeDistributed
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
from joblib import Parallel, delayed
import png

import constants
from dimex_sampler import TaggedAudio

n_frames = constants.n_frames
n_mfcc = constants.mfcc_numceps
batch_size = 2048
epochs = 300
patience = 5

TOP_SIDE = 0
BOTTOM_SIDE = 1
LEFT_SIDE = 2
RIGHT_SIDE = 3
VERTICAL_BARS = 4
HORIZONTAL_BARS = 5

truly_training_percentage = 0.80

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


def reshape(data, n_frames):
    """ Restores the flatten matrices (frames, n_mfcc) and pads them vertically.
    """

    reshaped = []
    for d in data:
        frames = d.size // n_mfcc
        d = d.reshape((frames, n_mfcc))
        d = constants.padding_cropping(d,n_frames)
        reshaped.append(d)

    return np.array(reshaped, dtype=np.float32)



def get_data(experiment, occlusion = None, bars_type = None, one_hot = False):

    # Load DIMEX-100 labels
    labels = np.load('Features/rand_Y.npy')

    all_labels = np.zeros(labels.shape)
    for i in range(all_labels.size):
        label = labels[i]
        idx = constants.phns_to_labels[label]
        all_labels[i] = idx

    # Load DIMEX-100 features and labels
    all_data = np.load('Features/rand_X.npy', allow_pickle=True) 

    all_data = reshape(all_data, n_frames)

    # all_data = add_noise(all_data, experiment, occlusion, bars_type)
    # minimum = all_data.min()
    # maximum = all_data.max()
    # all_data = (all_data - minimum)/ (maximum-minimum)

    if one_hot:
        # Changes labels to binary rows. Each label correspond to a column, and only
        # the column for the corresponding label is set to one.
        all_labels = to_categorical(all_labels)

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
    gru_1 = Bidirectional(GRU(constants.domain // 2))(input_data)
    drop_1 = Dropout(0.4)(gru_1)
    norm = LayerNormalization()(drop_1)

    # Produces an array of size equal to constants.domain.
    return norm


def get_decoder(encoded):
    repeat_1 = RepeatVector(n_frames)(encoded)
    gru_1 = GRU(constants.domain, activation='relu', return_sequences=True)(repeat_1)
    drop_1 = Dropout(0.4)(gru_1)
    gru_2 = GRU(constants.domain // 2, activation='relu', return_sequences=True)(drop_1)
    drop_2 = Dropout(0.4)(gru_2)
    output_mfcc = TimeDistributed(Dense(n_mfcc), name='autoencoder')(drop_2)

    # Produces an image of same size and channels as originals.
    return output_mfcc


def get_classifier(encoded, output_bias = None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    dense_1 = Dense(constants.domain, activation='relu')(encoded)
    drop = Dropout(0.4)(dense_1)
    classification = Dense(constants.n_labels, activation='softmax',
         bias_initializer=output_bias, name='classification')(drop)

    return classification


class EarlyStoppingAtLossCrossing(Callback):
    """ Stop training when the loss gets lower than val_loss.

        Arguments:
            patience: Number of epochs to wait after condition has been hit.
            After this number of no reversal, training stops.
            It starts working after 10% of epochs have taken place.
    """

    def __init__(self, patience=0):
        super(EarlyStoppingAtLossCrossing, self).__init__()
        self.patience = patience
        # best_weights to store the weights at which the loss crossing occurs.
        self.best_weights = None
        self.start = epochs // 10

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited since loss crossed val_loss.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')

        if (epoch < self.start) or (val_loss < loss):
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))


def train_networks(training_percentage, filename, experiment):

    stages = constants.training_stages

    (data, labels) = get_data(experiment)

    total = len(data)
    step = total/stages

    # Amount of training data, from which a percentage is used for
    # validation.
    training_size = int(total*training_percentage)

    n = 0
    histories = []
    for k in range(stages):
        i = k*step
        j = int(i + training_size) % total
        i = int(i)

        if j > i:
            training_data = data[i:j]
            training_labels = labels[i:j]
            testing_data = np.concatenate((data[0:i], data[j:total]), axis=0)
            testing_labels = np.concatenate((labels[0:i], labels[j:total]), axis=0)
        else:
            training_data = np.concatenate((data[i:total], data[0:j]), axis=0)
            training_labels = np.concatenate((labels[i:total], labels[0:j]), axis=0)
            testing_data = data[j:i]
            testing_labels = labels[j:i]

        truly_training = int(training_size*truly_training_percentage)

        validation_data = training_data[truly_training:]
        validation_labels = training_labels[truly_training:]
        training_data = training_data[:truly_training]
        training_labels = training_labels[:truly_training]

        weights, bias = get_weights_bias(training_labels)
        training_labels = to_categorical(training_labels)
        validation_labels = to_categorical(validation_labels)
        testing_labels = to_categorical(testing_labels)
        
        input_data = Input(shape=(n_frames, n_mfcc))
        encoded = get_encoder(input_data)
        classified = get_classifier(encoded, bias)
        decoded = get_decoder(encoded)
        model = Model(inputs=input_data, outputs=[classified, decoded])
        # model = Model(inputs=input_data, outputs=classified)

        model.compile(loss=['categorical_crossentropy', 'mean_squared_error'],
                    optimizer='adam',
                    metrics='accuracy')

        model.summary()

        history = model.fit(training_data,
            (training_labels, training_data),
                batch_size=batch_size,
                epochs=epochs,
                validation_data= (validation_data,
                    {'classification': validation_labels, 'autoencoder': validation_data}),
                callbacks=[EarlyStoppingAtLossCrossing(patience)],
                verbose=2)

        histories.append(history)
        history = model.evaluate(testing_data,
            (testing_labels, testing_data),return_dict=True)
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
    stages = constants.training_stages
    step = total/stages

    # Amount of data used for training the networks
    trdata = int(total*training_percentage)

    # Amount of data used for testing memories
    tedata = step

    n = 0
    histories = []
    for k in range(stages):
        i = k*step
        j = int(i + tedata) % total
        i = int(i)

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
        classifier = Model(model.input, model.output[0])
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


class SplittedNeuralNetwork:
    def __init__ (self, n):
        model_filename = constants.model_filename(constants.model_name, n)
        model = tf.keras.models.load_model(model_filename)
        classifier = Model(model.input, model.output[0])
        autoencoder = Model(model.input, model.output[1])

        input_enc = Input(shape=(n_frames, n_mfcc))
        input_cla = Input(shape=(constants.domain, ))
        input_dec = Input(shape=(constants.domain, ))
        encoded = get_encoder(input_enc)
        classified = get_classifier(input_cla)
        decoded = get_decoder(input_dec)

        self.encoder = Model(inputs = input_enc, outputs = encoded)
        self.encoder.summary()
        self.classifier = Model(inputs = input_cla, outputs = classified)
        self.classifier.summary()
        self.decoder = Model(inputs=input_dec, outputs=decoded)
        self.decoder.summary()

        for from_layer, to_layer in zip(classifier.layers[1:4], self.encoder.layers[1:]):
            to_layer.set_weights(from_layer.get_weights())

        for from_layer, to_layer in zip(classifier.layers[4:], self.classifier.layers[1:]):
            to_layer.set_weights(from_layer.get_weights())

        for from_layer, to_layer in zip(autoencoder.layers[4:], self.decoder.layers[1:]):
            to_layer.set_weights(from_layer.get_weights())


def process_sample(sample, snnet):
    features = snnet.encoder.predict(sample.segments)
    labels = snnet.classifier.predict(features)    
    sample.features = features
    sample.net_labels = np.argmax(labels, axis=1)
    return sample


def process_samples(samples, fold):
    snnet = SplittedNeuralNetwork(fold)

    new_samples = []
    for sample in samples: 
        new_sample = process_sample(sample, snnet)
        new_samples.append(new_sample)
    return new_samples
