#
# Licensed under the Apache License 2.0 (the "License");
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
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Dropout, Dense, MaxPooling2D, \
    Flatten, LayerNormalization, Reshape, RepeatVector, TimeDistributed
from tensorflow.keras.layers.experimental.preprocessing import Resizing
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
from joblib import Parallel, delayed
import png

import constants
import dimex

img_rows = 33
fit_img_rows = 32
img_columns = 25
fit_img_columns = 32
img_colors = 1
n_frames = constants.n_frames
batch_size = 2048
epochs = 1000
patience = 5
truly_training_percentage = 0.80


#######################################################################
# Getting data code

def get_data(experiment, one_hot = False):
    # Load DIMEX-100 labels
    all_labels = np.load('Features/labels.npy')
    # Load DIMEX-100 features and labels
    all_data = np.load('Features/data.npy') 
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

def vgg_block(n_conv, parameters, input_layer, k_size=3, dropout=0.4, pool_size=(2,2), first = False):
    input = input_layer
    if first:
        input = Resizing(fit_img_rows,fit_img_rows)(input_layer)
        # input = LayerNormalization()(input)
    for i in range(n_conv):
        input = Conv2D(parameters,kernel_size=k_size, activation='relu', kernel_initializer='he_uniform', padding='same')(input)
    pool = MaxPooling2D(pool_size)(input)
    drop = Dropout(dropout)(pool)
    return drop

def get_encoder(input_img):
    domain = constants.domain
    vgg_0 = vgg_block(2, img_colors, input_img, first = True)
    vgg_1 = vgg_block(2, domain//32, vgg_0)
    vgg_2 = vgg_block(2, domain//16, vgg_1)
    code = Flatten()(vgg_2)
    return code


def get_decoder(encoded):
    ini_rows = img_rows//8
    ini_cols = img_columns//8
    dense = Dense(units=ini_rows*ini_cols*constants.domain//2, activation='relu')(encoded)
    reshape = Reshape((ini_rows, ini_cols, constants.domain//2))(dense)
    drop_0 = Dropout(0.4)(reshape)
    trans_1 = Conv2DTranspose(constants.domain//4, kernel_size=3, strides=2,
        padding='same', activation='relu')(drop_0)
    drop_1 = Dropout(0.4)(trans_1)
    trans_2 = Conv2DTranspose(constants.domain//8, kernel_size=3, strides=2,
        padding='same', activation='relu')(drop_1)
    drop_2 = Dropout(0.4)(trans_2)
    output_img = Conv2DTranspose(img_colors, kernel_size=3, strides=2,
        activation='sigmoid', padding='same', name='autoencoder')(drop_2)
    # Produces an image of same size and channels as originals.
    return output_img


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
        
        input_data = Input(shape=(img_rows, img_columns, img_colors))
        encoded = get_encoder(input_data)
        classified = get_classifier(encoded, bias)
        decoded = get_decoder(encoded)
        # model = Model(inputs=input_data, outputs=[classified, decoded])
        model = Model(inputs=input_data, outputs=classified)
        model.summary()

        # model.compile(loss=['categorical_crossentropy', 'mean_squared_error'],
        #             optimizer='adam',
        #             metrics='accuracy')
        model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics='accuracy')


        history = model.fit(training_data,
            training_labels,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(validation_data,
                    validation_labels),
                callbacks=[EarlyStoppingAtLossCrossing(patience)],
                verbose=2)

        histories.append(history)
        history = model.evaluate(testing_data,
            testing_labels,return_dict=True)
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


def process_sample(sample: dimex.TaggedAudio, snnet: SplittedNeuralNetwork):
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
