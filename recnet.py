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
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, GRU, LSTM, SimpleRNN, Dropout, Dense, AveragePooling1D, \
    MaxPool1D, Bidirectional, LayerNormalization, Reshape, RepeatVector, TimeDistributed
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
from joblib import Parallel, delayed
import png

import constants
import dimex

n_frames = constants.n_frames
n_mfcc = constants.mfcc_numceps
encoder_nlayers = 5     # The number of layers defined in get_encoder.
batch_size = 2048
epochs = 300
patience = 10

TOP_SIDE = 0
BOTTOM_SIDE = 1
LEFT_SIDE = 2
RIGHT_SIDE = 3
VERTICAL_BARS = 4
HORIZONTAL_BARS = 5

truly_training_percentage = 0.80

def get_weights(labels):
    class_weights = compute_class_weight('balanced', classes=constants.all_labels, y=labels)
    return dict(enumerate(class_weights))


def get_encoder(input_data):
    in_dropout=0.2
    out_dropout=0.4
    # Recurrent encoder
    rnn = Bidirectional(GRU(4*n_mfcc, return_sequences=True, dropout=in_dropout))(input_data)
    drop = Dropout(out_dropout)(rnn)
    # rnn = GRU(4*n_mfcc, return_sequences=True, dropout=in_dropout)(drop)
    # drop = Dropout(out_dropout)(rnn)
    rnn = Bidirectional(GRU(constants.domain //2, dropout=in_dropout))(drop)
    drop = Dropout(out_dropout)(rnn)
    norm = LayerNormalization()(drop)
    return norm


def get_decoder(encoded):
    repeat_1 = RepeatVector(n_frames)(encoded)
    gru_1 = Bidirectional(GRU(constants.domain, activation='relu', return_sequences=True))(repeat_1)
    drop_1 = Dropout(0.4)(gru_1)
    gru_2 = Bidirectional(GRU(constants.domain // 2, activation='relu', return_sequences=True))(drop_1)
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
        self.prev_loss = float('inf')
        # best_weights to store the weights at which the loss crossing occurs.
        self.best_weights = None
        self.start = max(epochs // 20, 3)
        self.wait = 0

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited since loss crossed val_loss.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        accuracy = logs.get('accuracy')
        val_accuracy = logs.get('val_accuracy')

        if (epoch < self.start) or ((val_loss < self.prev_loss) and (val_loss < loss) and (accuracy < val_accuracy)) :
            self.wait = 0
            self.prev_loss = val_loss
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


def train_classifier(prefix, es):
    lds = dimex.LearnedDataSet(es)
    confusion_matrix = np.zeros((constants.n_labels, constants.n_labels))
    histories = []
    for fold in range(constants.n_folds):
        training_data, training_labels = lds.get_training_data(fold)
        testing_data, testing_labels = lds.get_testing_data(fold)
        truly_training = int(len(training_labels)*truly_training_percentage)

        validation_data = training_data[truly_training:]
        validation_labels = training_labels[truly_training:]
        training_data = training_data[:truly_training]
        training_labels = training_labels[:truly_training]

        weights = get_weights(training_labels)        
        training_labels = to_categorical(training_labels)
        validation_labels = to_categorical(validation_labels)
        testing_labels = to_categorical(testing_labels)

        input_data = Input(shape=(n_frames, n_mfcc))
        encoded = get_encoder(input_data)
        classified = get_classifier(encoded)
        model = Model(inputs=input_data, outputs=classified)
        model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics='accuracy')
        model.summary()
        history = model.fit(training_data,
                training_labels,
                batch_size=batch_size,
                epochs=epochs,
                class_weight=weights, # Only supported for single output models.
                validation_data= (validation_data, validation_labels),
                callbacks=[EarlyStoppingAtLossCrossing(patience)],
                verbose=2)
        histories.append(history)
        history = model.evaluate(testing_data, testing_labels, return_dict=True)
        predicted_labels = model.predict(testing_data)
        confusion_matrix += tf.math.confusion_matrix(np.argmax(testing_labels, axis=1), 
            np.argmax(predicted_labels, axis=1), num_classes=constants.n_labels)
        histories.append(history)
        model.save(constants.classifier_filename(prefix, es, fold))
    confusion_matrix = confusion_matrix.numpy()
    totals = confusion_matrix.sum(axis=1).reshape(-1,1)
    return histories, confusion_matrix/totals


def obtain_features(model_prefix, features_prefix, labels_prefix, data_prefix,
            training_percentage, am_filling_percentage, experiment, stage):
    """ Generate features for images.
    
    Uses the previously trained neural networks for generating the features corresponding
    to the images.
    """
    (data, labels) = get_data(experiment, stage)
    total = len(data)
    step = total/constants.n_folds

    # Amount of data used for training the networks
    training_size = int(total*training_percentage)
    filling_size = int(total*am_filling_percentage)
    testing_size = total - training_size - filling_size

    histories = []
    for fold in range(constants.n_folds):
        i = int(fold*step)
        j = (i+training_size) % total
        training_data = constants.get_data_in_range(data, i, j)
        training_labels = constants.get_data_in_range(labels, i, j)

        k = (j+filling_size) % total
        filling_data = constants.get_data_in_range(data, j, k)
        filling_labels = constants.get_data_in_range(labels, j, k)

        l = (k+testing_size) % total
        testing_data = constants.get_data_in_range(data, k, l)
        testing_labels = constants.get_data_in_range(labels, k, l)

        # Recreate the exact same model, including its weights and the optimizer
        model = tf.keras.models.load_model(constants.classifier_filename(model_prefix, fold))

        # Drop the autoencoder and the last layers of the full connected neural network part.
        # classifier = Model(model.input, model.output[0])
        classifier = Model(model.input, model.output)
        no_hot = to_categorical(testing_labels)
        classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
        history = classifier.evaluate(testing_data, no_hot, batch_size=100, verbose=1, return_dict=True)
        histories.append(history)
        model = Model(classifier.input, classifier.layers[-4].output)
        # model.summary()

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
            data_fn = constants.data_filename(data_prefix+suffix, fold, experiment, stage)
            features_fn = constants.data_filename(features_prefix+suffix, fold, experiment, stage)
            labels_fn = constants.data_filename(labels_prefix+suffix, fold, experiment, stage)

            d, f, l = dict[suffix]
            np.save(data_fn, d)
            np.save(features_fn, f)
            np.save(labels_fn, l)    
    return histories


def train_decoder(filename, experiment):
    (labels, _) = get_data(experiment)
    total = len(labels)
    step = total/constants.n_folds
    histories = []
    for fold in range(constants.n_folds):
        suffix = constants.training_suffix
        training_features_filename = constants.features_name(experiment) + suffix        
        training_features_filename = constants.data_filename(training_features_filename, fold)

        suffix = constants.filling_suffix
        filling_features_filename = constants.features_name(experiment) + suffix        
        filling_features_filename = constants.data_filename(filling_features_filename, fold)

        suffix = constants.testing_suffix
        testing_features_filename = constants.features_name(experiment) + suffix        
        testing_features_filename = constants.data_filename(testing_features_filename, fold)

        training_features = np.load(training_features_filename)
        filling_features = np.load(filling_features_filename)
        testing_features = np.load(testing_features_filename)
        testing_features = np.concatenate((filling_features, testing_features), axis=0)

        truly_training = int(len(training_features)*truly_training_percentage)
        validation_data = training_features[truly_training:]
        training_data = training_features[:truly_training]
        testing_data = testing_features

        i = int(fold*step)
        j = (i+len(training_data)) % total
        training_labels = constants.get_data_in_range(labels,i, j)
        k = (j+len(validation_data)) % total
        validation_labels = constants.get_data_in_range(labels,j, k)
        l = (k+len(testing_data)) % total
        testing_labels = constants.get_data_in_range(labels, k, l)

        input_data = Input(shape=(constants.domain))
        decoded = get_decoder(input_data)
        model = Model(inputs=input_data, outputs=decoded)
        model.compile(loss='mean_squared_error', optimizer='adam', metrics='accuracy')
        model.summary()
        history = model.fit(training_data,
                training_labels,
                batch_size=batch_size,
                epochs=epochs,
                validation_data= (validation_data, validation_labels),
                callbacks=[EarlyStoppingAtLossCrossing(patience)],
                verbose=2)
        histories.append(history)
        history = model.evaluate(testing_data,
            (testing_labels, testing_data),return_dict=True)
        histories.append(history)
        model.save(constants.decoder_filename(filename, fold, experiment))
    return histories


def train_next_encoder(training_data, training_labels,
    validation_data, validation_labels, model_prefix, fold, tolerance, stage):
    input_data = Input(shape=(n_frames, n_mfcc))
    weights = get_weights(training_labels)
    training_labels = to_categorical(training_labels, constants.n_labels, dtype='int')
    validation_labels = to_categorical(validation_labels, constants.n_labels, dtype='int')
    encoded = get_encoder(input_data)
    classified = get_classifier(encoded)
    model = Model(inputs=input_data, outputs=classified)
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics='accuracy')
    model.summary()
    history = model.fit(training_data,
            training_labels,
            batch_size = batch_size,
            epochs = epochs,
            class_weight = weights, # Only supported for single output models.
            validation_data = (validation_data, validation_labels),
            callbacks=[EarlyStoppingAtLossCrossing(patience)],
            verbose=2)
    model.save(constants.classifier_filename(model_prefix, fold, tolerance, stage))
    model = Model(inputs=input_data, outputs=encoded)
    return model, history

def train_next_decoder(training_data, training_labels,
    validation_data, validation_labels, model_prefix, fold, tolerance, stage):
    input_data = Input(shape=(constants.domain))
    decoded = get_decoder(input_data)
    model = Model(inputs=input_data, outputs=decoded)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics='accuracy')
    model.summary()
    history = model.fit(training_data,
            training_labels,
            batch_size = batch_size,
            epochs = epochs,
            validation_data = (validation_data, validation_labels),
            callbacks=[EarlyStoppingAtLossCrossing(patience)],
            verbose=2)
    model.save(constants.decoder_filename(model_prefix, fold, tolerance=tolerance, stage=stage))
    return model, history

def train_next_network(training_data, training_labels,
    validation_data, validation_labels, filling_data, filling_labels,
    model_prefix, fold, tolerance, stage):
    histories = []
    encoder, history = train_next_encoder(training_data, training_labels,
        validation_data, validation_labels, model_prefix, fold, tolerance, stage)
    histories.append(history)
    training_features = encoder.predict(training_data)
    validation_features = encoder.predict(validation_data)
    filling_features = encoder.predict(filling_data)
    _, history = train_next_decoder(training_features, training_data,
        validation_features, validation_data, model_prefix, fold, tolerance, stage)
    histories.append(history)
    return filling_features, histories





class SplittedNeuralNetwork:
    def __init__ (self, prefix, fold, tolerance, stage):
        model_filename = constants.classifier_filename(prefix, fold, tolerance, stage)
        classifier = tf.keras.models.load_model(model_filename)
        model_filename = constants.decoder_filename(prefix, fold, tolerance=tolerance, stage=stage)
        self.decoder = tf.keras.models.load_model(model_filename)

        input_enc = Input(shape=(n_frames, n_mfcc))
        input_cla = Input(shape=(constants.domain))
        encoded = get_encoder(input_enc)
        classified = get_classifier(input_cla)

        self.encoder = Model(inputs = input_enc, outputs = encoded)
        # self.encoder.summary()
        self.classifier = Model(inputs = input_cla, outputs = classified)
        # self.classifier.summary()
        # self.decoder.summary()

        for from_layer, to_layer in zip(classifier.layers[1:encoder_nlayers+1], self.encoder.layers[1:]):
            to_layer.set_weights(from_layer.get_weights())

        for from_layer, to_layer in zip(classifier.layers[encoder_nlayers+1:], self.classifier.layers[1:]):
            to_layer.set_weights(from_layer.get_weights())


def process_sample(sample: dimex.TaggedAudio, snnet: SplittedNeuralNetwork, decode):
    features = snnet.encoder.predict(sample.segments)
    labels = snnet.classifier.predict(features)
    sample.features = features
    sample.net_labels = np.argmax(labels, axis=1)
    if decode:
        sample.net_segments = snnet.decoder.predict(features)
    return sample


def process_samples(samples, prefix, fold, tolerance, stage, decode=False):
    n = 0
    print('Processing samples with neural network.')

    snnet = SplittedNeuralNetwork(prefix, fold, tolerance, stage)
    new_samples = []
    for sample in samples: 
        new_sample = process_sample(sample, snnet, decode)
        new_samples.append(new_sample)
        n += 1
        constants.print_counter(n,100,10)
    return new_samples


def reprocess_samples(samples, prefix, fold, tolerance, stage, decode=False):
    n = 0
    print('Reprocessing samples with neural network.')

    snnet = SplittedNeuralNetwork(prefix, fold, tolerance, stage)
    for sample in samples: 
        features = np.array(sample.ams_features)
        sample.ams_segments = snnet.decoder.predict(features)
        n += 1
        constants.print_counter(n,100,10)
    return samples
