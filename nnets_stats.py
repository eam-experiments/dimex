# Copyright [2021] Luis Alberto Pineda Cortés,
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

"""Nnets-stats: Generate graph of neural networks improvements.

Usage:
  nnets_stats -h | --help
  nnets_stats [-s] [-n] [--json_file=<json_file>] [--dir=<dir>] [--learned=<learned>] [--tolerance=<tolerance>] [--lang=<lang>]

Options:
  -h                        Show this screen.
  -s                        Whether to generate a simple (-s) or detailed graph.
  -n                        Do not plot, only print data.
  --json_file=<json_file>   Analyse a single given JSON file. Ignores all following options.
  --dir=<dir>               Base directory for finding JSON files. [default: runs].
  --learned=<learned>       Index of data learned (original, agreed, ...) [default: 4].
  --tolerance=<tolerance>   Differences allowed between memory and cue. [default: 0].
  --lang=<language>         Chooses Language for graphs [default: en].

Arguments:
  language: en - English
  language: es - Spanish           
"""

from docopt import docopt
import gettext
import sys
import json
import numpy as np
from matplotlib import pyplot as plt
import constants

# Translation
gettext.install('eam', localedir=None, codeset=None, names=None)

# Keys for data
LOSS = 'loss'
VAL = 'val_'
AUTO_MEASURE='root_mean_squared_error'
CLASS_MEASURE='accuracy'
ytitles = {
    AUTO_MEASURE: 'Root Mean Squared Error',
    CLASS_MEASURE: 'Accuracy'}

STAGES = 6

def trplot(a_measure, b_measure, a_label, b_label, epoch, nn):
    fig = plt.figure()
    end = min(len(a_measure), epoch)
    x = np.arange(0,end)
    plt.errorbar(x, a_measure[:end], fmt='b-.', label=a_label)
    plt.errorbar(x, b_measure[:end], fmt='r--,', label=b_label)
    plt.legend(loc=0)

    plt.suptitle(f'Neural net No. {nn}')
    plt.show()
    

def teplot(a_measure, a_label, ymin=0.0, ymax=None):
    fig = plt.figure()
    x = np.arange(0,len(a_measure))
    plt.errorbar(x, a_measure[:epoch], fmt='b-.', label=a_label)
    
    if not (ymax is None):
        plt.ylim(ymin, ymax)
    plt.legend(loc=0)
    plt.suptitle(f'Average results')
    plt.show()
    

def training_stats(data):
    """ Analyse neural nets training data. 
        
        Training stats data is a list of dictionaries with the full
        set of keys declared above.
    """
    n = 0
    for d in data:
        trplot(d[LOSS], d[VAL+LOSS], LOSS, VAL+LOSS,epoch,n)
        trplot(d[CLASS_MEASURE], d[VAL+CLASS_MEASURE], CLASS_MEASURE, VAL+CLASS_MEASURE,epoch,n)
        n += 1


def testing_stats(data, simple):
    """ Analyse neural nets testing data. 
    """
    n = len(data)
    m = {LOSS: [], CLASS_MEASURE: []}
    for d in data:
        m[LOSS].append(d[LOSS])
        m[CLASS_MEASURE].append(d[CLASS_MEASURE])
    
    if simple:
        print(f'{LOSS}: {m[LOSS]}')
        print(f'{CLASS_MEASURE}: {m[CLASS_MEASURE]}')
    else:
        teplot(m[LOSS], LOSS)
        teplot(m[CLASS_MEASURE], CLASS_MEASURE, ymax=1.0)

def process_single_json(filename, simple, plotting):
    history = {}
    with open(filename) as json_file:
        history = json.load(json_file)
    history = history['history']

    # Now, history contains a list with the statistics from the neural nets.
    # Odd elements have statistics from training and validation, while
    # even elements have statistics from testing.
    training = []
    testing = []

    odd = True
    for s in history:
        if odd:
            training.append(s)
        else:
            testing.append(s)
        odd = not odd
    
    tes = testing_stats(testing, simple)
    if not simple:
        trs = training_stats(training)
    
def plot_sequence_graph(data, error, y_title, prefix, label, es):
    plt.clf()
    plt.figure(figsize=(6.4,4.8))
    full_length = 100.0
    step = 0.1
    xlabels=range(len(data))
    xtitle = _('Learning stage')
    ytitle = _(y_title)
    main_step = full_length/len(xlabels)
    x = np.arange(0, full_length, main_step)

    # One main step less because levels go on sticks, not
    # on intervals.
    xmax = full_length - main_step + step
    # Gives space to fully show markers in the top.
    plt.errorbar(x, data*100, yerr=error*100, fmt='b-d', label=label)
    plt.xlim(-0.5, xmax)
    plt.xticks(x, xlabels)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.legend()
    plt.grid(True)
    graph_filename = constants.picture_filename(prefix, es)
    plt.savefig(graph_filename, dpi=600)


def process_json_sequence(plotting,es):
    autoencoder_data, classifier_data = get_data(es)
    if (len(autoencoder_data.shape) > 1):
        autoencoder_error = np.std(autoencoder_data, axis=1)
        autoencoder_data = np.mean(autoencoder_data, axis=1)
    else:
        autoencoder_error = np.zeros(autoencoder_data.shape, dtype=float)
    if (len(classifier_data.shape) > 1):
        classifier_error = np.std(classifier_data, axis=1)
        classifier_data = np.mean(classifier_data, axis=1)
    else:
        classifier_error = np.zeros(classifier_data.shape, dtype=float)
    if (plotting):
        plot_sequence_graph(
            autoencoder_data, autoencoder_error, 
            ytitles[AUTO_MEASURE], 'graph_autoencoder', 'Autoencoder', es)
        plot_sequence_graph(
            classifier_data, classifier_error, 
            ytitles[CLASS_MEASURE], 'graph_classifier', 'Classifier', es)
    else:
        print('Autoencoder mean: ')
        constants.print_csv(autoencoder_data)
        print('Autoencoder standard deviation: ')
        constants.print_csv(autoencoder_error)
        print('Classifier mean: ')
        constants.print_csv(classifier_data)
        print('Classifier standard deviation: ')
        constants.print_csv(classifier_error)

def get_data(es: constants.ExperimentSettings):
    autoencoder_data = []
    classifier_data = []
    autoencoder_prefix = \
        constants.model_prefix + constants.decoder_suffix
    classifier_prefix = \
        constants.model_prefix + constants.classifier_suffix
    for stage in range(STAGES):
        es.stage = stage
        autoencoder_fname = constants.json_filename(autoencoder_prefix, es)
        classifier_fname = constants.json_filename(classifier_prefix, es)
        measure = get_measure(autoencoder_fname, AUTO_MEASURE)
        autoencoder_data.append(measure)
        measure = get_measure(classifier_fname, CLASS_MEASURE)
        classifier_data.append(measure)
    return np.array(autoencoder_data), np.array(classifier_data)

def get_measure(fname, measure):
    _, _, _, _, _, value = process_json(fname, measure)
    return value

def process_json(fname, measure):
    history = None
    with open(fname) as file:
        history = json.load(file)
    # The JSON file becomes a dictionary with a single entry, 'history'
    history = history['history']
    # Now history contains a list with the statistics from the
    # neural nets. Even elements have statistics from training
    # and validation, while odd elements have statistics from
    # testing.
    loss = []
    acc = []
    val_loss = []
    val_acc = []
    test_loss = []
    test_acc = []
    odd = False
    for s in history:
        if odd:
            test_loss.append(s[LOSS])
            test_acc.append(s[measure])
        else:
            loss.append(s[LOSS])
            acc.append(s[measure])
            val_loss.append(s[VAL + LOSS])
            val_acc.append(s[VAL + measure])            
        odd = not odd
    return loss, acc, val_loss, val_acc, test_loss, test_acc

if __name__== "__main__" :
    args = docopt(__doc__)

    # Processing language
    languages = ['en', 'es']
    valid = False
    for lang in languages:
        if lang == args['--lang']:
            valid = True
            break
    if valid:
        lang = args['--lang']
        if lang != 'en':
            lang = gettext.translation('eam', localedir='locale', languages=[lang])
            lang.install()
    else:
        print(__doc__)
        exit(1)

    # Processing base dir.
    runpath=args['--dir']
    constants.run_path = runpath

    # Processing learned data.
    learned = 0
    if args['--learned']:
        try:
            learned = int(args['--learned'])
            if (learned < 0) or (learned >= constants.learned_data_groups):
                raise Exception('Number out of range.')
        except:
            constants.print_error(
                f'<learned> must be an integer between 0 and {constants.learned_data_groups}.')
            exit(1)
    
    # Processing tolerance.
    tolerance = 0
    if args['--tolerance']:
        try:
            tolerance = int(args['--tolerance'])
            if (tolerance < 0) or (tolerance >= constants.domain):
                raise Exception('Number out of range.')
        except:
            constants.print_error(
                f'<tolerance> must be an integer between 0 and {constants.domain}.')
            exit(1)
    
    plotting = not args['-n']

    # Processing single JSON file
    json_file = args['--tolerance'] if args['--tolerance'] else ''
    simple = True if args['-s'] else False
    exp_set = constants.ExperimentSettings(learned=learned, extended=True, tolerance=tolerance)
   
    if json_file == '':
        process_single_json(json_file, simple, plotting)
    else:
        process_json_sequence(plotting, exp_set)
       
