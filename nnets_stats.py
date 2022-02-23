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
  nnets_stats [-s] [--json_file=<json_file>] [--dir=<dir>] [--learned=<learned>] [--tolerance=<tolerance>] [--lang=<lang>]

Options:
  -h                        Show this screen.
  -s                        Whether to generate a simple (-s) or detailed graph.
  --json_file=<json_file>   Analyse a single given JSON file. Ignores all following options.
  --dir=<dir>               Base directory for finding JSON files. [default: runs].
  --learned=<learned>       Index of data learned (original, agreed, ...) [default: 0].
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
gettext.install('ame', localedir=None, codeset=None, names=None)

# Keys for data
LOSS = 'loss'
ACCURACY = 'accuracy'
VAL = 'val_'



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
        trplot(d[ACCURACY], d[VAL+ACCURACY], ACCURACY, VAL+ACCURACY,epoch,n)
        n += 1


def testing_stats(data, simple):
    """ Analyse neural nets testing data. 
    """
    n = len(data)
    m = {LOSS: [], ACCURACY: []}
    for d in data:
        m[LOSS].append(d[LOSS])
        m[ACCURACY].append(d[ACCURACY])
    
    if simple:
        print(f'{LOSS}: {m[LOSS]}')
        print(f'{ACCURACY}: {m[ACCURACY]}')
    else:
        teplot(m[LOSS], LOSS)
        teplot(m[ACCURACY], ACCURACY, ymax=1.0)

def process_single_json(filename, simple):
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
    
def plot_sequence_graph(a_data, a_error, c_data, c_error, base_dir, es):
    plt.clf()
    plt.figure(figsize=(6.4,4.8))
    full_length = 100.0
    step = 0.1
    xlabels=range(len(a_data))
    xtitle = _('Learning stage')
    ytitle = _('Accuracy')
    main_step = full_length/len(xlabels)
    x = np.arange(0, full_length, main_step)

    # One main step less because levels go on sticks, not
    # on intervals.
    xmax = full_length - main_step + step
    # Gives space to fully show markers in the top.
    ymax = 100.0 + 2.0
    plt.errorbar(x, a_data*100, fmt='b-o', yerr=a_error*100, label=_('Autoencoder'))
    plt.errorbar(x, c_data*100, fmt='r-s', yerr=c_error*100, label=_('Classifier'))
    plt.xlim(-0.5, xmax)
    plt.ylim(0, ymax)
    plt.xticks(x, xlabels)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.legend()
    plt.grid(True)
    s = 'graph_networks' + _('-english')
    graph_filename = constants.picture_filename(s, es)
    graph_filename = graph_filename.replace('runs', base_dir)

    plt.savefig(graph_filename, dpi=600)


def process_json_sequence(dir, es):
    autoencoder_data, classifier_data = get_data(dir, es)
    autoencoder_error = np.std(autoencoder_data, axis=1)
    autoencoder_data = np.mean(autoencoder_data, axis=1)
    classifier_error = np.std(classifier_data, axis=1)
    classifier_data = np.mean(classifier_data, axis=1)
    print('Autoencoder mean: ' + str(autoencoder_data))
    print('Autoencoder error: ' + str(autoencoder_error))
    print('Classifier mean: ' + str(classifier_data))
    print('Classifier error: ' + str(classifier_error))
    plot_sequence_graph(autoencoder_data, autoencoder_error,
        classifier_data, classifier_error, dir, es)

def get_data(dir, es: constants.ExperimentSettings):
    autoencoder_data = []
    classifier_data = []
    n = -1
    for stage in range(10):
        try:
            if stage < 9:
                autoencoder_fname = f'{dir}/model-autoencoder-stg_{stage:03}-lrn_{es.learned:03}-tol_{tolerance:03}.json'
                classifier_fname = f'{dir}/model-classifier-stg_{stage:03}-lrn_{es.learned:03}-tol_{tolerance:03}.json'
            else:
                autoencoder_fname = f'{dir}/model-autoencoder-stg_{stage:03}-lrn_{es.learned:03}-ext-tol_{tolerance:03}.json'
                classifier_fname = f'{dir}/model-classifier-stg_{stage:03}-lrn_{es.learned:03}-ext-tol_{tolerance:03}.json'
            accuracy = get_accuracy(autoencoder_fname)
            autoencoder_data.append(accuracy)
            accuracy = get_accuracy(classifier_fname)
            classifier_data.append(accuracy)
            n += 1
        except:
            break
    es.stage = n
    return np.array(autoencoder_data), np.array(classifier_data)

def get_accuracy(fname):
    _, _, _, _, _, accuracy = process_json(fname)
    return accuracy

def process_json(fname):
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
            test_loss.append(s['loss'])
            test_acc.append(s['accuracy'])
        else:
            loss.append(s['loss'])
            acc.append(s['accuracy'])
            val_loss.append(s['val_loss'])
            val_acc.append(s['val_accuracy'])            
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
            lang = gettext.translation('ame', localedir='locale', languages=[lang])
            lang.install()
    else:
        print(__doc__)
        exit(1)

    # Processing base dir.
    base_dir = args['--dir']

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

    # Processing single JSON file
    json_file = args['--tolerance'] if args['--tolerance'] else ''
    simple = True if args['-s'] else False
    exp_set = constants.ExperimentSettings(learned=learned, extended=False, tolerance=tolerance)
   
    if json_file == '':
        process_single_json(json_file, simple)
    else:
        process_json_sequence(base_dir, exp_set)
       
