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

"""Mismatches: Generate graph of mismatches

Usage:
  mismatches -h | --help
  mismatches [--learned=<learned>] [--tolerance=<tolerance>] [--lang=<lang>]

Options:
  -h                  Show this screen.
  --learned=<learned>  Index of data learned (original, agreed, ...) [default: 0].
  --tolerance=<tolerance>  Differences allowed between memory and cue. [default: 0].
  --lang=<language>   Chooses Language for graphs [default: en].

Arguments:
  language: en - English
  language: es - Spanish           
"""
from docopt import docopt
import gettext
import numpy as np
import matplotlib.pyplot as plt

import constants

# Translation
gettext.install('ame', localedir=None, codeset=None, names=None)

def generate_graph(es: constants.ExperimentSettings):
    data = get_data(es)
    average = np.mean(data, axis=0, dtype=float)
    stdev = np.std(data, axis=0)
    print('Means: ' + ', '.join([str(x) for x in average])) 
    print('Stdvs: ' + ', '.join([str(x) for x in stdev])) 
    plot_graph(average, stdev, es)

def plot_graph(data, error, es: constants.ExperimentSettings):
    plt.clf()
    plt.figure(figsize=(6.4,4.8))
    full_length = 100.0
    step = 0.1
    xlabels=constants.memory_fills
    xtitle = _('Percentage of memory corpus')
    ytitle = _('Mismatches')
    main_step = full_length/len(xlabels)
    x = np.arange(0, full_length, main_step)

    # One main step less because levels go on sticks, not
    # on intervals.
    xmax = full_length - main_step + step
    # Gives space to fully show markers in the top.
    ymax = np.max(data + error) + 2
    plt.errorbar(x, data, fmt='b-o', yerr=error, label=_('Mismatches'), )
    plt.xlim(-0.5, xmax)
    plt.ylim(0, ymax)
    plt.xticks(x, xlabels)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.legend()
    plt.grid(True)
    s = 'graph_mismatches' + _('-english')
    graph_filename = constants.picture_filename(s, es)
    plt.savefig(graph_filename, dpi=600)


def get_data(es: constants.ExperimentSettings):
    data = []
    n = -1
    for stage in range(10):
        try:
            filename = f'runs/main_total_mismatches-stg_{stage:03}-lrn_{es.learned:03}-tol_{tolerance:03}.csv'
            row = np.loadtxt(filename, delimiter=',')
            data.append(row)
            n += 1
        except:
            try:
                filename = f'runs/main_total_mismatches-stg_{stage:03}-lrn_{es.learned:03}-ext-tol_{tolerance:03}.csv'
                row = np.loadtxt(filename, delimiter=',')
                data.append(row)
                n += 1
            except:
                pass
    es.stage = n
    return np.array(data)

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
    exp_set = constants.ExperimentSettings(learned=learned, extended=False, tolerance=tolerance)
    generate_graph(exp_set)


