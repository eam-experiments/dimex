# Copyright [2022] Luis Alberto Pineda Cortés, Rafael Morales Gamboa.
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

"""Memories confusion matrix

Usage:
  mem_confrix -h | --help
  mem_confrix [<stage>] [--learned=<learned_data>] [-x] [--tolerance=<tolerance>] [--sigma=<sigma>] [--iota=<iota>] [--kappa=<kappa>] [--runpath=<runpath>] [ -l (en | es) ]

Options:
  --learned=<learned_data>      Selects which learneD Data is used for evaluation, recognition or learning [default: 0].
  -x                            Use the eXtended data set as testing data for memory.
  --tolerance=<tolerance>       Allow Tolerance (unmatched features) in memory [default: 0].
  --sigma=<sigma>               Scale of standard deviation of the distribution of influence of the cue [default: 0.25]
  --iota=<iota>                 Scale of the expectation (mean) required as minimum for the cue to be accepted by a memory [default: 0.0]
  --kappa=<kappa>               Scale of the expectation (mean) rquiered as minimum for the cue to be accepted by the system of memories [default: 0.0]
  --runpath=<runpath>           Sets the path to the directory where everything will be saved [default: runs]
  -l                            Chooses Language for graphs.            

The parameter <stage> indicates the stage of learning from which data is used.
Default is the first one.
"""
from os import confstr
from docopt import docopt
import gettext
import matplotlib.pyplot as plt
import numpy as np
import constants
import dimex

# Translation
gettext.install('eam', localedir=None, codeset=None, names=None)

# Analysis is of memories filled with all the corpus.
fill = 100.0
TP = (0,0)
FP = (0,1)
FN = (1,0)
TN = (1,1)

category_sizes = None

def plot_graph(label, tags, values, errors, prefix, es):
  fig = plt.figure(figsize=(6.4, 4.8))
  title = dimex.labels_to_phns[label]
  width = 0.75
  plt.bar(tags, values, width, yerr=errors, capsize=2)
  plt.ylim(0.0, 1.0)
  plt.ylabel('Percentage')
  plt.title(title)
  label_prefix = prefix + '-lbl_' + str(label).zfill(3)
  filename = constants.picture_filename(label_prefix, es)
  plt.savefig(filename, dpi=600)
  plt.close()


def plot_confrix(means, stdvs, prefix, es):
  xtags = [_('TP'), _('FN'), _('FP'), _('TN')]
  for label in range(constants.n_labels):
    m = means[label]
    s = stdvs[label]
    lmeans = [m[TP], m[FN], m[FP], m[TN]]
    lstdvs = [s[TP], s[FN], s[FP], s[TN]]
    plot_graph(label, xtags, lmeans, lstdvs, prefix, es)

def plot_distrib(distribution, prefix, es):
  xtags = dimex.labels_to_phns
  fig = plt.figure(figsize=(6.4, 4.8))
  width = 0.75
  plt.bar(xtags, distribution, width)
  plt.ylabel('Frecuency')
  filename = constants.picture_filename(prefix, es)
  plt.savefig(filename, dpi=600)
  plt.close()

def export_table(means, stdvs, prefix, es):
  filename = constants.csv_filename(prefix, es)
  table = np.zeros((constants.n_labels, 8))
  for i in range(constants.n_labels):
    table[i, :4] = np.ravel(means[i], 'F')
    table[i, 4:] = np.ravel(stdvs[i], 'F')
  np.savetxt(filename, table, delimiter=',')  

def normalized(cms):
  global category_sizes
  totals = np.sum(cms, axis=1)
  if category_sizes is None:
    category_sizes = totals[:,0]
  for i in range(constants.n_labels):
    for j in range(2):
      cms[i, :, j] = cms[i, :, j] / totals[i, j]
  return cms

def adjust(values, weights):
  for i in range(len(weights)):
    values[i,:,:] = values[i,:,:]*weights[i]
  return values

def mem_confrix(es):
  confrixs = np.zeros((constants.n_folds, constants.n_labels, 2, 2), dtype=float)
  for fold in range(constants.n_folds):
    filename = constants.memory_conftrix_filename(fill, es, fold)
    cms = np.load(filename)
    confrixs[fold] = normalized(cms)
  means = np.mean(confrixs, axis=0)
  stdvs = np.std(confrixs, axis=0)
  prefix = "mem_confrix"
  export_table(means, stdvs, prefix, es)
  plot_confrix(means, stdvs, prefix, es)
  prefix = "corpus_distrib"
  plot_distrib(category_sizes, prefix, es)

if __name__== "__main__" :
    args = docopt(__doc__)

    # Processing language.
    lang = 'en'
    if args['es']:
        lang = 'es'
        es = gettext.translation('eam', localedir='locale', languages=['es'])
        es.install()

    # Processing stage. 
    stage = 0
    if args['<stage>']:
        try:
            stage = int(args['<stage>'])
            if stage < 0:
                raise Exception('Negative number.')
        except:
            constants.print_error('<stage> must be a positive integer.')
            exit(1)

    # Processing learned data.
    learned = 0
    if args['--learned']:
        try:
            learned = int(args['--learned'])
            if (learned < 0) or (learned >= constants.learned_data_groups):
                raise Exception('Number out of range.')
        except:
            constants.print_error('<learned_data> must be a positive integer.')
            exit(1)

    # Processing use of extended data as testing data for memory
    extended = args['-x']

    # Processing tolerance.
    tolerance = 0
    if args['--tolerance']:
        try:
            tolerance = int(args['--tolerance'])
            if (tolerance < 0) or (tolerance > constants.domain):
                raise Exception('Number out of range.')
        except:
            constants.print_error('<tolerance> must be a positive integer.')
            exit(1)

    # Processing sigma.
    sigma = 0.25
    if args['--sigma']:
        try:
            sigma = float(args['--sigma'])
            if (sigma < 0):
                raise Exception('<sigma> must be positive.')
        except:
            constants.print_error('<sigma> must be a positive number.')
            exit(1)

    # Processing sigma.
    iota = 0.0
    if args['--iota']:
        try:
            iota = float(args['--iota'])
            if (iota < 0):
                raise Exception('<iota> must be positive.')
        except:
            constants.print_error('<iota> must be a positive number.')
            exit(1)

    # Processing sigma.
    kappa = 0.0
    if args['--kappa']:
        try:
            kappa = float(args['--kappa'])
            if (kappa < 0):
                raise Exception('<kappa> must be positive.')
        except:
            constants.print_error('<kappa> must be a positive number.')
            exit(1)

    # Processing runpath.
    constants.run_path = args['--runpath']
    exp_set = constants.ExperimentSettings(
        stage, learned, extended, tolerance, sigma, iota, kappa)
    print(f'Working directory: {constants.run_path}')
    print(f'Experimental settings: {exp_set}')

    mem_confrix(exp_set)