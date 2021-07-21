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

import sys
import json
import numpy as np
from matplotlib import pyplot as plt

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
    

def training_stats(data, epoch):
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


if __name__== "__main__" :
    if ((len(sys.argv) != 3) and ((len(sys.argv) != 4) or (sys.argv[1] != '-s'))):
        print(f'Usage: {sys.argv[0]} [-s] file.json epochs')
        sys.exit(1)
    
    simple = False
    next = 1
    if len(sys.argv) == 4:
        simple = True
        next = 2
    fname = sys.argv[next]
    epoch = int(sys.argv[next+1])
    
    history = {}
    with open(fname) as json_file:
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
        trs = training_stats(training, epoch)
