import csv
import gettext
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import constants
import dimex

# Translation
gettext.install('eam', localedir=None, codeset=None, names=None)

queue_prefix = 'runs/queue'
memory_prefix = 'runs/memory'
sigma_prefix = 'runs/sigma'

def plot_heatmap(matrix, prefix):
    plt.clf()
    plt.figure(figsize=(6.4, 4.8))
    xtick_labels = [i for i in range(32)]
    ytick_labels = [i for i in range(16)]
    seaborn.heatmap(matrix, xticklabels=xtick_labels,
        yticklabels=ytick_labels, cbar = False, cmap='coolwarm')
    plt.xlabel(_('Characteristics'))
    plt.ylabel(_('Values'))
    filename = prefix + _('-english') + '.svg'
    plt.savefig(filename, dpi=600)

def plot_matrix(prefix):
    matrix = np.loadtxt(prefix + '.csv', delimiter=',')
    plot_heatmap(matrix, prefix)

if __name__== "__main__" :
    plot_matrix(queue_prefix)
    plot_matrix(memory_prefix)
    plot_matrix(sigma_prefix)
