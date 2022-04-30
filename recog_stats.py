import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import constants

stages = 10
tolerance = 0
learned = 4
sigma = 0.10
iota = 0.30
kappa = 1.50
extended = True
runpath = f'runs-d{learned}-t{tolerance}-i{iota:.1f}-k{kappa:.1f}-s{sigma:.2f}'
constants.run_path = runpath
es = constants.ExperimentSettings(learned=learned, tolerance = tolerance, extended=extended,
        iota=iota, kappa=kappa, sigma=sigma)

print(f'Getting data from {constants.run_path}')

def plot_recognition_graph(data, errs, es):
    plt.clf()
    fig = plt.figure()
    x = range(stages)
    plt.ylim(0, 1.0)
    plt.xlim(0, stages)
    plt.autoscale(True)
    plt.errorbar(x, means[:,0], fmt='r-o', yerr=errs[:,0], label='Correct to network')
    plt.errorbar(x, means[:,1], fmt='b-s', yerr=errs[:,1], label='Correct to memory')

    plt.ylabel('Normalized distance')
    plt.xlabel('Stages')
    plt.legend()

    prefix = constants.recognition_prefix
    filename = constants.picture_filename(prefix, es)
    fig.savefig(filename, dpi=600)


def get_fold_stats(es, fold):
    prefix = constants.recognition_prefix
    filename = constants.recog_filename(prefix, es, fold)
    print(f'Reading file: {filename}')
    df = pd.read_csv(filename)
    df = df[['CorrSize', 'Cor2Net', 'Cor2Mem']]
    data = df.to_numpy(dtype=float)
    data[:, 1] = data[:,1] / (data[:,1] + data[:,0])
    data[:, 2] = data[:,2] / (data[:,2] + data[:,0])
    return data[:,1:]

stats = []
for stage in range(stages):
    es.stage = stage
    stage_stats = []
    for fold in range(constants.n_folds):
        fold_stats = get_fold_stats(es, fold)
        stage_stats.append(fold_stats)
    stage_stats = np.array(stage_stats, dtype=float)
    stats.append(stage_stats)

stats = np.array(stats, dtype=float)    
print(stats.shape)
# Reduce folds to their means.
means = np.mean(stats, axis=1)
# Means and standard deviations of measures per stage.
stdvs = np.std(means, axis=1)
means = np.mean(means, axis=1)
plot_recognition_graph(means, stdvs, es)
