import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import constants

stages = 10
tolerance = 0
learned = 4
sigma = 50
runpath = f'runs-32-d{learned}-t{tolerance}-s{sigma}'
constants.run_path = runpath
es = constants.ExperimentSettings(learned=learned, tolerance = tolerance)

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
    es.extended = (stage == (stages - 1))
    stage_stats = []
    for fold in range(constants.n_folds):
        fold_stats = get_fold_stats(es, fold)
        print(fold_stats.shape)
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

def recognition_stats(tolerance: int, stage: int):
    means = np.zeros((constants.n_folds, 3))
    stdvs = np.zeros((constants.n_folds, 3))
    for fold in range(constants.n_folds):
        filename = constants.recog_filename(constants.recognition_prefix, EXPERIMENT,
            fold, tolerance, stage)
        df = pd.read_csv(filename)
        df['C2N'] = df['Cor2Net'] / df['CorrSize']
        df['C2M'] = df['Cor2Mem'] / df['CorrSize']
        df['N2M'] = 2*df['Net2Mem'] / (df['NetSize'] + df['MemSize'])

        stats = df.describe(include=[np.number])
        means[fold,:] = stats.loc['mean'].values[-3:]
        stdvs[fold,:] = stats.loc['std'].values[-3:]
    print(means[:,1])
    plot_recognition_graph(stage, tolerance, means, stdvs)
