def sort (seed, means, stdvs):
    total = seed + means
    total, seed, means, stdvs = (list(t) for t in zip(*sorted(zip(total, seed, means, stdvs), reverse=True))) 
    return seed, means, stdvs

def plot_recognition_graph(stage, tolerance, means, errs):
    plt.clf()
    fig = plt.figure()
    x = range(constants.n_folds)
    plt.ylim(0, 1.0)
    plt.xlim(0, 10.0)
    plt.autoscale(False)
    plt.errorbar(x, means[:,0], fmt='r-o', yerr=errs[:,0], label='Correct to network produced')
    plt.errorbar(x, means[:,1], fmt='g-d', yerr=errs[:,1], label='Correct to memory produced')
    plt.errorbar(x, means[:,2], fmt='b-s', yerr=errs[:,2], label='Network produced to memory produced')

    plt.ylabel('Normalized distance')
    plt.xlabel('Folds')
    plt.legend()
    prefix = constants.recognition_prefix
    filename = constants.picture_filename(prefix, EXPERIMENT, tolerance, stage)
    fig.savefig(filename, dpi=600)

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
