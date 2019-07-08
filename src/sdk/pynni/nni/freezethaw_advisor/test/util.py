import json
import numpy as np

PATH = './src/sdk/pynni/nni/freezethaw_advisor/test'
COLORS = ['b', 'g', 'r', 'c', 'm']


def create_fake_data_simple():
    X = np.array([[1],
                  [2]])
    y = np.array([[1, 2],
                  [1, 2, 3]])
    return X, y


def create_fake_data_expdacay(exp_lambda=0.5, asymp=0.5, gaussian_noise=0.1):
    MAXTIME = 50
    asymps = [0.4, 0.3, 0.2, 0.1]
    #asymps = [0.4]

    X = np.array([1, 2, 3, 4]).reshape(-1, 1)
    #X = np.array([1]).reshape(-1, 1)
    y = np.empty(len(asymps), dtype=object)

    for i, asymp in enumerate(asymps):
        y[i] = []
        for t in range(MAXTIME):
            sample = np.exp(-exp_lambda * t)
            sample = sample * (1-asymp) + asymp
            noise = np.random.normal(0, 1/(t+1) * gaussian_noise)
            sample += noise
            y[i] += [sample]

    return X, y


def create_fake_data_mnist():
    with open('{}/experiment.json'.format(PATH)) as json_file:
        data = json.load(json_file)
        trials = data['trialMessage']
        X = np.empty([len(trials), 1])
        y = np.empty(len(trials), dtype=object)

        for i, trial in enumerate(trials):
            # X
            X[i] = [trial['hyperParameters']['parameters']['dropout_rate']]
            # y
            y[i] = []
            intermediate = trial['intermediate']
            for j, res in enumerate(intermediate):
                y[i] += [1 - float(res['data'])]

        X = X[: 3][:]
        y = y[: 3][: 3]

    return X, y
