import json
import numpy as np


import matplotlib.pyplot as plt
from matplotlib import gridspec

from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor

from nni.freezethaw_advisor.predictor import Predictor


PATH = './src/sdk/pynni/nni/freezethaw_advisor/test'


def target(x):
    return (np.exp(-(x - 2)**2) + np.exp(-(x - 6)**2/10) + 1 / (x**2 + 1))/2


def posterior(predictor, x_obs, y_obs, grid):
    predictor.fit(x_obs, y_obs)

    mu, sigma = predictor.predict(grid, return_std=True)
    return mu, sigma


def plot_gp(predictor, x, y, x_obs, y_obs):
    fig = plt.figure(figsize=(16, 10))
    steps = len(x_obs)
    fig.suptitle(
        'Gaussian Process and Utility Function After {} Steps'.format(steps),
        fontdict={'size': 30}
    )

    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    axis = plt.subplot(gs[0])
    acq_ei = plt.subplot(gs[1])

    mu, sigma = posterior(predictor, x_obs, y_obs, x)
    axis.plot(x, y, linewidth=3, label='Target')
    axis.plot(x[np.argmax(y)], np.max(y), '*', markersize=15,
              label=u'Next Best Result', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
    axis.plot(x_obs.flatten(), y_obs, 'D', markersize=8,
              label=u'Observations', color='r')
    axis.plot(x, mu, '--', color='k', label='Prediction')

    axis.fill(np.concatenate([x, x[::-1]]),
              np.concatenate(
                  [mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
              alpha=.6, fc='c', ec='None', label='95% confidence interval')

    axis.set_xlim((-2, 10))
    axis.set_ylim((None, None))
    axis.set_ylabel('f(x)', fontdict={'size': 20})
    axis.set_xlabel('x', fontdict={'size': 20})

    axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    # acq_ei.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    plt.savefig('{}/plot_gp.png'.format(PATH))


x = np.linspace(-2, 10, 10000).reshape(-1, 1)
y = target(x)
x_obs = np.array([[-1], [2], [5], [8]])
y_obs = np.array([target(-1), target(2), target(5), target(8)])

predictor = GaussianProcessRegressor(kernel=Matern(nu=2.5))
# plot_gp(predictor, x, y, x_obs, y_obs)


def fake_X_y():
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

        X = X[:3][:]
        y = y[:3][:3]

        print(X)
        print(y)
        print(X.shape)
        print(y.shape)

        return X, y


def plot_asymptote():
    X, y = fake_X_y()

    predictor = Predictor()

    predictor.fit(X, y)
    mean, var = predictor.predict_asymptote_old()

    for i in range(len(y)):
        length = len(y[i])
        print('length:')
        print(length)
        plt.plot(np.arange(length), y[i])

        mu = mean[i][0]
        print('mu:')
        print(mu)
        sigma = np.sqrt(var[i][i])
        print('sigma:')
        print(sigma)
        plt.plot(np.arange(length), [mu] *
                 length, '--', label='Prediction')
        plt.fill_between(np.arange(length), [
                         mu-sigma]*length, [mu+sigma]*length, alpha=0.5, interpolate=True)
    plt.savefig('{}/plot_asym.png'.format(PATH))


plot_asymptote()
