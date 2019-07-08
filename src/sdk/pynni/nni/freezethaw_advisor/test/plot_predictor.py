# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge,
# to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
plot_predictor.py
"""
import numpy as np


import matplotlib.pyplot as plt
from matplotlib import gridspec

from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor

from nni.freezethaw_advisor.predictor import Predictor
from nni.freezethaw_advisor.test.util import PATH
from nni.freezethaw_advisor.test.util import create_fake_data_expdacay


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


def plot_asymptote():
    X, y = create_fake_data_expdacay()

    predictor = Predictor(optimizer=None)

    predictor.fit(X, y)
    mean, std = predictor.predict_asymptote_old(return_std=True)

    for i in range(len(y)):
        length = len(y[i])
        print('length:')
        print(length)
        plt.plot(np.arange(length), y[i])

        mu = mean[i][0]
        print('mu:')
        print(mu)
        sigma = np.sqrt(std[i])
        print('sigma:')
        print(sigma)
        plt.plot(np.arange(length), [mu] *
                 length, '--', label='Prediction')
        plt.fill_between(np.arange(length), [
                         mu-sigma]*length, [mu+sigma]*length, alpha=0.5, interpolate=True)
    plt.savefig('{}/plot_asym.png'.format(PATH))


plot_asymptote()
