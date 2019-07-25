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

from nni.ftbo_advisor.predictor import Predictor
from nni.mtsmac_advisor.test.util import COLORS
from nni.mtsmac_advisor.test.util import create_fake_data_expdacay, create_fake_data_expdacay_diff_length
from nni.mtsmac_advisor.test.util import create_fake_data_mnist, get_obs

PATH = './src/sdk/pynni/nni/ftbo_advisor/test'

# pylint:disable=missing-docstring
# pylint:disable=no-member
# pylint:disable=invalid-name


def plot_asymptote():
    '''
    Fig 2(b)， 2(c)
    '''
    X, y = create_fake_data_mnist()
    size_obs = 13
    len_y = [21, 21, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1]
    X_obs, y_obs = get_obs(X, y, size_obs, len_y)

    predictor = Predictor()
    predictor.fit(X_obs, y_obs)
    mean, std = predictor.predict_asymptote_old(X_obs)
    # print('mean, std:')
    # print(mean, std)

    mean_points, std_points = predictor.predict_point_old(X_obs)
    print('mean, std:')
    print(mean, std)

    # figure 2(b)
    for i, y_i in enumerate(y_obs):
        idx_color = i % 5
        # plot observed learning curve
        if i < len(y_obs):
            plt.plot(range(len(y_obs[i])), y_obs[i], color=COLORS[idx_color], linewidth=5.0,
                     label='y_obs:{}'.format(i))

        # plot true learning curve
        N = len(y[i])
        plt.plot(range(N), y[i], color=COLORS[idx_color],
                 label='y_true:{}'.format(i))

        # plot next point prediction
        plt.vlines(x=len(y_obs[i]), ymin=mean_points[i] -
                   1.96 * std_points[i], ymax=mean_points[i] + 1.96 * std_points[i], label='next point prediction')

        # plot aymptote prediction
        mu = mean[i]
        sigma = std[i]
        plt.plot(np.arange(N), [mu] * N,
                 '--', color=COLORS[i % 5], label='Prediction')
        plt.fill_between(np.arange(N), [
            mu-1.9600 * sigma * 0.1]*N, [mu + 1.9600 * sigma * 0.1]*N, color=COLORS[idx_color], alpha=0.5, interpolate=True)
        # TODO: remove * 0.1

        plt.title('Training Curve Predictions')
        plt.ylim(-0.2, 1.2)
        plt.legend()
        plt.savefig('{}/image/2_b_{}.png'.format(PATH, i))
        # plt.savefig('{}/image/2_b_mnist.png'.format(PATH))
        plt.close()

    '''
    # figure 2(c)
    print('X\n', X)
    print('mean\n', mean)
    print('std\n ', std)

    # TODO: remove * 0.1
    plt.plot(X, mean, label='mean')
    plt.fill(np.concatenate([X, X[::-1]]), np.concatenate([mean - 1.9600 * std *
                                                           0.1, (mean + 1.9600 * std * 0.1)[::-1]]), alpha=.6, fc='c', ec='None')
    plt.title('Asymptotic GP')
    plt.legend()
    plt.savefig('{}/image/2_c_expdecay.png'.format(PATH))
    plt.close()
    '''


plot_asymptote()
