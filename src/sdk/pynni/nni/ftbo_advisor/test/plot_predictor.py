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

from sklearn.gaussian_process.kernels import Matern

from nni.ftbo_advisor.predictor import Predictor
from nni.ftbo_advisor.test.util import PATH, COLORS
from nni.ftbo_advisor.test.util import create_fake_data_expdacay, create_fake_data_expdacay_diff_length, create_fake_data_mnist_diff_length

# pylint:disable=missing-docstring
# pylint:disable=no-member
# pylint:disable=invalid-name


def plot_asymptote():
    '''
    Fig 2(b)， 2(c)
    '''
    X, y = create_fake_data_expdacay()
    # X, y = create_fake_data_expdacay_diff_length()
    #X, y = create_fake_data_mnist_diff_length()

    predictor = Predictor()

    predictor.fit(X, y)
    mean, std = predictor.predict_asymptote_old(X)
    print('mean, std:')
    print(mean, std)

    # figure 2(b)
    for i, y_i in enumerate(y):
        length = len(y[i])

        plt.plot(np.arange(length), y_i, color=COLORS[i], )
        mu = mean[i]
        sigma = std[i]
        print('mu:')
        print(mu)
        print('sigma:')
        print(sigma)
        plt.plot(np.arange(length), [mu] * length,
                 '--', color=COLORS[i], label='Prediction')
        plt.fill_between(np.arange(length), [
            mu-1.9600 * sigma * 0.1]*length, [mu + 1.9600 * sigma * 0.1]*length, color=COLORS[i], alpha=0.5, interpolate=True)
        # TODO: remove * 0.1

    plt.title('Training Curve Predictions')
    plt.legend()
    plt.savefig('{}/image/2_b_expdecay.png'.format(PATH))
    # plt.savefig('{}/image/2_b_mnist.png'.format(PATH))
    plt.close()

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


plot_asymptote()
