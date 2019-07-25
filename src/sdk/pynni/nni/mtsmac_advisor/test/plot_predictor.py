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

# pylint:disable=import-error
from nni.mtsmac_advisor.test.util import PATH, COLORS
from nni.mtsmac_advisor.predictor import Predictor
from nni.mtsmac_advisor.test.util import create_fake_data_simple, create_fake_data_mnist
from nni.mtsmac_advisor.test.util import create_fake_data_expdacay, create_fake_data_expdacay_diff_length, create_fake_data_mnist_diff_length


# pylint:disable=missing-docstring
# pylint:disable=no-member
# pylint:disable=invalid-name


def plot_rfr():
    size_X = 50
    len_y = 10
    X, y = create_fake_data_mnist(size_X, len_y)
    #X, y = create_fake_data_mnist_diff_length()

    predictor = Predictor(multi_task=True)
    size_train = 48
    predictor.fit(X[:size_train], y[:size_train])
    mean, std = predictor.predict(X[size_train:])

    # plot true learning curve
    idx_color = 0
    for i in range(size_train, size_X):
        N = len(y[i])
        plt.plot(range(N), y[i], color=COLORS[idx_color],
                 label='y_true:{}'.format(i - size_train))
        idx_color += 1

    # plot prediction
    size_predict = size_X - size_train
    for i in range(size_predict):
        mu = mean[i]
        sigma = std[i]

        plt.plot(range(len(y[i])), mu, label='y_predict:{}'.format(i))
        T = np.arange(len(y[i])).reshape(-1, 1)
        plt.fill(np.concatenate([T, T[::-1]]), np.concatenate([mu - 1.9600 * sigma * 0.5, (mu + 1.9600 * sigma * 0.5)[::-1]]),
                 color=COLORS[i], alpha=.6)

    plt.title('Learning curve MNIST')
    plt.legend()
    plt.savefig('{}/image/lr_final.png'.format(PATH))
    plt.close()


plot_rfr()
