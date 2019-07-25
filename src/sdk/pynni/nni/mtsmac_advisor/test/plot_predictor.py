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
from nni.mtsmac_advisor.test.util import create_fake_data_simple, create_fake_data_mnist, get_obs
from nni.mtsmac_advisor.test.util import create_fake_data_expdacay, create_fake_data_expdacay_diff_length


# pylint:disable=missing-docstring
# pylint:disable=no-member
# pylint:disable=invalid-name


def plot_rfr():
    X, y = create_fake_data_mnist()

    size_obs = 13
    len_y = [21, 21, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1]
    X_obs, y_obs = get_obs(X, y, size_obs, len_y)

    predictor = Predictor(multi_task=True)
    predictor.fit(X_obs, y_obs)
    mean, std = predictor.predict(X)

    for i in range(0, 15):
        idx_color = i % 5
        # plot observed learning curve
        if i < len(y_obs):
            plt.plot(range(len(y_obs[i])), y_obs[i], color=COLORS[idx_color], linewidth=5.0,
                     label='y_obs:{}'.format(i))

        # plot true learning curve
        N = len(y[i])
        plt.plot(range(N), y[i], color=COLORS[idx_color],
                 label='y_true:{}'.format(i))

        # plot prediction
        mu = mean[i]
        sigma = std[i]

        plt.plot(range(N), mu, label='y_predict:{}'.format(i))
        T = np.arange(N).reshape(-1, 1)
        plt.fill(np.concatenate([T, T[::-1]]), np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
                 color=COLORS[idx_color], alpha=.6)

        plt.title('Learning curve MNIST')
        plt.ylim(-0.2, 1.2)
        plt.legend()
        plt.savefig('{}/image/lr_final_{}.png'.format(PATH, i))
        plt.close()


plot_rfr()
