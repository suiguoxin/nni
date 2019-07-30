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
from nni.mtsmac_advisor.predictor import Predictor
from nni.mtsmac_advisor.test.util import create_fake_data_simple, create_fake_data_mnist, get_obs
from nni.mtsmac_advisor.test.util import create_fake_data_expdacay, create_fake_data_expdacay_diff_length


PATH = './examples/experiments'

# pylint:disable=missing-docstring
# pylint:disable=no-member
# pylint:disable=invalid-name


def plot_prediction_comp():
    X, y = create_fake_data_mnist()

    len_completed = [21, 21, 21] # size 3

    len_half = [np.random.randint(1, 21) for _ in range(30)]
    len_y = len_completed + len_half

    size_obs = len(len_y)
    X_obs, y_obs = get_obs(X, y, size_obs, len_y)

    predictor = Predictor(multi_task=True)
    predictor.fit(X_obs, y_obs)
    mean, std = predictor.predict(X)

    # predicted err of old points
    err_abs = [0] * 21
    for t in range(21):
        count_t = 0
        for j in range(len(len_completed), len(len_y)):
            if len_y[j] <= t:
                err_abs[t] += abs(y[j][t]  - mean[j][t])
                count_t += 1
        if count_t:
            err_abs[t] /= count_t
    plt.plot(range(1, 21), err_abs[1:], label='MTSMAC')
    plt.title('Average ERR(obs) Comparasion')
    plt.legend()
    plt.savefig('{}/analyse/image/predict_comp.png'.format(PATH))
    plt.close()

def test_CF():
    pass

test_CF()
# plot_prediction_comp()
