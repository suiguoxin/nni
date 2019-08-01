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
from nni.curvefitting_assessor.model_factory import CurveModel

from nni.mtsmac_advisor.test.util import create_fake_data_simple, create_fake_data_mnist, get_obs
from nni.mtsmac_advisor.test.util import create_fake_data_expdacay, create_fake_data_expdacay_diff_length




PATH = './examples/experiments'

# pylint:disable=missing-docstring
# pylint:disable=no-member
# pylint:disable=invalid-name


def get_err_abs_mtsmac(X, y, len_y):
    ''' predicted err of old points '''
    X_obs, y_obs = get_obs(X, y, len_y)

    predictor = Predictor(multi_task=True)
    predictor.fit(X_obs, y_obs)
    mean, _ = predictor.predict(X)

    err_abs = [0] * 21
    for t in range(21):
        count_t = 0
        for j, len_y_j in enumerate(len_y):
            if len_y_j <= t:
                err_abs[t] += abs(y[j][t]  - mean[j][t])
                count_t += 1
        if count_t:
            err_abs[t] /= count_t

    return err_abs


def get_err_abs_cf(X, y, len_y):
    _, y_obs = get_obs(X, y, len_y)

    err_abs = [0] * 21
    for t in range(21):
        count_t = 0
        for j, len_y_j in enumerate(len_y):
            if len_y_j <= t:
                print("t:", t)
                print("y_obs[{}]:\n".format(j), y_obs[j])
                curvemodel = CurveModel(target_pos=t)
                predict_y = curvemodel.predict(trial_history=y_obs[j])

                del curvemodel
                print("predict_y: ", predict_y)
                err_abs[t] += abs(y[j][t]  - predict_y)
                count_t += 1
        if count_t:
            err_abs[t] /= count_t
        print("err_abs", err_abs[t])

    return err_abs

def test():
    '''
    X, y = create_fake_data_mnist(metric="err_rate")
    len_y = [21,21,21,6]
    _, y_obs = get_obs(X, y, len_y)
    print("y_obs[3]:\n", y_obs[3])

    curvemodel = CurveModel(target_pos=6)
    predict_y = curvemodel.predict(trial_history=y_obs[3])
    print("predict_y: ", predict_y)
    '''
    for i in range(10):
        np.random.seed(i)
        t = 8
        y_obs = [0.11349999904632568, 0.09799999743700027, 0.11349999904632568, 0.11349999904632568, 0.10100000351667404, 0.10320000350475311, 0.09740000218153]
        predict_y = CurveModel(target_pos=t).predict(trial_history=y_obs)
        # del curvemodel
        print("predict_y: ", predict_y)


def plot_prediction_comp():
    X, y = create_fake_data_mnist(metric="accuracy")
    len_completed = [21] * 3
    len_half = [6, 7, 8]
    # len_half = [np.random.randint(6, 21) for _ in range(20)]
    len_y = len_completed + len_half

    err_abs_mtsmac = get_err_abs_mtsmac(X, y, len_y)
    err_abs_cf = get_err_abs_cf(X, y, len_y)

    plt.plot(range(21), err_abs_mtsmac, label='MTSMAC')
    plt.plot(range(21), err_abs_cf, label='CF')
    plt.title('Average ERR(obs) Comparasion')
    plt.legend()
    plt.savefig('{}/analyse/image/predict_comp.png'.format(PATH))
    plt.close()


def test_CF():
    X, y = create_fake_data_mnist(metric="accuracy")

    len_completed = [21] * 10

    # len_half = [np.random.randint(5, 21) for _ in range(3)]
    len_half = [10, 13, 15]
    len_y = len_completed + len_half

    _, y_obs = get_obs(X, y, len_y)

    mean = np.empty(len(len_y), dtype=object)
    for i in range(len(len_y)):
        mean[i] = []
        trial_history = y_obs[i]
        for pos in range(len(trial_history), 21):
            curvemodel = CurveModel(target_pos=pos)
            predict_y = curvemodel.predict(trial_history)
            mean[i].append(predict_y)

    for i in range(len(len_completed), len(len_y)):
        plt.plot(range(len(y_obs[i])), y_obs[i], linewidth=5.0)
        plt.plot(range(len(y_obs[i]), 21), mean[i], label='CF_{}'.format(i))
        plt.plot(range(21), y[i], label='LC_{}'.format(i))
    plt.title('True learning curve')
    plt.legend()
    plt.savefig('{}/analyse/image/LC.png'.format(PATH))
    plt.close()

test()
#test_CF()
#plot_prediction_comp()
