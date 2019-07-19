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
util.py
"""
import json
import numpy as np

# pylint:disable=missing-docstring
# pylint:disable=no-member
# pylint:disable=invalid-name

PATH = './src/sdk/pynni/nni/freezethaw_advisor/test'
COLORS = ['b', 'g', 'r', 'y', 'm']


def create_fake_data_simple():
    X = np.array([[1],
                  [2]])
    y = np.array([[1, 2],
                  [1, 2, 3]])
    return X, y


def create_fake_data_expdacay(exp_lambda=0.5, asymp=0.5, gaussian_noise=0.1):
    MAXTIME = 50
    #asymps = [0.4, 0.3, 0.2, 0.1]
    asymps = [0.4]
    # asymps = [0.4]

    #X = np.array([1, 2, 3, 4]).reshape(-1, 1)
    X = np.array([1]).reshape(-1, 1)
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


def create_fake_data_expdacay_diff_length(exp_lambda=0.5, asymp=0.5, gaussian_noise=0.1):
    '''
    data for fig 2(b)
    '''
    asymps = [0.2, 0.25, 0.4, 0.2, 0.35]
    length = [50, 35, 5, 20, 15]

    X = np.arange(1, 6).reshape(-1, 1)
    y = np.empty(len(asymps), dtype=object)

    for i, asymp in enumerate(asymps):
        y[i] = []
        for t in range(length[i]):
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
            for _, res in enumerate(intermediate):
                y[i] += [1 - float(res['data'])]

        X = X[: 3][:]
        y = y[: 3][: 3]

    return X, y


def create_fake_data_mnist_diff_length():
    MAX = 3
    length = [7, 5, 9]
    with open('{}/experiment.json'.format(PATH)) as json_file:
        data = json.load(json_file)
        trials = data['trialMessage']
        X = np.empty([len(trials), 1])
        y = np.empty(len(trials), dtype=object)

        for i, trial in enumerate(trials):
            if i >= MAX:
                break
            # X
            X[i] = [trial['hyperParameters']['parameters']['dropout_rate']]
            # y
            y[i] = []
            intermediate = trial['intermediate']
            for j, res in enumerate(intermediate):
                if j > length[i]:
                    break
                y[i] += [1 - float(res['data'])]

        X = X[: MAX][:]
        y = y[: MAX][:]
        print(X)
        print(y)

    return X, y
