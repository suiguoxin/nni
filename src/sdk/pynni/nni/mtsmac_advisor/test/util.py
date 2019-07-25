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

PATH = './src/sdk/pynni/nni/mtsmac_advisor/test'
COLORS = ['b', 'g', 'r', 'y', 'm']


def create_fake_data_simple():
    X = np.array([[1],
                  [2]])
    y = np.array([[1, 2],
                  [1, 2, 3]])
    return X, y


def target(x):
    return np.exp(-(x - 2)**2) + np.exp(-(x - 6)**2/10) + 1 / (x**2 + 1)


def create_fake_data_one_dimension():
    x = [-2, ]


x = np.linspace(-2, 10, 10000).reshape(-1, 1)
y = target(x)


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


def create_fake_data_mnist(size_X=100, len_y=21):
    '''
    returns
    X: shape (size_X, 5)
    y: shape(size_X,) where element is a list of len_y
    '''
    assert size_X <= 100
    assert len_y <= 21

    with open('{}/data/experiment.json'.format(PATH)) as json_file:
        data = json.load(json_file)
        trials = data['trialMessage']
        X = np.empty([size_X, 5])
        y = np.empty(size_X, dtype=object)

        for i in range(size_X):
            trial = trials[i]
            parameters = trial['hyperParameters']['parameters']
            X[i] = [val for _, val in parameters.items()]
            y[i] = []
            intermediate = trial['intermediate']
            for _, res in enumerate(intermediate):
                y[i] += [1 - float(res['data'])]

    return X, y


def get_obs(X, y, size_obs, len_y):
    assert size_obs == len(len_y)
    assert all(ele <= len(y[0]) for ele in len_y)

    X_obs = X[:size_obs][:]
    y_obs = np.empty(size_obs, dtype=object)

    for i in range(size_obs):
        y_obs[i] = y[i][:len_y[i]]

    return X_obs, y_obs
