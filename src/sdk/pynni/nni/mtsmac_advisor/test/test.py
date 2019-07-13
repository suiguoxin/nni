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
test.py
"""

import warnings
import numpy as np

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from nni.mtsmac_advisor.predictor import Predictor

from nni.mtsmac_advisor.test.util import PATH, COLORS
from nni.mtsmac_advisor.test.util import create_fake_data_simple, create_fake_data_mnist
from nni.mtsmac_advisor.test.util import create_fake_data_expdacay, create_fake_data_expdacay_diff_length, create_fake_data_mnist_diff_length

from sklearn.datasets import make_classification


# pylint:disable=missing-docstring
# pylint:disable=no-member
# pylint:disable=invalid-name


def plot_rfc():
    size_X = 10
    size_y = 10
    X, y = create_fake_data_mnist(size_X, size_y)

    predictor = Predictor()
    size_train = 5
    predictor.fit(X[:size_train], y[:size_train])
    mean, std = predictor.predict(X[size_train:])

    # plot true learning curve
    for i in range(10):
        N = len(y[i])
        #plt.plot(range(N), y[i], label='y_true')

    size_predict = size_X - size_train
    for i in range(size_predict):
        mu = mean[i] # TODO: check
        sigma = std[i]
        print('mu:')
        print(mu)
        print('sigma:')
        print(sigma)
        length = len(y[i])
        plt.plot(range(length), [mu] * length, label='y_predict')
        plt.fill_between(np.arange(length), [
            mu-1.9600 * sigma]*length, [mu + 1.9600 * sigma]*length, alpha=0.5, interpolate=True)

    plt.title('Learning curve MNIST')
    plt.savefig('{}/image/lr.png'.format(PATH))
    plt.close()


X, y = make_classification(n_samples=10, n_features=4,
                           n_informative=2, n_redundant=0, random_state=0, shuffle=False)

print(X)
print(y)


plot_rfc()
