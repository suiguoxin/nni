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
from scipy.linalg import block_diag

from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.ensemble import RandomForestClassifier

from nni.freezethaw_advisor.test.util import create_fake_data_simple
from nni.freezethaw_advisor.test.util import create_fake_data_expdacay, create_fake_data_expdacay_diff_length, create_fake_data_mnist_diff_length


# pylint:disable=missing-docstring
# pylint:disable=no-member
# pylint:disable=invalid-name


def transform_data(X_in, y_in):
    X = []
    y = []

    for i, (X_i, y_i) in enumerate(zip(X_in, y_in)):
        for t in len(y_i):
            X.vstack[X_i.hstack(t)]
            y.append[y_i[t]] 
    
    return X, y


def plot_rfc():
    X, y = create_fake_data_expdacay()
    clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    clf.fit(X, y)  

    MAX = 50
    for t in range(MAX):
        clf.predict([X[0]])



plot_rfc()
