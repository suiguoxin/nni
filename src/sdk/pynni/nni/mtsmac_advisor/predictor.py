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
predictor_rf.py
"""

import warnings
from functools import reduce
from operator import itemgetter

import numpy as np

from sklearn.ensemble import RandomForestRegressor


# pylint:disable=invalid-name
# pylint:disable=attribute-defined-outside-init


class Predictor():
    """
    Freeze-Thaw Bayesian Optimization: Two Step Gaussian Process Predictor
    """

    def __init__(self):
        """
        Parameters
        ----------
        """
        self.regr = RandomForestRegressor(
            n_estimators=10, max_depth=2, random_state=0)

    def fit(self, X, y):
        """Fit Freeze-Thaw Two Step Gaussian process regression model.

        Parameters
        ----------
        X : array-like, shape = (N, 1)
            Training data

        y : matrix-like, shape = (N, ), dtype = object (list)
            Target values

        Returns
        -------
        self: returns an instance of self.
        """

        X, y = self.transform_data(X, y)

        self.regr.fit(X, y)

        return self

    def transform_data(self, X_in, y_in):
        '''
        transform data
        '''
        X = X_in
        y = np.array([y_i[-1] for _, y_i in enumerate(y_in)])
        print(X)
        print(y)
        return X, y

    def predict(self, X):
        mean = self.regr.predict(X)
        print('mean')
        print(mean)
        std = 0

        res = []
        for _, estimator in enumerate(self.regr.estimators_):
            tmp = estimator.predict(X)
            #print('tmp')
            #print(tmp)
            res.append(tmp) 
        mean = np.mean(res, axis=0)
        std = np.std(res, axis=0)
        print('res')
        print(res)
        print('mean')
        print(mean)
        print('std')
        print(std)

        return mean, std
