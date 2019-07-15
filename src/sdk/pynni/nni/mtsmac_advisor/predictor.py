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

import numpy as np
from sklearn.ensemble import RandomForestRegressor

# pylint:disable=invalid-name


class Predictor():
    """
    Freeze-Thaw Bayesian Optimization: Two Step Gaussian Process Predictor
    """

    def __init__(self, final_only=False):
        """
        Parameters
        ----------
        """
        self.regr = RandomForestRegressor(
            n_estimators=10, max_depth=3, min_samples_split=2, max_features=5/6, random_state=0)
        self.final_only = final_only
        self.epochs = None

    def fit(self, X, y):
        """Fit Freeze-Thaw Two Step Gaussian process regression model.

        Parameters
        ----------
        X : array-like, shape = (N, N_features)
            Training data

        y : matrix-like, shape = (N, ), dtype = object (list)
            Target values

        Returns
        -------
        self: returns an instance of self.
        """
        self.epochs = len(y[0])  # TODO: maybe not exact
        X, y = self.transform_data(X, y)
        self.regr.fit(X, y)

        return self

    def transform_data(self, X, y):
        '''
        transform data
        '''
        if self.final_only:
            X_new = X
            y_new = np.array([y_i[-1] for _, y_i in enumerate(y)])

        else:
            N = X.shape[0]
            N_features = X[0].shape[0]
            X_new = np.empty([0, N_features+1])
            y_new = np.empty(0)
            for i in range(N):
                for t in range(len(y[i])):
                    X_new = np.vstack((X_new, np.append(X[i], [t])))
                    y_new = np.append(y_new, y[i][t])

        print('shape of new X, y:')
        print(X_new.shape)
        print(y_new.shape)
        # print(X_new)
        # print(y_new)

        return X_new, y_new

    def predict(self, X):
        """ predict
        Parameters
        ----------
        X : array-like, shape = (N, N_features)
            Training data

        Returns
        -------
        result : mean, std of shape(len(X), len(epochs)) if final_only == False; else shape(len(X),)
        """
        if self.final_only:
            res = np.empty([0, len(X)])
            for _, estimator in enumerate(self.regr.estimators_):
                tmp = estimator.predict(X)
                res = np.vstack((res, tmp))
                print('tmp, res')
                print(tmp)
                print(res)
            mean = np.mean(res, axis=0)
            std = np.std(res, axis=0)
            assert (mean == self.regr.predict(X)).all()
        else:
            mean = np.empty([len(X), 0])
            std = np.empty([len(X), 0])
            for t in range(self.epochs):
                X_t = np.hstack((X, np.ones([len(X), 1])*t))
                print('X_t')
                print(X_t)
                res = np.empty([0, len(X)])
                for _, estimator in enumerate(self.regr.estimators_):
                    tmp = estimator.predict(X_t)
                    res = np.vstack((res, tmp))
                mean_t = np.mean(res, axis=0)
                std_t = np.std(res, axis=0)
                mean = np.hstack((mean, mean_t.reshape(-1, 1)))
                std = np.hstack((std, std_t.reshape(-1, 1)))

        # print('shape of mean, std:')
        # print(mean.shape)
        # print(std.shape)
        # print('res')
        # print(res)
        # print('mean')
        # print(mean)
        # print('std')
        # print(std)

        return mean, std
