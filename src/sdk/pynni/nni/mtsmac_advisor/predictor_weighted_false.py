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
predictor.py
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# pylint:disable=invalid-name


class BayesianForestRegressor():
    '''
    Regressor takes y, mean, std as training set, predicts distribution
    '''

    def __init__(self, random_state=0):
        self.regr = RandomForestRegressor(
            n_estimators=10, max_depth=100, min_samples_split=2, max_features=5/6, bootstrap=True, random_state=random_state)

    def fit(self, X, y, weights):
        self.regr.fit(X, y, sample_weight=weights)

    def predict(self, X):
        res = np.empty([0, len(X)])
        for _, estimator in enumerate(self.regr.estimators_):
            tmp = estimator.predict(X)
            res = np.vstack((res, tmp))
        mean = np.mean(res, axis=0)
        std = np.std(res, axis=0)

        return mean, std


class PredictorWeighted():
    """
    Random Forest Predictor
    """

    def __init__(self):
        """
        Parameters
        ----------
        """
        self.epochs = None
        self.regr = BayesianForestRegressor()

    # pylint:disable=attribute-defined-outside-init
    def fit(self, X, y):
        """Fit DSMAC regression model.

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
        # max epochs in training data
        self.epochs = max([len(y_i) for y_i in y])
        self.X = X
        self.y = y

        # distribution prediction of all running X
        self.num_completed = 0

        for y_i in self.y:
            if len(y_i) == self.epochs:
                self.num_completed += 1

        y_completed = np.empty(0)
        
        for X_i, y_i in zip(self.X, self.y):
            if len(y_i) >= self.epochs:
                y_completed = np.append(y_completed, y_i[-1])
            else:
                break
        print("y_completed:\n", y_completed)

        # fit a predictor for new configs
        X_train = X[:self.num_completed]
        y_train = y_completed
        self.regr.fit(X_train, y_train, None)

        print('-------------------SMAC Weighted--------------------')
        print("X_train:", X_train)
        print("y_train:", y_train)

        return self

    def predict(self, X):
        """ predict
        Parameters
        ----------
        X : array-like, shape = (N, N_features)
            Training data

        Returns
        -------
        result : numpy array,  if multi_task && !final_only, mean, std of shape(len(X), len(epochs)) ; else shape(len(X),)
        """
        mean, std = self.regr.predict(X)
        print('-------------------SMAC Weighted--------------------')
        print("mean:\n", mean)
        print("std:\n", std)
        
        return mean, std
