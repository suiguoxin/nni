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
from sklearn.ensemble import RandomForestRegressor

# pylint:disable=invalid-name


class PredictorGradient():
    """
    Random Forest Predictor
    """

    def __init__(self, random_state=0):
        """
        Parameters
        ----------
        """
        self.regr = RandomForestRegressor(
            n_estimators=10, max_depth=100, min_samples_split=2, max_features=5/6, bootstrap=True, random_state=random_state)
        self.epochs = None

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
        self.X = X
        self.y = y
        self.epochs = max([len(y_i) for y_i in y])
        X, y = self.transform_data(X, y)
        self.regr.fit(X, y)

        return self

    def transform_data(self, X, y):
        '''
        transform data for training model
        '''
        N = X.shape[0]
        N_features = X[0].shape[0]

        X_new = np.empty([0, N_features+1])
        y_new = np.empty(0)
        for i in range(N):
            for t in range(len(y[i])):
                X_new = np.vstack((X_new, np.append(X[i], [t])))
                if t == 0:
                    y_new = np.append(y_new, y[i][t])
                else: 
                    y_new = np.append(y_new, y[i][t]-y[i][t-1])

        return X_new, y_new

    def predict(self, X, final_only=True):
        """ predict
        Parameters
        ----------
        X : array-like, shape = (N, N_features)
            Training data

        Returns
        -------
        result : numpy array,  if final_only, mean, std of shape(len(X), len(epochs)) ; else shape(len(X),)
        """

        # check if X_i is a old point
        mean = np.empty([len(X), self.epochs])
        var = np.empty([len(X), self.epochs])
        
        for t in range(self.epochs):
            X_t = np.hstack((X, np.ones([len(X), 1])*t))
            res = np.empty([0, len(X)])
            for _, estimator in enumerate(self.regr.estimators_):
                tmp = estimator.predict(X_t)
                res = np.vstack((res, tmp))
            if t == 0:
                mean[:, t] = np.clip(np.mean(res, axis=0), 0, 1)
                var[:, t] = np.var(res, axis=0)
            else:
                mean[:, t] = np.clip(mean[:, t-1] + np.mean(res, axis=0), 0, 1)
                var[:, t] = var[:, t-1] + np.var(res, axis=0)
        std = np.sqrt(var)
        if final_only:
            return mean[:, self.epochs-1], std[:, self.epochs-1]
        else:
            return mean, std

        return mean, std
