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

# pylint:disable=invalid-name


class BayesianForestRegressor():
    '''
    Regressor takes y, mean, std as training set, predicts distribution
    '''

    def __init__(self):
        self.size = 10
        self.estimators_ = [DecisionTreeRegressor(
            max_depth=None, min_samples_split=2, max_features=5/6)]*self.size
        self.random_state = np.random.RandomState()

    def fit(self, X, y, X_running, means, stds):
        for i in range(self.size):
            X_train, y_train = X, y
            if X_running.shape[0]:
                y_fant = self.random_state.normal(means[:len(X_running)], stds[:len(X_running)])
                X_train = np.vstack((X, X_running))
                y_train = np.append(y, y_fant)

            self.estimators_[i].fit(X_train, y_train)

    def predict(self, X):
        res = np.empty([0, len(X)])
        for _, estimator in enumerate(self.estimators_):
            tmp = estimator.predict(X)
            res = np.vstack((res, tmp))
        mean = np.mean(res, axis=0)
        std = np.std(res, axis=0)

        return mean, std


class PredictorNew():
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

        # sort (X, y) by len(y_i), from large to small
        self.X = np.array([x for x, _ in sorted(
            zip(X, y), key=lambda pair: len(pair[1]), reverse=True)])
        self.y = sorted(y, key=len, reverse=True)

        # distribution prediction of all running X
        self.means = []
        self.stds = []

        # take all the intermediate result as featuers for the predictor, except for the final result
        size_features = X.shape[1] + self.epochs - 1
        X_completed = np.empty([0, size_features])
        y_completed = np.empty(0)
        self.num_completed = 0
        for X_i, y_i in zip(self.X, self.y):
            if len(y_i) >= self.epochs:
                X_completed = np.vstack((X_completed, np.append(X_i, y_i[:-1])))
                y_completed = np.append(y_completed, y_i[-1])
                self.num_completed += 1
            else:
                break
        # print("X_completed:\n", X_completed)
        # print("y_completed:\n", y_completed)

        # get the distribution prediction of all running X
        for X_i, y_i in zip(self.X[self.num_completed:], self.y[self.num_completed:]):
            size_new_X = len(X_i) + len(y_i)
            X_running = np.empty([0, size_new_X])
            for X_j, y_j in zip(X[self.num_completed:], y[self.num_completed:]):
                if len(y_j) > self.epochs:  # maybe = can be included
                    X_running = np.vstack((X_running, np.append(X_j, y_j[:len(y_i)])))
                else:
                    break

            regr = BayesianForestRegressor()
            regr.fit(X_completed[:, :size_new_X], y_completed,
                     X_running, self.means, self.stds)

            X_predict = np.append(X_i, y_i).reshape(1, -1)
            mean, std = regr.predict(X_predict)

            self.means.append(mean[0])
            self.stds.append(std[0])
        # print("self.means:\n", self.means)
        # print("self.stds:\n", self.stds)

        # fit a predictor for new configs
        self.regr.fit(self.X[:self.num_completed], y_completed,
                      self.X[self.num_completed:], self.means, self.stds)

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

        for i, X_i in enumerate(self.X[self.num_completed:]):
            if np.array_equal(X, X_i):
                return self.means[i], self.stds[i]

        mean, std = self.regr.predict(X)
        return mean, std


def test():
    X = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5],
                  [4, 5, 6], [5, 6, 7], [6, 7, 8]])
    y = np.array([list([1, 2, 3, 4, 5]), list([2, 3, 4, 5, 6]), list(
        [3, 4, 5]), list([4, 5, 6]), list([5]), list([6])])

    print(X)
    print(X.shape)
    print(y)
    print(y.shape)

    predictor = PredictorNew()
    predictor.fit(X, y)

    X_test = [[2, 3, 4], [7, 8, 9]]
    mean, std = predictor.predict(X_test)

# test()
