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

import warnings
import numpy as np
from scipy.linalg import block_diag

from sklearn.base import clone
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from .kernels import KTC

# pylint:disable=invalid-name
# pylint:disable=attribute-defined-outside-init
# TODO: successive matmul? pinv ?


class Predictor():
    """
    Freeze-Thaw Two Step Gaussian Process Predictor
    """

    def __init__(self, normalize_y=False, copy_X_train=True, random_state=None):
        """
        Parameters
        ----------
        """
        # TODO: check alpha : positive definite ?
        self.kernel_as = Matern(nu=2.5)  # kernel for asymptotes
        # kernel for training curves TODO: add KTC
        self.kernel_tc = KTC(alpha=1, beta=0.5) + WhiteKernel(1e-4)

        self.normalize_y = normalize_y
        self.copy_X_train = copy_X_train
        self.random_state = random_state

    def fit(self, X, y):
        """Fit Freeze-Thaw Two Step Gaussian process regression model.

        Parameters
        ----------
        X : array-like, shape = (N, 1)
            Training data

        y : matrix-like, shape = (N, )
            Target values

        Returns
        -------
        self: returns an instance of self.
        """
        self.kernel_as_ = clone(self.kernel_as)
        self.kernel_tc_ = clone(self.kernel_tc)

        self.X_train_ = np.copy(X) if self.copy_X_train else X
        self.y_train_ = np.copy(y) if self.copy_X_train else y

        y_train_flatten_ = []
        for t in self.y_train_:
            y_train_flatten_ += t
        self.y_train_flatten_ = np.array(y_train_flatten_).reshape(-1, 1)

        # Precompute quantities required for predictions which are independent of actual query points
        # Posterior distribution : P(f|y, X) = N(f;mu, C)
        # Equation 13(18)
        # O (NT, N)
        O = block_diag(*[np.ones((len(self.y_train_[i]), 1))
                         for i in range(self.y_train_.shape[0])])

        # K_x, K_t
        self.K_x = self.kernel_as_(self.X_train_)
        self.K_t = block_diag(*[self.kernel_tc_(np.reshape(self.y_train_[i], (-1, 1)))
                                for i in range(self.y_train_.shape[0])])

        # m
        self.m = np.zeros((self.y_train_.shape[0], 1))

        # gamma
        tmp = np.matmul(np.transpose(O), np.linalg.pinv(self.K_t))
        gamma = np.matmul(tmp, self.y_train_flatten_ - np.matmul(O, self.m))

        # Lambda
        tmp = np.matmul(np.transpose(O), np.linalg.pinv(self.K_t))
        self.Lambda = np.matmul(tmp, O)

        # C， Equation 13(18)
        tmp = np.matmul(self.K_x, np.linalg.pinv(
            self.K_x + np.linalg.pinv(self.Lambda)))
        self.C = self.K_x - np.matmul(tmp, self.K_x)

        # Check if any of the variances is negative because of numerical issues. If yes: set the variance to 0.
        '''
        C_negative = self.C < 0
        if np.any(C_negative):
            warnings.warn("Predicted variances smaller than 0. "
                          "Setting those variances to 0.")
            self.C[C_negative] = 0.0
        '''
        # mu, Equation 12(17)
        self.mu = np.matmul(self.C, gamma)
        self.mu += self.m

        return self

    def predict_asymptote_old(self):
        '''
        posterior distribution of a old hyperparameter's asymptote : Equation 12, 13(17, 18)
        '''
        return self.mu, self.C

    def predict_asymptote_new(self, X):
        '''
        posterior distribution of a new hyperparameter's asymptote : Equation 14(19)
        '''
        K_x_s = self.kernel_as_(self.X_train_, X)  # TODO: why not inverse ?
        K_x_s_trans = np.transpose(K_x_s)

        tmp = np.matmul(K_x_s_trans, np.linalg.inv(self.K_x))
        f_mean = np.matmul(tmp, self.mu - self.m)

        tmp = np.matmul(np.transpose(K_x_s), np.linalg.inv(
            self.K_x + np.linalg.inv(self.Lambda)))
        f_var = np.matmul(tmp, K_x_s)

        # Check if any of the variances is negative because of numerical issues. If yes: set the variance to 0.
        '''
        f_var_negative = f_var < 0
        if np.any(f_var_negative):
            warnings.warn("Predicted variances smaller than 0. "
                          "Setting those variances to 0.")
            f_var[f_var_negative] = 0.0
        '''
        return f_mean, f_var

    def predict_point_old(self, idx):
        '''
        posterior distribution for a new point in a training curve: Equation 15(20)
        '''
        y_t = np.reshape(self.y_train_[idx], (-1, 1))
        T = y_t.shape[0]
        T_arr = np.arange(1, T+1).reshape(-1, 1)
        K_t_n = self.kernel_tc_(T_arr)
        K_t_n_s = self.kernel_tc_(T_arr, np.array([T+1]))

        tmp = np.matmul(np.transpose(K_t_n_s), np.linalg.inv(K_t_n))
        Omega = np.matmul(tmp, np.ones(T).reshape(-1, 1))
        Omega = 1 - Omega

        mu_n = self.mu[idx]
        C_n_n = self.C[np.ix_([idx], [idx])]
        y_n = self.y_train_[idx]

        tmp = np.matmul(np.transpose(K_t_n_s), np.linalg.inv(K_t_n))
        mean = np.matmul(tmp, y_n) + np.matmul(Omega, mu_n)

        K_t_n_s_s = self.kernel_tc_(np.array([T+1]), np.array([T+1]))

        tmp = np.matmul(np.transpose(K_t_n_s), np.linalg.inv(K_t_n))
        var = K_t_n_s_s - np.matmul(tmp, K_t_n_s) + \
            np.matmul(np.matmul(Omega, C_n_n), np.transpose(Omega))

        return mean, var

    def predict_point_new(self, X):
        '''
        posterior distribution for a new point in the absence of any observations : Equation 16(21)
        '''
        mean, var = self.predict_asymptote_new(X)
        K_t = self.kernel_tc_(np.array([1]))
        var += K_t

        return mean, var
