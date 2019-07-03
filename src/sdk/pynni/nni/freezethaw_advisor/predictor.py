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
from scipy.linalg import cholesky, cho_solve, solve_triangular
import warnings

from sklearn.base import clone
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.utils.validation import check_X_y, check_array

from .kernels import KTC


class Predictor():
    """
    Freeze-Thaw Two Step Gaussian Process Predictor
    """

    def __init__(self, kernel_x, kernel_t, search_space, normalize_y=False, copy_X_train=True, random_state=None):
        """
        Parameters
        ----------
        """
        # TODO: check alpha : positive ?

        self.kernel_as = Matern(nu=2.5)  # kernel for asymptotes
        self.kernel_tc = WhiteKernel()  # kernel for training curves

        self.normalize_y = normalize_y
        self.copy_X_train = copy_X_train
        self.random_state = random_state

    def fit(self, X, y):
        """Fit Freeze-Thaw Two Step Gaussian process regression model.

        Parameters
        ----------
        X : array-like, shape = (N, 1)
            Training data

        y : matrix-like, shape = (N, [n_output_dims])
            Target values

        Returns
        -------
        self: returns an instance of self.
        """
        self.kernel_as_ = clone(self.kernel_as)
        self.kernel_tc_ = clone(self.kernel_tc)

        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)

        # Normalize target value
        if self.normalize_y:
            self._y_train_mean = np.mean(y, axis=0)
            # demean yF
            y = y - self._y_train_mean
        else:
            self._y_train_mean = np.zeros(1)

        self.X_train_ = np.copy(X) if self.copy_X_train else X
        self.y_train_ = np.copy(y) if self.copy_X_train else y

        # Precompute quantities required for predictions which are independent of actual query points
        K_x = self.kernel_as_(self.X_train_)

        # Posterior distribution : P(f|y, X) = N(f;mu, C)
        # Equation 13(18)
        self.Lambda_ = []  # TODO
        self.KLam_inv = np.linalg.pinv(
            K_x + np.linalg.pinv(self.Lambda_))  # K_x + Lambda_inv_
        tmp = np.matmul(K_x, self.KLam_inv)
        self.C_ = K_x - np.matmul(tmp, K_x)
        # Equation 12(17)
        gamma = []  # TODO
        self.mu_ = np.matmul(self.C_, gamma)  # TODO: plus prior mean

        try:
            self.L_ = cholesky(K_x, lower=True)
            self._K_inv = None
        except np.linalg.LinAlgError as exc:
            exc.args = ("The kernel, %s, is not returning a "
                        "positive definite matrix. Try gradually "
                        "increasing the 'alpha' parameter of your "
                        "GaussianProcessRegressor estimator."
                        % self.kernel_as_,) + exc.args
            raise
        # TODO minus prior mean
        self.alpha_ = cho_solve((self.L_, True), self.mu_)

        return self

    def predict_asymptote_old(self, X):
        '''
        posterior distribution of a old hyperparameter's asymptote : Equation 12, 13(17, 18)
        '''

    def predict_asymptote_new(self, X):
        '''
        posterior distribution of a new hyperparameter's asymptote : Equation 14(19)
        '''
        K_trans = self.kernel_as_(X, self.X_train_)  # TODO: why not inverse

        # Compute mean of predictive distribution
        y_mean = K_trans.dot(self.alpha_)
        y_mean = self._y_train_mean + y_mean  # undo normal.

        # Compute variance of predictive distribution
        y_std = self.kernel_as_.diag(X)
        y_std -= np.einsum("ij,ij->i", np.dot(K_trans,
                                              self.KLam_inv), K_trans)

        # Check if any of the variances is negative because of numerical issues. If yes: set the variance to 0.
        y_std_negative = y_std < 0
        if np.any(y_std_negative):
            warnings.warn("Predicted variances smaller than 0. "
                          "Setting those variances to 0.")
            y_std[y_std_negative] = 0.0

        return y_mean, y_std

    def predict_point_old(self, X):
        '''
        posterior distribution for a new point in a training curve: Equation 15(20)
        '''
        Omga = []  # TODO

    def predict_point_new(self, X):
        '''
        posterior distribution for a new point in the absence of any observations : Equation 16(21)
        '''
        mean, std = self.predict_asymptote(X)
        K_t = self.kernel_tc_(self.X_train_)
        std += K_t

        return mean, std
