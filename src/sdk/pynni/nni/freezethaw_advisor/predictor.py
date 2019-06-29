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
import warnings

from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import WhiteKernel

from .kernels import KTC


class Predictor():
    """
    Freeze-Thaw Two Step Gaussian Process Predictor
    """

    def __init__(self, search_space, random_state=None):
        """
        Parameters
        ----------
        """
        self.X_train_ = None
        self.y_train_ = None
        self._y_train_mean = None

        self.kas_ = Matern(nu=2.5)  # kernel for asymptotes
        self.ktc_ = WhiteKernel()  # kernel for training curves

        self.alpha_ = None
        self._K_inv = None

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
        self._y_train_mean = np.zeros(1)

        # Precompute quantities required for predictions which are independent
        # of actual query points
        K_x = self.kas_(X)
        K_t = self.ktc_(X)

        self.alpha_ = K_x
        self.beta_ = K_t

        return self

    def predict_asymptote(self, X):
        '''
        posterior distribution of a new hyperparameter setting : Equation 14(19)
        '''

        K_trans = self.kas_(X, self.X_train_)

        # mean
        y_mean = K_trans.dot(self.alpha_)
        y_mean = self._y_train_mean + y_mean  # undo normal.

        # Compute variance of predictive distribution
        y_var = self.kas_.diag(X)
        y_var -= np.einsum("ij,ij->i", np.dot(K_trans, self._K_inv), K_trans)

        # Check if any of the variances is negative because of numerical issues. If yes: set the variance to 0.
        y_var_negative = y_var < 0
        if np.any(y_var_negative):
            warnings.warn("Predicted variances smaller than 0. "
                          "Setting those variances to 0.")
            y_var[y_var_negative] = 0.0

        return y_mean, y_var

    def predict_point_old(self, x):
        '''
        posterior distribution for a new point in a training curve: Equation 15(20)
        '''

    def predict_point_new(self, x):
        '''
        posterior distribution for a new point in the absence of any observations : Equation 16(21)
        '''
