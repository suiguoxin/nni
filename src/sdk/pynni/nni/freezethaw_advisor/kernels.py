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
kernels.py
"""

import numpy as np
from sklearn.gaussian_process.kernels import StationaryKernelMixin, Kernel, Hyperparameter

# pylint:disable=invalid-name


class KTC(StationaryKernelMixin, Kernel):
    '''Kernel for Training Curve.

    The kernel given by:

    k(t, t') = beta^alpha/(t+t'+beta)^alpha

    Parameters
    ----------
    alpha : float > 0, default: 0.5
        Scale mixture parameter
    beta : float > 0, default: 0.5
        Scale mixture parameter
    '''

    def __init__(self, alpha=0.5, beta=0.5, alpha_bounds=(1e-6, 1), beta_bounds=(1e-6, 1)):
        self.alpha = alpha
        self.beta = beta
        self.alpha_bounds = alpha_bounds
        self.beta_bounds = beta_bounds

    @property
    def hyperparameter_alpha(self):  # pylint:disable=missing-docstring
        return Hyperparameter(
            "alpha", "numeric", self.alpha_bounds)

    @property
    def hyperparameter_beta(self):  # pylint:disable=missing-docstring
        return Hyperparameter(
            "beta", "numeric", self.beta_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : array, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined. Only supported when Y is None.

        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : array (opt.), shape (n_samples_X, n_samples_X, n_dims)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        X = np.atleast_2d(X)
        if Y is not None and eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")

        if Y is None:
            tmp = np.float_power(self.beta, self.alpha)
            K = np.empty(shape=(X.shape[0], X.shape[0]))
            for i in range(X.shape[0]):
                for j in range(X.shape[0]):
                    K[i, j] = tmp / \
                        np.float_power(X[i][0] + X[j][0] +  # X is shape(,1) here
                                       self.beta, self.alpha)
            if eval_gradient:
                n_gradient_dim = 2
                K_gradient = np.zeros((K.shape[0], K.shape[0], n_gradient_dim))
                for i in range(X.shape[0]):
                    for j in range(X.shape[0]):
                        t_1 = X[i][0]
                        t_2 = X[j][0]
                        K_gradient[i, j, 0] = K[i, j] * \
                            np.log(self.beta / (self.beta + t_1 + t_2))
                        K_gradient[i, j, 1] = self.alpha * K[i, j] * \
                            (t_1 + t_2 / (self.beta*(self.beta + t_1 + t_2)))
                return K, K_gradient
            else:
                return K
        else:
            Y = np.atleast_2d(Y)
            tmp = np.float_power(self.beta, self.alpha)
            K = np.empty(shape=(X.shape[0], Y.shape[0]))
            # print(X.shape, Y.shape)
            for i in range(X.shape[0]):
                for j in range(Y.shape[0]):
                    K[i, j] = tmp / \
                        np.float_power(X[i][0] + Y[j][0] +  # X, Y is shape(,1) here
                                       self.beta, self.alpha)
            return K

    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Returns
        -------
        K_diag : array, shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        X = np.atleast_2d(X)

        tmp = np.float_power(self.beta, self.alpha)
        diag = np.empty(shape=(X.shape[0]))
        for i in range(X.shape[0]):
            diag[i] = tmp / np.float_power(X[i][0] * 2 +
                                           self.beta, self.alpha)
        return diag

    def __repr__(self):
        return "{0}(alpha={1}, beta={2})".format(self.__class__.__name__, self.alpha, self.beta)
