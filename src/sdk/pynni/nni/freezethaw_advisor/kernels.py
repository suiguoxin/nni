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
from scipy.spatial.distance import pdist, squareform

from sklearn.gaussian_process.kernels import StationaryKernelMixin, NormalizedKernelMixin, Kernel, Hyperparameter


def _check_length_scale(X, length_scale):
    length_scale = np.squeeze(length_scale).astype(float)
    if np.ndim(length_scale) > 1:
        raise ValueError("length_scale cannot be of dimension greater than 1")
    if np.ndim(length_scale) == 1 and X.shape[1] != length_scale.shape[0]:
        raise ValueError("Anisotropic kernel must have the same number of "
                         "dimensions as data (%d!=%d)"
                         % (length_scale.shape[0], X.shape[1]))
    return length_scale


class KTC(StationaryKernelMixin, Kernel):
    '''Kernel for Training Curve.

    The kernel given by:

    k(t, t') = beta^alpha/(t+t'+beta)^alpha + delta(t,t')*sigma^2

    Parameters
    ----------
    alpha : float > 0, default: 1.0
        Scale mixture parameter
    '''

    def __init__(self, alpha=0.5, beta=0.5, alpha_bounds=(1e-5, 1), beta_bounds=(1e-5, 1)):
        self.alpha = alpha
        self.beta = beta
        self.alpha_bounds = alpha_bounds
        self.beta_bounds = beta_bounds

        # self.hyperparameter_alpha = Hyperparameter("alpha", "numeric", self.alpha_bounds)
        # self.hyperparameter_beta = Hyperparameter("beta", "numeric", self.beta_bounds)

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
                        t1 = X[i][0]
                        t2 = X[j][0]
                        K_gradient[i, j, 0] = K[i, j] * \
                            np.log(self.beta / (self.beta + t1 + t2))
                        K_gradient[i, j, 1] = K[i, j] * \
                            (t1 + t2 / (self.beta*(self.beta + t1 + t2)))
                return K, K_gradient
            else:
                return K
        else:
            # TODO: not used here
            return np.zeros((X.shape[0], Y.shape[0]))

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
        return "{0}(alpha=[{1}], beta={2})".format(self.__class__.__name__,
                                                   ", ".join(map("{0:.3g}".format, self.alpha)), self.beta)
