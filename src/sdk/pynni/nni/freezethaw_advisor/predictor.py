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
from functools import reduce
from operator import itemgetter

import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import fmin_l_bfgs_b
from sklearn.utils import check_random_state
from sklearn.base import clone
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.exceptions import ConvergenceWarning

from .kernels import KTC

# pylint:disable=invalid-name
# pylint:disable=attribute-defined-outside-init


class Predictor():
    """
    Freeze-Thaw Bayesian Optimization: Two Step Gaussian Process Predictor
    """

    def __init__(self,
                 alpha=1e-10,
                 optimizer="fmin_l_bfgs_b",
                 n_restarts_optimizer=0,
                 normalize_y=False,
                 copy_X_train=True,
                 random_state=None):
        """
        Parameters
        ----------
        """
        # TODO: check alpha : positive definite ?
        self.kernel_as = Matern(nu=2.5)  # kernel for asymptotes
        self.kernel_tc = KTC(alpha=0.5, beta=0.5) + \
            WhiteKernel(1e-4)  # kernel for trainning curves
        self.alpha = alpha

        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.normalize_y = normalize_y
        self.copy_X_train = copy_X_train
        self.random_state = random_state

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
        self.kernel_as_ = clone(self.kernel_as)
        self.kernel_tc_ = clone(self.kernel_tc)

        # TODO: add mean in theta
        self.theta = np.hstack(
            [self.kernel_as_.theta, self.kernel_tc_.theta])
        self.bounds = np.vstack(
            [self.kernel_as_.bounds, self.kernel_tc_.bounds])
        '''
        print('self.kernel_as_')
        print('self.kernel_as_.theta')
        print('self.theta')
        print('self.bounds')
        '''
        self._rng = check_random_state(self.random_state)

        self.X_train_ = np.copy(X) if self.copy_X_train else X
        self.y_train_ = np.copy(y) if self.copy_X_train else y

        y_train_flatten_ = []
        for y_i in self.y_train_:
            y_train_flatten_ += y_i
        self.y_train_flatten_ = np.array(y_train_flatten_).reshape(-1, 1)

        # mean_prior
        self.mean_prior = np.zeros((self.y_train_.shape[0], 1))

        # fit theta
        if self.optimizer is not None:
            '''
            def obj_func(theta):
                return -self.log_marginal_likelihood(theta)

            # First optimize starting from theta specified in kernel
            optima = [(self._constrained_optimization(obj_func,
                                                      self.theta,
                                                      self.bounds))]
            # Additional runs are performed from log-uniform chosen initial
            # theta
            if self.n_restarts_optimizer > 0:
                if not np.isfinite(self.bounds).all():
                    raise ValueError(
                        "Multiple optimizer restarts (n_restarts_optimizer>0) "
                        "requires that all bounds are finite.")
                bounds = self.bounds
                for _ in range(self.n_restarts_optimizer):
                    theta_initial = \
                        self._rng.uniform(bounds[:, 0], bounds[:, 1])
                    optima.append(
                        self._constrained_optimization(obj_func, theta_initial,
                                                       bounds))
            # Select result from run with minimal (negative) log-marginal
            # likelihood
            lml_values = list(map(itemgetter(1), optima))
            self.theta = optima[np.argmin(lml_values)][0]
            thetas = np.hsplit(
                self.theta, [self.kernel_as_.theta.shape[0], self.theta.shape[0]])
            self.kernel_as_.theta = thetas[0]
            self.kernel_tc_.theta = thetas[1]
            self.log_marginal_likelihood_value_ = -np.min(lml_values)
            '''
            # First optimize starting from theta specified in kernel
            optima = [(self.theta, -self.log_marginal_likelihood(self.theta))]
            # Additional runs are performed from log-uniform chosen initial
            self.n_starting_points = 100
            if self.n_starting_points > 0:
                if not np.isfinite(self.bounds).all():
                    raise ValueError(
                        "Multiple optimizer restarts (n_restarts_optimizer>0) "
                        "requires that all bounds are finite.")
                bounds = self.bounds
                for _ in range(self.n_starting_points):
                    theta_initial = \
                        self._rng.uniform(bounds[:, 0], bounds[:, 1])
                    optima.append(
                        (theta_initial, -self.log_marginal_likelihood(theta_initial)))
            # Select result from run with minimal (negative) log-marginal
            # likelihood

            lml_values = list(map(itemgetter(1), optima))
            # print('optima')
            # print(optima)
            # print('lml_values')
            # print(lml_values)
            self.theta = optima[np.argmin(lml_values)][0]
            thetas = np.hsplit(
                self.theta, [self.kernel_as_.theta.shape[0], self.theta.shape[0]])
            self.kernel_as_.theta = thetas[0]
            self.kernel_tc_.theta = thetas[1]
            self.log_marginal_likelihood_value_ = -np.min(lml_values)
            print('kernels after fitting')
            print(self.kernel_as_)
            print(self.kernel_tc_)

        else:
            self.log_marginal_likelihood_value_ = \
                self.log_marginal_likelihood(self.theta)

        # compute gamma, Lambda, K_x, K_t
        self.__pre_compute()

        # Precompute quantities required for predictions which are independent of actual query points
        # Posterior distribution : P(f|y, X) = N(f;mu, C)
        # C， Equation 13(18)
        self.C = np.linalg.inv(self.K_x_inv + self.Lambda)
        # Check if any of the variances is negative because of numerical issues. If yes: set the variance to 0.
        C_negative = self.C <= 0
        if np.any(C_negative):
            warnings.warn("Predicted variances smaller than 0. "
                          "Setting those variances to 0.")
            self.C[C_negative] = 0.0

        # mu, Equation 12(17)
        self.mu = np.matmul(self.C, self.gamma)
        self.mu += self.mean_prior

        return self

    def __pre_compute(self):
        '''
        Precompute gamma, Lambda, K_x, K_t, save in self.
        '''
        O = block_diag(*[np.ones((len(self.y_train_[i]), 1))
                         for i in range(self.y_train_.shape[0])])
        O_trans = np.transpose(O)

        # K_x, K_t
        self.K_x = self.kernel_as_(self.X_train_)
        self.K_x[np.diag_indices_from(self.K_x)] += self.alpha  # TODO
        self.K_t = block_diag(*[self.kernel_tc_(np.arange(1, len(self.y_train_[i])+1).reshape(-1, 1))
                                for i in range(self.y_train_.shape[0])])
        self.K_x_inv = np.linalg.inv(self.K_x)
        self.K_t_inv = np.linalg.inv(self.K_t)

        # gamma, Lambda
        self.gamma = reduce(np.matmul, [
                            O_trans, self.K_t_inv, self.y_train_flatten_ - np.matmul(O, self.mean_prior)])
        self.Lambda = reduce(np.matmul, [O_trans, self.K_t_inv, O])

        return self

    def __predict_asymptote_old(self, return_std=True):
        '''
        posterior distribution of a old hyperparameter's asymptote : Equation 12, 13(17, 18)

        Returns
        -------
        result : mean, std of shape(len(X),)
        '''
        if return_std:
            mean = np.array([self.mu[i][0] for i in range(len(self.mu))])
            std = np.sqrt(np.diag(self.C))
            return mean, std
        return self.mu, self.C

    def predict_asymptote_old(self, X):
        '''
        a wrapper for target_space

        Returns
        -------
        mean, std : numpy array of shape(len(X),)
        '''
        mean_full, std_full = self.__predict_asymptote_old(return_std=True)

        N = len(X)
        mean = np.empty(N)
        std = np.empty(N)
        for i in range(N):
            idx = -1
            for j in range(self.X_train_.shape[0]):
                if np.array_equal(X[i], self.X_train_[j]):
                    idx = j
                    mean[i], std[i] = mean_full[j], std_full[j]
                    break
            if idx == -1:
                raise ValueError("X[i] not found")
        return mean, std

    def predict_asymptote_new(self, X, return_std=True):
        '''
        posterior distribution of a new hyperparameter's asymptote : Equation 14(19)
        # TODO check if fitted for multiple points

        Returns
        -------
        result : mean, std of shape(len(X),)
        '''
        K_x_s = self.kernel_as_(self.X_train_, X)  # TODO: why not inverse ?
        K_x_s_trans = np.transpose(K_x_s)

        tmp = np.matmul(K_x_s_trans, self.K_x_inv)
        f_mean = np.matmul(tmp, self.mu - self.mean_prior)

        tmp = np.matmul(np.transpose(K_x_s), np.linalg.inv(
            self.K_x + np.linalg.inv(self.Lambda)))
        f_var = np.matmul(tmp, K_x_s)

        # Check if any of the variances is negative because of numerical issues. If yes: set the variance to 0.
        f_var_negative = f_var < 0
        if np.any(f_var_negative):
            warnings.warn("Predicted variances smaller than 0. "
                          "Setting those variances to 0.")
            f_var[f_var_negative] = 0.0

        if return_std:
            mean = np.array([f_mean[i][0] for i in range(len(f_mean))])
            std = np.sqrt(np.diag(f_var))
            return mean, std
        else:
            return f_mean, f_var

    def __predict_point_old(self, idx, return_std=True):
        '''
        posterior distribution for a new point in a training curve: Equation 15(20)

        Returns
        -------
        mean, var/std of type scalar
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

        if return_std:
            std = np.sqrt(var)
            return mean[0], std[0][0]
        return mean[0], var[0][0]

    def predict_point_old(self, X):
        '''
        a wrapper for target_space

        Returns
        -------
        mean, std : numpy array of shape(len(X),)
        '''
        N = len(X)
        mean = np.empty(N)
        std = np.empty(N)
        for i in range(N):
            idx = -1
            for j in range(self.X_train_.shape[0]):
                if np.array_equal(X[i], self.X_train_[j]):
                    idx = j
                    mean[i], std[i] = self.__predict_point_old(j)
                    break
            if idx == -1:
                raise ValueError("X[i] not found")
        return mean, std

    def predict_point_new(self, X, return_std=True):
        '''
        posterior distribution for a new point in the absence of any observations : Equation 16(21)

        Returns
        -------
        result : mean, std of shape(len(X),)
        '''
        f_mean, f_var = self.predict_asymptote_new(X, return_std=False)
        K_t = self.kernel_tc_(np.array([1]))
        f_var += K_t

        if return_std:
            mean = np.array([f_mean[i][0] for i in range(len(f_mean))])
            std = np.sqrt(np.diag(f_var))
            return mean, std
        return f_mean, f_var

    def log_marginal_likelihood(self, theta=None, eval_gradient=False):
        '''
        Returns log-marginal likelihood of theta for training data : Equation 11
        '''
        if theta is None:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated for theta!=None")
            return self.log_marginal_likelihood_value_

        thetas = np.hsplit(
            theta, np.array([self.kernel_as_.theta.shape[0], self.theta.shape[0]]))

        kernel_as = self.kernel_as_.clone_with_theta(thetas[0])
        kernel_tc = self.kernel_tc_.clone_with_theta(thetas[1])

        O = block_diag(*[np.ones((len(self.y_train_[i]), 1))
                         for i in range(self.y_train_.shape[0])])
        O_trans = np.transpose(O)

        # K_x, K_t
        K_x = kernel_as(self.X_train_)
        K_t = block_diag(*[kernel_tc(np.arange(1, len(self.y_train_[i])+1).reshape(-1, 1))
                           for i in range(self.y_train_.shape[0])])
        K_x_inv = np.linalg.inv(K_x)
        K_t_inv = np.linalg.inv(K_t)

        # gamma, Lambda
        gamma = reduce(np.matmul, [
            O_trans, K_t_inv, self.y_train_flatten_ - np.matmul(O, self.mean_prior)])
        Lambda = reduce(np.matmul, [O_trans, K_t_inv, O])

        # log_likelihood
        y_flatten_demean = self.y_train_flatten_ - \
            np.matmul(O, self.mean_prior)
        log_likelihood = -0.5 * \
            reduce(np.matmul, [np.transpose(
                y_flatten_demean), K_t_inv, y_flatten_demean])
        #print('log_likelihood step 1:', log_likelihood)

        log_likelihood += 0.5 * \
            reduce(np.matmul, [np.transpose(gamma),
                               np.linalg.inv(K_x_inv+Lambda), gamma])
        #print('log_likelihood step 2:', log_likelihood)

        _, tmp_0 = np.linalg.slogdet(np.linalg.inv(K_x)+Lambda)
        _, tmp_1 = np.linalg.slogdet(K_x)
        _, tmp_2 = np.linalg.slogdet(K_t)
        log_likelihood -= 0.5 * (tmp_0 + tmp_1 + tmp_2)
        #print('log_likelihood step 3:', log_likelihood)

        return log_likelihood[0][0]

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        if self.optimizer == "fmin_l_bfgs_b":
            # print('initial_theta')
            # print(initial_theta)
            # print('bounds')
            # print(bounds)
            theta_opt, func_min, convergence_dict = fmin_l_bfgs_b(
                obj_func, initial_theta, bounds=bounds)
            if convergence_dict["warnflag"] != 0:
                warnings.warn("fmin_l_bfgs_b terminated abnormally with the "
                              " state: %s" % convergence_dict,
                              ConvergenceWarning)
        elif callable(self.optimizer):
            theta_opt, func_min = self.optimizer(
                obj_func, initial_theta, bounds=bounds)
        else:
            raise ValueError("Unknown optimizer %s." % self.optimizer)

        return theta_opt, func_min
