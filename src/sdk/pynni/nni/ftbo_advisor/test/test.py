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
test.py
"""

import warnings
import numpy as np
from scipy.linalg import block_diag

from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from nni.ftbo_advisor.kernels import KTC
from nni.ftbo_advisor.predictor import Predictor
from nni.ftbo_advisor.test.util import create_fake_data_simple


# pylint:disable=missing-docstring
# pylint:disable=no-member
# pylint:disable=invalid-name


def kernel_ktc_test():
    # test of K(X, X), k(x,x) = 0
    ktc = KTC(alpha=1, beta=0)
    X = np.array([[1], [2]])
    K = np.array([[0, 0],
                  [0, 0]])
    K_cal = ktc(X)

    K_diag = np.array([0, 0])
    K_diag_cal = ktc.diag(X)

    assert np.array_equal(K, K_cal)
    assert np.array_equal(K_diag, K_diag_cal)
    print('test 1 pass !')

    # test of K(X, X), k(x,x) != 0
    ktc = KTC(alpha=1, beta=1)
    X = [[1], [2]]
    X = np.arange(3).reshape(-1, 1)
    K = np.array([[1, 1/2, 1/3],
                  [1/2, 1/3, 1/4],
                  [1/3, 1/4, 1/5]])

    K_cal = ktc(X)

    K_diag = np.array([1, 1/3, 1/5])
    K_diag_cal = ktc.diag(X)

    assert np.array_equal(K, K_cal)
    assert np.array_equal(K_diag, K_diag_cal)
    print('test 2 pass !')

    # test of K(X, Y), k(x,y) = 0
    ktc = KTC(alpha=1, beta=0)
    X = [[1], [2]]
    Y = [[1], [2]]
    K = np.array([[0, 0],
                  [0, 0]])

    K_cal = ktc(X, Y)

    assert np.array_equal(K, K_cal)
    print('test 3 pass !')

    # test of K(X, Y), k(x,y) != 0
    ktc = KTC(alpha=1, beta=1)
    X = [[1], [2]]
    Y = [[1],
         [2],
         [3]]
    K = np.array([[1/3, 1/4, 1/5],
                  [1/4, 1/5, 1/6]])

    K_cal = ktc(X, Y)

    assert np.array_equal(K, K_cal)
    print('test 4 pass !')


def predict_test():
    # X, y
    X, y = create_fake_data_simple()
    print(X)
    print(y)
    print('x.shape:', X.shape)
    print('y.shape:', y.shape)

    # X_train_, y_train_, y_train_flatten_
    X_train_ = np.copy(X)
    y_train_ = np.copy(y)
    y_train_flatten_ = []
    for t in y_train_:
        y_train_flatten_ += t
    y_train_flatten_ = np.array(y_train_flatten_).reshape(-1, 1)
    print('X_train_.shape:', X_train_.shape)
    print('y_train_flatten_.shape:', y_train_flatten_.shape)

    # m
    m = np.zeros((y_train_.shape[0], 1))
    print('m:')
    print(m)

    # O (NT, N)
    O = block_diag(*[np.ones((len(y_train_[i]), 1))
                     for i in range(y_train_.shape[0])])
    print('O:\n', O)

    # K_x, K_t
    kernel_as_ = Matern(nu=2.5)
    kernel_tc_ = KTC(alpha=0.5, beta=0.5) + WhiteKernel(1e-4)
    K_x = kernel_as_(X_train_)
    # K_t = block_diag(*[kernel_tc_(np.reshape(y_train_[i], (-1, 1)))
    #                   for i in range(y_train_.shape[0])])
    K_t = block_diag(*[kernel_tc_(np.arange(1, len(y_train_[i])+1).reshape(-1, 1))
                       for i in range(y_train_.shape[0])])
    print('K_x:\n', K_x)
    print('K_t:\n', K_t)

    # gamma
    tmp = np.matmul(np.transpose(O), np.linalg.inv(K_t))  # shape (N,NT)
    gamma = np.matmul(tmp, y_train_flatten_ - np.matmul(O, m))
    print('gamma:')
    print(gamma)

    # Lambda
    tmp = np.matmul(np.transpose(O), np.linalg.inv(K_t))
    Lambda = np.matmul(tmp, O)
    print('Lambda:')
    print(Lambda)

    # C
    C = np.linalg.inv(np.linalg.inv(K_x) + Lambda)
    print('C:\n', C)

    C_negative = C < 0
    if np.any(C_negative):
        warnings.warn("Predicted variances smaller than 0. "
                      "Setting those variances to 0.")
        C[C_negative] = 0.0
        print('C:\n', C)

    # mu
    mu = np.matmul(C, gamma)
    mu += m
    print('mu:\n', mu)

    # mean, std
    mean = np.array([mu[i][0] for i in range(len(mu))])
    std = np.sqrt(np.diag(C))
    print('mean:\n', mean)
    print('std:\n', std)

    predictor = Predictor(optimizer=None)
    predictor.fit(X, y)
    mean_pred, std_pred = predictor.predict_asymptote_old(X_train_)

    print('mean_pred:\n', mean_pred)
    print('std_pred:\n', std_pred)

    # assert np.array_equal(mean, mean_pred)
    # assert np.array_equal(std, std_pred)
    assert np.allclose(mean, mean_pred, rtol=1e-6)
    assert np.allclose(std, std_pred, rtol=1e-6)

    print('--------------test asymptote_old pass !----------------------')

    x = np.array([[0.7],
                  [0.8]])

    K_x_s = kernel_as_(X_train_, x)
    tmp = np.matmul(np.transpose(K_x_s), np.linalg.inv(K_x))
    mean = np.matmul(tmp, mu - m)
    print('mean: \n', mean)

    tmp = np.matmul(np.transpose(K_x_s), np.linalg.inv(
        K_x + np.linalg.inv(Lambda)))
    cov = np.matmul(tmp, K_x_s)
    print('cov: \n', cov)

    mean_pred, cov_pred = predictor.predict_asymptote_new(x, return_std=False)
    assert np.allclose(mean, mean_pred, rtol=1e-6)
    assert np.allclose(cov, cov_pred, rtol=1e-6)

    # mean, std
    mean = np.array([mean[i][0] for i in range(len(mean))])
    std = np.sqrt(np.diag(cov))
    print('mean:\n', mean)
    print('std:\n', std)

    mean_pred, std_pred = predictor.predict_asymptote_new(x)
    print('mean_pred:\n', mean_pred)
    print('std_pred:\n', std_pred)

    assert np.allclose(mean, mean_pred, rtol=1e-6)
    assert np.allclose(std, std_pred, rtol=1e-6)

    print('--------------test asymptote_new pass !----------------------')

    idx = 1

    y_t = np.reshape(y_train_[idx], (-1, 1))
    T = y_t.shape[0]
    T_arr = np.arange(1, T+1).reshape(-1, 1)
    print('T_arr:', T_arr)
    K_t_n = kernel_tc_(T_arr)
    K_t_n_s = kernel_tc_(T_arr, np.array([T+1]))
    print('K_t_n:\n', K_t_n)
    print('K_t_n_s: \n', K_t_n_s)
    tmp = np.matmul(np.transpose(K_t_n_s), np.linalg.inv(K_t_n))
    print('tmp: \n', tmp)
    Omega = np.matmul(tmp, np.ones(T).reshape(-1, 1))
    Omega = 1 - Omega
    print('Omega: \n', Omega)

    mu_n = mu[idx]
    C_n_n = C[np.ix_([idx], [idx])]
    y_n = y_train_[idx]

    tmp = np.matmul(np.transpose(K_t_n_s), np.linalg.inv(K_t_n))
    mean = np.matmul(tmp, y_n) + np.matmul(Omega, mu_n)
    print('tmp: \n', tmp)
    print('mean: \n', mean)

    K_t_n_s_s = kernel_tc_(np.array([T+1]), np.array([T+1]))

    tmp = np.matmul(np.transpose(K_t_n_s), np.linalg.inv(K_t_n))
    cov = K_t_n_s_s - np.matmul(tmp, K_t_n_s) + \
        np.matmul(np.matmul(Omega, C_n_n), np.transpose(Omega))
    print('cov: \n', cov)

    # std
    std = np.sqrt(cov)
    print('std:\n', std)

    mean_pred, std_pred = predictor.predict_point_old([X_train_[1]])
    print('mean_pred:\n', mean_pred)
    print('std_pred:\n', std_pred)

    assert np.allclose(mean, mean_pred, rtol=1e-6)
    assert np.allclose(std[0], std_pred, rtol=1e-6)
    print('--------------test point_old pass !----------------------')

    mean, cov = predictor.predict_asymptote_new(x, return_std=False)
    K_t = kernel_tc_([1])
    cov += K_t

    print('mean')
    print(mean)
    print('cov')
    print(cov)

    # mean, std
    mean = np.array([mean[i][0] for i in range(len(mean))])
    std = np.sqrt(np.diag(cov))
    print('mean:\n', mean)
    print('std:\n', std)

    mean_pred, std_pred = predictor.predict_point_new(x, return_std=True)

    assert np.array_equal(mean, mean_pred)
    assert np.array_equal(std, std_pred)

    print('--------------test point_new pass !----------------------')


def log_likelihood_test():
    X, y = create_fake_data_simple()

    predictor = Predictor()
    predictor.fit(X, y)

    print('log_likelihood final of theta:')
    print(predictor.theta)
    print(predictor.bounds)
    print(predictor.log_marginal_likelihood_value_)

    print('log_likelihood of random theta')
    theta = np.array([0, -0.69314718, -0.69314718, -9.21034037])
    print(predictor.log_marginal_likelihood(theta))


def inv_test():
    X = np.arange(4).reshape(2, 2)
    print(np.linalg.inv(X))
    print(np.linalg.inv(X))


# kernel_ktc_test()
predict_test()
# log_likelihood_test()
# inv_test()
