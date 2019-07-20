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
plot_kernels.py
"""

import numpy as np
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor

import matplotlib.pyplot as plt

from nni.ftbo_advisor.kernels import KTC
from nni.ftbo_advisor.predictor import Predictor

from nni.ftbo_advisor.test.util import PATH, COLORS
from nni.ftbo_advisor.test.util import create_fake_data_expdacay


# pylint:disable=missing-docstring
# pylint:disable=no-member
# pylint:disable=invalid-name


def ktc_1():
    '''
    Fig 1(a)
    '''
    x = np.arange(100)
    for lam in np.arange(0, 1, 0.05):
        y = np.exp(-x*lam)
        plt.plot(x, y)

    plt.title('Exponential Decay Basis')
    plt.savefig('{}/image/1_a.png'.format(PATH))
    plt.close()


def ktc_2():
    '''
    Fig 1(b)
    '''
    ktc = KTC(alpha=1, beta=0.5)
    gp = GaussianProcessRegressor(
        kernel=ktc
    )

    for _ in range(10):
        X = np.arange(1, 100).reshape(-1, 1)
        # K = ktc(X)
        # y = np.random.multivariate_normal([0]*99, K)
        mean, cov = gp.predict(X, return_cov=True)
        y = np.random.multivariate_normal(mean, cov)

        plt.plot(range(1, 100), y)

    plt.title('Samples')
    plt.savefig('{}/image/1_b.png'.format(PATH))
    plt.close()


def ktc_3():
    '''
    Fig 1(c)
    '''
    ktc = KTC(alpha=0.42, beta=0.47) + WhiteKernel(1e-4)
    gp = GaussianProcessRegressor(
        optimizer=None,
        kernel=ktc
    )

    _, ys = create_fake_data_expdacay()

    for _, y_true in enumerate(ys):
        y_true = np.array(y_true)
        N = len(y_true)
        mean_prior = 0
        # warm up
        N_obs = 10
        X_obs = np.arange(N_obs).reshape(-1, 1)
        y_obs = y_true[:N_obs]
        gp.fit(X_obs, y_obs - mean_prior)

        # predict
        X_s = np.arange(N).reshape(-1, 1)
        mean, cov = gp.predict(X_s, return_cov=True)
        mean += mean_prior

        # plt.plot(range(0, N), y_true,
        #         label='y_true with mean prior:{0:.2g}'.format(mean_prior))

        y = np.random.multivariate_normal(mean, cov)

        plt.plot(range(0, N), y, label='y samples')
    plt.title('Training Curve Samples')
    plt.legend()
    plt.savefig('{}/image/1_c.png'.format(PATH))
    plt.close()


def ktc_3_1():
    '''
    variation of Fig 1(3): show 95% confidence interval
    '''
    # TODO: get negative variances if delete WhiteKernel
    ktc = KTC(alpha=0.42, beta=0.47) + WhiteKernel(1e-4)
    gp = GaussianProcessRegressor(
        optimizer=None,
        kernel=ktc
    )

    _, ys = create_fake_data_expdacay()

    for i, y_true in enumerate(ys):
        y_true = np.array(y_true)
        N = len(y_true)
        mean_prior = 0
        # warm up
        N_obs = 20
        X_obs = np.arange(N_obs).reshape(-1, 1)
        y_obs = y_true[:N_obs]
        gp.fit(X_obs, y_obs - mean_prior)

        # predict
        X_s = np.arange(N).reshape(-1, 1)
        mean, std = gp.predict(X_s, return_std=True)
        mean += mean_prior

        plt.plot(range(0, N), y_true, color=COLORS[i], label='y_true')
        plt.plot(range(0, N), mean, color=COLORS[i],
                 label='y_predict')

        T = np.arange(N).reshape(-1, 1)
        plt.fill(np.concatenate([T, T[::-1]]), np.concatenate([mean - 1.9600 * std, (mean + 1.9600 * std)[::-1]]),
                 color=COLORS[i], alpha=.6)

    plt.title('Training Curve Samples with 95% confidence interval')
    plt.legend()
    plt.savefig('{}/image/1_c_1.png'.format(PATH))
    plt.close()


def ktc_3_2():
    '''
    variation of Fig 1(3): show influence of different mean_prior
    '''
    ktc = KTC(alpha=0.42, beta=0.47) + WhiteKernel(1e-4)
    gp = GaussianProcessRegressor(
        optimizer=None,
        kernel=ktc
    )

    _, ys = create_fake_data_expdacay()
    y_true = ys[0]

    i = 0
    for mean_prior in np.arange(0.1, 0.3, 0.1):
        i += 1
        N = len(y_true)
        # warm up
        N_obs = 20
        X_obs = np.arange(N_obs).reshape(-1, 1)
        y_obs = y_true[:N_obs]
        gp.fit(X_obs, y_obs - mean_prior)

        # predict
        X_s = np.arange(N).reshape(-1, 1)
        mean, std = gp.predict(X_s, return_std=True)
        mean += mean_prior

        plt.plot(range(N), y_true, color=COLORS[i], label='y_true')
        plt.plot(range(N), mean, color=COLORS[i],
                 label='y_predict with prior mean: {}'.format(mean_prior))

        T = np.arange(N).reshape(-1, 1)
        plt.fill(np.concatenate([T, T[::-1]]), np.concatenate([mean - 1.9600 * std, (mean + 1.9600 * std)[::-1]]),
                 color=COLORS[i], alpha=.6, fc='c', ec='None', label='95% confidence interval')

    plt.title('Training Curve Samples with different prior mean')
    plt.legend()
    plt.savefig('{}/image/1_c_2.png'.format(PATH))
    plt.close()



ktc_1()
ktc_2()
ktc_3()
ktc_3_1()
ktc_3_2()


""" 
print('ktc after fit:')
print(gp.kernel_)
print('gp.kernel_.theta')
print(gp.kernel_.theta)
print('gp.kernel_.bounds')
print(gp.kernel_.bounds)
print('gp.log_marginal_likelihood(gp.kernel_.theta)')
print(gp.log_marginal_likelihood(gp.kernel_.theta)) 
"""
