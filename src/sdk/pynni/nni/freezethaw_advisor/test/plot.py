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
from sklearn.gaussian_process import GaussianProcessRegressor

import matplotlib.pyplot as plt

from nni.freezethaw_advisor.kernels import KTC
from nni.freezethaw_advisor.predictor import Predictor


# pylint:disable=missing-docstring
# pylint:disable=no-member
# pylint:disable=invalid-name


PATH = './src/sdk/pynni/nni/freezethaw_advisor/test'


def ktc_1():
    x = np.arange(100)
    for lam in np.arange(0, 1, 0.05):
        y = np.exp(-x*lam)
        plt.plot(x, y)

    plt.title('Exponential Decay Basis')
    plt.savefig('{}/ktc_1.png'.format(PATH))
    plt.close()


def ktc_2():
    # matern = Matern(nu=2.5)
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
    plt.savefig('{}/ktc_2.png'.format(PATH))
    plt.close()


def ktc_3():
    ktc = KTC(alpha=1, beta=0.5) + WhiteKernel(noise_level=1e-4)
    gp = GaussianProcessRegressor(
        kernel=ktc
    )

    for mean_prior in np.arange(0, 0.3, 0.05):
        # warm up
        # X_obs = [[1], [2]]
        # y_obs = [0.9, 0.7]
        X_obs = [[0]]
        y_obs = [1]
        gp.fit(X_obs, y_obs)

        N = 100

        X_s = np.arange(1, N).reshape(-1, 1)
        mean, cov = gp.predict(X_s, return_cov=True)
        y_s = np.random.multivariate_normal(mean+mean_prior, cov)
        y = np.append(y_obs, y_s, axis=0)

        plt.plot(range(0, N), y, label='mean prior:{0:.2g}'.format(mean_prior))
    plt.title('Training Curve Samples')
    plt.legend()
    plt.savefig('{}/ktc_3.png'.format(PATH))
    plt.close()


def ktc_4():
    ktc = KTC(alpha=1, beta=0.5) + WhiteKernel(noise_level=1e-4)
    gp = GaussianProcessRegressor(
        optimizer=None,
        kernel=ktc
    )

    warm_up = True

    for _ in range(1):
        #X = np.empty(shape=(0, 1))
        #y = np.empty(shape=(0))

        if warm_up:
            X = np.array([[1], [2], [3], [4], [5]])
            y = np.array([0.8752999976277351, 0.21170002222061157,
                          0.1291000247001648, 0.10839998722076416, 0.09160000085830688])
            # warmup_y = [0.8752999976277351, 0.21170002222061157, 0.1291000247001648, 0.10839998722076416, 0.09160000085830688, 0.0812000036239624, 0.07380002737045288, 0.06919997930526733, 0.06590002775192261, 0.06290000677108765,
            #             0.05909997224807739, 0.055899977684020996, 0.051500022411346436, 0.04839998483657837, 0.04989999532699585, 0.044700026512145996, 0.04250001907348633, 0.04180002212524414, 0.04280000925064087, 0.04030001163482666, 0.03799998760223389]

            gp.fit(X, y)

        for x in range(6, 100):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mean_prior = 0.1
                mean, std = gp.predict(np.array([[x]]), return_std=True)
                y_sample = np.random.normal(
                    mean+mean_prior, std)  # pylint:disable=no-member
                X = np.append(X, [[x]], axis=0)
                y = np.append(y, y_sample, axis=0)
                gp.fit(X, y)

        plt.plot(range(1, 100), y)
    plt.title('Training Curve Samples')
    plt.savefig('{}/ktc_4.png'.format(PATH))
    plt.close()


def ktc_5():
    ktc = KTC(alpha=1, beta=0.5) + WhiteKernel(noise_level=1e-4)
    gp = GaussianProcessRegressor(
        kernel=ktc,
        optimizer=None
    )

    for mean_prior in np.arange(0, 0.1, 0.02):
        # warm up
        X_obs = [[1], [2], [3], [4], [5]]
        y_obs = [0.8752999976277351, 0.21170002222061157,
                 0.1291000247001648, 0.10839998722076416, 0.09160000085830688]
        # X_obs = [[0]]
        # y_obs = [1]
        gp.fit(X_obs, y_obs)

        N = 100

        X_s = np.arange(6, N).reshape(-1, 1)
        mean, cov = gp.predict(X_s, return_cov=True)
        y_s = np.random.multivariate_normal(mean+mean_prior, cov)
        y = np.append(y_obs, y_s, axis=0)

        plt.plot(range(1, N), y, label='mean prior:{0:.2g}'.format(mean_prior))
    plt.title('Training Curve Samples')
    plt.legend()
    plt.savefig('{}/ktc_5.png'.format(PATH))
    plt.close()


# ktc_1()
# ktc_2()
# ktc_3()
# ktc_4()
ktc_5()
