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
    # TODO: get negative variances if delete WhiteKernel
    ktc = KTC(alpha=0.42, beta=0.47) + WhiteKernel(1e-4)
    gp = GaussianProcessRegressor(
        # optimizer=None,
        kernel=ktc
    )

    y_true = [1.0079430977511248, 0.76330811643219, 0.6032876372496866, 0.5168444483741073,
              0.508861394142888, 0.4497688131732428, 0.425768623752674, 0.41931871156546363,
              0.3937311341078193, 0.395356073476545, 0.404160268384624, 0.4155731870866553,
              0.4041162868757059, 0.40535759334652105, 0.385601675552465, 0.41066804715942196,
              0.402773438132659, 0.40282431477974856, 0.39815940349966084, 0.3965369598572702,
              0.3939059994292343, 0.40036844994894766, 0.40237072170638555, 0.3992678671362422,
              0.4001442467909903, 0.40163208405314343, 0.39720951332291565, 0.4075465165026093,
              0.400577844627453, 0.40388590130410923, 0.39923794188541434, 0.40341386476802144,
              0.3986970571777314, 0.40092351782685054, 0.40154781507529796, 0.4007543310111782,
              0.39861314181038077, 0.3977118733915813, 0.4005187692633148, 0.4036269361641092,
              0.4015537217666074, 0.39978830997620074, 0.3999756165922316, 0.3998083424295331,
              0.39969519330378667, 0.3969197859555333, 0.39982714973931366, 0.39787110070109244,
              0.400697870379133, 0.399801453513319]

    N = len(y_true)

    for mean_prior in np.arange(0.3, 0.35, 0.1):
        # warm up
        N_obs = 30
        X_obs = np.arange(N_obs).reshape(-1, 1)
        y_obs = y_true[:N_obs]
        gp.fit(X_obs, y_obs - mean_prior)

        print('ktc after fit:')
        print(gp.kernel_)
        print('gp.kernel_.theta')
        print(gp.kernel_.theta)
        print('gp.kernel_.bounds')
        print(gp.kernel_.bounds)
        print('gp.log_marginal_likelihood(gp.kernel_.theta)')
        print(gp.log_marginal_likelihood(gp.kernel_.theta))

        # predict
        X_s = np.arange(N).reshape(-1, 1)
        mean, std = gp.predict(X_s, return_std=True)
        mean += mean_prior
        #y_s = np.random.multivariate_normal(mean+mean_prior, cov)

        plt.plot(range(0, N), mean,
                 label='y_predict with mean prior:{0:.2g}'.format(mean_prior))

        T = np.arange(N).reshape(-1, 1)
        plt.fill(np.concatenate([T, T[::-1]]), np.concatenate([mean - 1.9600 * std, (mean + 1.9600 * std)[::-1]]),
                 alpha=.6, fc='c', ec='None', label='95% confidence interval')

    plt.plot(range(0, N), y_true, label='y_true')
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
ktc_3()
# ktc_4()
# ktc_5()
