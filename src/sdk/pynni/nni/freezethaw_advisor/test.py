import warnings

import numpy as np

from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor

import matplotlib.pyplot as plt

from nni.freezethaw_advisor.kernels import KTC


'''
t1 = np.array([[1], [2], [3])
t2 = np.array([[2], [3], [4])


ufunc = np.frompyfunc(lambda x, y: x+y, 2, 1)

print(ufunc(2,1))

def outer(X, Y):
    r = np.empty((X.shape[0], Y.shape[0]))
    print((X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            r[i][j] = x[0] +y
    return r


print(outer(t1, t2))

'''

# pylint:disable=missing-docstring
# pylint:disable=no-member


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
    #matern = Matern(nu=2.5)
    ktc = KTC(alpha=1, beta=0.5)
    gp = GaussianProcessRegressor(
        kernel=ktc
    )

    for _ in range(10):
        X = np.arange(1, 100).reshape(-1, 1)
        #K = ktc(X)
        #y = np.random.multivariate_normal([0]*99, K)
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
        #X_obs = [[1], [2]]
        #y_obs = [0.9, 0.7]
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

    for i in range(1):
        X = np.empty(shape=(0, 1))
        y = np.empty(shape=(0))

        if warm_up:
            warmup_X = [[1], [2], [3], [4], [5]]
            warmup_y = [1-0.098, 1-0.716, 1-0.830, 1-0.848, 1-0.878]
            X = np.append(X, warmup_X, axis=0)
            y = np.append(y, warmup_y, axis=0)
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


ktc_1()
ktc_2()
ktc_3()
ktc_4()
