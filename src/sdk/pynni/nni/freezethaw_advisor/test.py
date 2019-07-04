from scipy.linalg import cholesky, cho_solve, solve_triangular, block_diag
import warnings

import numpy as np

from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from itertools import chain

import matplotlib.pyplot as plt

from nni.freezethaw_advisor.kernels import KTC
from nni.freezethaw_advisor.predictor import Predictor


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


# ktc_1()
# ktc_2()
# ktc_3()
# ktc_4()

def kernel_ktc_test():
    # test of K(X, X), k(x,x) = 0
    ktc = KTC(alpha=1, beta=0)
    X = np.array([[1], [2]])
    K = np.array([[0, 0],
                  [0, 0]])

    K_cal = ktc(X)

    assert np.array_equal(K, K_cal)
    print('test 1 pass !')

    # test of K(X, X), k(x,x) != 0
    ktc = KTC(alpha=1, beta=1)
    X = [[1], [2]]
    K = np.array([[1/3, 1/4],
                  [1/4, 1/5]])

    K_cal = ktc(X)

    assert np.array_equal(K, K_cal)
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
    X = np.array([[1],
                  [2]])
    y = np.array([[1, 2],
                  [1, 2, 3]])
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

    # O (NT, N)
    O = block_diag(*[np.ones((len(y_train_[i]), 1))
                     for i in range(y_train_.shape[0])])
    print('O:\n', O)

    # K_x, K_t
    kernel_as_ = Matern(nu=2.5)
    kernel_tc_ = KTC(alpha=1, beta=0.5)
    K_x = kernel_as_(X_train_)
    K_t = block_diag(*[kernel_tc_(np.reshape(y_train_[i], (-1, 1)))
                       for i in range(y_train_.shape[0])])
    print('K_x:\n', K_x)
    print('K_t:\n', K_t)

    # m
    m = np.zeros((y_train_.shape[0], 1))
    print('m:')
    print(m)

    # gamma
    tmp = np.matmul(np.transpose(O), np.linalg.pinv(K_t))  # shape (N,NT)
    gamma = np.matmul(tmp, y_train_flatten_ - np.matmul(O, m))
    print('gamma:')
    print(gamma)

    # Lambda
    tmp = np.matmul(np.transpose(O), np.linalg.pinv(K_t))
    Lambda = np.matmul(tmp, O)
    print('Lambda:')
    print(Lambda)

    # C
    tmp = np.matmul(K_x, np.linalg.pinv(K_x + np.linalg.pinv(Lambda)))
    C = K_x - np.matmul(tmp, K_x)
    print('C:')
    print(C)

    # mu
    mu = np.matmul(C, gamma)
    mu += m
    print('mu:')
    print(mu)

    predictor = Predictor()
    predictor.fit(X, y)
    mu_pred, C_pred = predictor.predict_asymptote_old()

    assert np.array_equal(mu, mu_pred)
    assert np.array_equal(C, C_pred)

    print('--------------test asymptote_new pass !----------------------')

    x = np.array([[3],
                  [4]])

    K_x_s = kernel_as_(X_train_, x)
    tmp = np.matmul(np.transpose(K_x_s), np.linalg.inv(K_x))
    mean = np.matmul(tmp, mu - m)
    print('mean:')
    print(mean)

    tmp = np.matmul(np.transpose(K_x_s), np.linalg.inv(
        K_x + np.linalg.inv(Lambda)))
    var = np.matmul(tmp, K_x_s)
    print('var:')
    print(var)

    mean_pred, var_pred = predictor.predict_asymptote_new(x)

    assert np.array_equal(mean, mean_pred)
    assert np.array_equal(var, var_pred)

    print('--------------test asymptote_old pass !----------------------')

    idx = 1

    y_t = np.reshape(y_train_[idx], (-1, 1))
    T = y_t.shape[0]
    T_arr = np.arange(1, T+1).reshape(-1, 1)
    print('T_arr:', T_arr)
    K_t_n = kernel_tc_(T_arr)
    K_t_n_s = kernel_tc_(T_arr, T+1)
    print('K_t_n:')
    print(K_t_n)
    print('K_t_n_s:')
    print(K_t_n_s)
    tmp = np.matmul(np.transpose(K_t_n_s), np.linalg.inv(K_t_n))
    print('tmp:')
    print(tmp)
    Omega = np.matmul(tmp, np.ones(T).reshape(-1, 1))
    Omega = 1 - Omega
    print('Omega:')
    print(Omega)

    mu_n = mu[idx]
    C_n_n = C[np.ix_([idx], [idx])]
    print('C_n_n')
    print(C_n_n)
    y_n = y_train_[idx]

    tmp = np.matmul(np.transpose(K_t_n_s), np.linalg.inv(K_t_n))
    mean = np.matmul(tmp, y_n) + np.matmul(Omega, mu_n)
    print('mean')
    print(mean)

    K_t_n_s_s = kernel_tc_(T+1, T+1)

    tmp = np.matmul(np.transpose(K_t_n_s), np.linalg.inv(K_t_n))
    var = K_t_n_s_s - np.matmul(tmp, K_t_n_s) + \
        np.matmul(np.matmul(Omega, C_n_n), np.transpose(Omega))
    print('var')
    print(var)

    mean_pred, var_pred = predictor.predict_point_old(1)

    assert np.array_equal(mean, mean_pred)
    assert np.array_equal(var, var_pred)
    print('--------------test point_old pass !----------------------')


# kernel_ktc_test()
predict_test()
