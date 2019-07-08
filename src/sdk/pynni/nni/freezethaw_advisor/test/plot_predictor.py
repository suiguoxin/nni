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
plot_predictor.py
"""
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import gridspec

from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor

from nni.freezethaw_advisor.predictor import Predictor
from nni.freezethaw_advisor.test.util import PATH, COLORS
from nni.freezethaw_advisor.test.util import create_fake_data_expdacay, create_fake_data_expdacay_diff_length


def plot_asymptote():
    '''
    Fig 2(b)
    '''
    X, y = create_fake_data_expdacay_diff_length()

    predictor = Predictor(optimizer=None)

    predictor.fit(X, y)
    mean, std = predictor.predict_asymptote_old(return_std=True)

    for i in range(len(y)):
        length = len(y[i])

        plt.plot(np.arange(length), y[i], color=COLORS[i], )
        mu = mean[i][0]
        sigma = std[i]
        plt.plot(np.arange(length), [mu] * length,
                 '--', color=COLORS[i], label='Prediction')
        plt.fill_between(np.arange(length), [
            mu-1.9600 * sigma * 0.1]*length, [mu + 1.9600 * sigma * 0.1]*length, color=COLORS[i], alpha=0.5, interpolate=True)
        # TODO: remove * 0.1

    plt.title('Training Curve Predictions')
    plt.legend()
    plt.savefig('{}/image/2_b.png'.format(PATH))


plot_asymptote()
