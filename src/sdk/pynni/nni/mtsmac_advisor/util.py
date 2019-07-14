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
util.py
"""

import warnings
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize


def acq_max():
    '''
    A function to find the maximum of the acquisition function.
    Step 1: get a basket by 'Expected Improvement'
    Step 2; get a config by 'Information Gain'
    '''

    # get a basket
    _get_basket(predictor, num_old, num_new)


def _get_basket(predictor, num_old, num_new):
    '''
    get a basket of num_old + num_new canditate by EI
    '''
    old_points = []
    for X in old_points:
        mean, std = predictor.predict_asymptote_old(X)


def _ei(x, mean, std, y_max, xi=0):
    '''
    calculate ei of one configurationi
    '''
    z = (mean - y_max - xi)/std
    return (mean - y_max - xi) * norm.cdf(z) + std * norm.pdf(z)


def _ei_ed(x, mean, std, y_max, xi=0):
    pass


def _ig(X, gp):
    '''
    calculate information gain of several configutions
    '''
    # calculate P of X
    # ...
    # TODO: slice sampling
