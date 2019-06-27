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
target_space.py
"""

from sklearn.gaussian_process.kernels import Matern

from .kernels import KTC

class TargetSpace():
    """
    Holds the param-space coordinates (X) and target values (Y)
    """
    def __init__(self, search_space, random_state=None):
        """
        Parameters
        ----------
        pbounds : dict
            Dictionary with parameters names as keys and a tuple with minimum
            and maximum values.

        random_state : int, RandomState, or None
            optionally specify a seed for a random number generator
        """
        self.X = None
        self.y = None
        self.random_state = random_state
        self.matern = Matern(nu=2.5)
        self.ktc = KTC(0, 0)

    def register(self, x, y):
        '''
        receive new result
        '''
    
    def predict_asymptote(self, x):
        '''
        posterior distribution of a new hyperparameter setting : Equation 14(19)
        '''
        mean = 0
        std = 0
        return mean, std

    def predict_point_old(self, x):
        '''
        posterior distribution for a new point in a training curve: Equation 15(20)
        '''

    def predict_point_new(self, x):
        '''
        posterior distribution for a new point in the absence of any observations : Equation 16(21)
        '''

        