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

import numpy as np
import nni.parameter_expressions as parameter_expressions


class TargetSpace():
    """
    Holds the param-space coordinates (X) and target values (Y)
    """

    def __init__(self, search_space, random_state=None):
        """
        Parameters
        ----------
        search_space : dict
                example: search_space = {
                        "dropout_rate":{"_type":"uniform","_value":[0.5,0.9]},
                        "conv_size":{"_type":"choice","_value":[2,3,5,7]}
                        }

        random_state : int, RandomState, or None
            optionally specify a seed for a random number generator
        """
        self.random_state = random_state

        # Get the name of the parameters
        self._keys = sorted(search_space)
        # Create an array with parameters bounds
        self._bounds = np.array(
            [item[1]
                for item in sorted(search_space.items(), key=lambda x: x[0])]
        )

        self._params = None  # X
        self._target = None  # y

        self.hyper_configs_running = {}  # {id: {params:, perf: [], length: N}}
        self.hyper_configs_completed = {}  # {id: {params:, perf: [], length: N}}

    @property
    def params(self):
        '''
        params: numpy array
        '''
        return self._params

    @property
    def target(self):
        '''
        target: numpy array
        '''
        return self._target

    @property
    def dim(self):
        '''
        dim: int
            length of keys
        '''
        return len(self._keys)

    @property
    def keys(self):
        '''
        keys: numpy array
        '''
        return self._keys

    @property
    def bounds(self):
        '''bounds'''
        return self._bounds

    def register(self, parameter_id, seq, value):
        '''
        insert a result into target space
        '''
        # TODO: if parameter_id doesn't exist
        self.hyper_configs_running[parameter_id]['perf'].append(value)

        # update _params, _target
        X = np.empty(0, self.dim)
        y = np.empty(0, dtype=object)

        # for _ in hyper_configs_running:
        #    np.vstack
        self._params = X
        self._target = y

    def trial_end(self, parameter_id):
        '''
        trial end
        '''
        params = self.hyper_configs_running.pop(parameter_id)
        self.hyper_configs_running[parameter_id] = params.hyper_params

    def random_sample(self):
        """
        Creates a random point within the bounds of the space.
        """
        params = np.empty(self.dim)
        for col, _bound in enumerate(self._bounds):
            if _bound['_type'] == 'choice':
                params[col] = parameter_expressions.choice(
                    _bound['_value'], self.random_state)
            elif _bound['_type'] == 'randint':
                params[col] = self.random_state.randint(
                    _bound['_value'][0], _bound['_value'][1], size=1)
            elif _bound['_type'] == 'uniform':
                params[col] = parameter_expressions.uniform(
                    _bound['_value'][0], _bound['_value'][1], self.random_state)
            elif _bound['_type'] == 'quniform':
                params[col] = parameter_expressions.quniform(
                    _bound['_value'][0], _bound['_value'][1], _bound['_value'][2], self.random_state)
            elif _bound['_type'] == 'loguniform':
                params[col] = parameter_expressions.loguniform(
                    _bound['_value'][0], _bound['_value'][1], self.random_state)
            elif _bound['_type'] == 'qloguniform':
                params[col] = parameter_expressions.qloguniform(
                    _bound['_value'][0], _bound['_value'][1], _bound['_value'][2], self.random_state)

        return params
