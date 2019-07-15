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

    def __init__(self, search_space, max_epoch=20, random_state=None):
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
        self._bounds = np.array([item[1] for item in sorted(
            search_space.items(), key=lambda x: x[0])])

        # preallocated memory for X and Y points
        self._params = np.empty(shape=(0, self.dim))
        self._target = np.empty(shape=(0), dtype=object)

        self.hyper_configs_running = {}  # {id: {params:, perf: [], length: N}}
        self.hyper_configs_completed = {}  # {id: {params:, perf: [], length: N}}

        self.next_param_id = 0
        self.max_epoch = max_epoch

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

    def params_to_array(self, params):
        ''' dict to array '''
        try:
            assert set(params) == set(self.keys)
        except AssertionError:
            raise ValueError(
                "Parameters' keys ({}) do ".format(sorted(params)) +
                "not match the expected set of keys ({}).".format(self.keys)
            )
        return np.asarray([params[key] for key in self.keys])

    def array_to_params(self, x):
        '''
        array to dict

        maintain int type if the paramters is defined as int in search_space.json
        '''
        try:
            assert len(x) == len(self.keys)
        except AssertionError:
            raise ValueError(
                "Size of array ({}) is different than the ".format(len(x)) +
                "expected number of parameters ({}).".format(self.dim())
            )

        params = {}
        for i, _bound in enumerate(self._bounds):
            if _bound['_type'] == 'choice' and all(isinstance(val, int) for val in _bound['_value']):
                params.update({self.keys[i]: int(x[i])})
            elif _bound['_type'] in ['randint']:
                params.update({self.keys[i]: int(x[i])})
            else:
                params.update({self.keys[i]:  x[i]})

        return params

    def register_new_config(self, parameter_id, params):
        '''
        register new config without performance
        '''
        self.hyper_configs_running[parameter_id] = {
            'params': self.array_to_params(params),
            'perf': []
        }

        # param = [val for _, val in params.items()]
        self._params = np.vstack((self._params, params))
        self._target = np.append(self._target, ['NO_VALUE']) # TODO: refine

    def register(self, parameter_id, value):
        '''
        insert a result into target space
        '''
        self.hyper_configs_running[parameter_id]['perf'].append(value)
        if self._target[parameter_id] == 'NO_VALUE':
            self._target[parameter_id] = [value]
        else:
            self._target[parameter_id].append(value)

    def trial_end(self, parameter_id):
        '''
        trial end
        '''
        if len(self.hyper_configs_running[parameter_id]['perf']) >= self.max_epoch:
            params = self.hyper_configs_running.pop(
                parameter_id)  # TODO: check
            self.hyper_configs_completed[parameter_id] = params.hyper_params

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

    def select_config(self):
        '''
        function to find the maximum of the acquisition function.
        Step 1: get a basket by 'Expected Improvement'
        Step 2; get a config by 'Information Gain'
        '''
        # TODO: select from running configs
        # select from new configs
        params = self.random_sample()
        parameter_id = self.next_param_id
        self.next_param_id += 1

        self.register_new_config(parameter_id, params)

        parameter_json = self.array_to_params(params)

        return parameter_id, parameter_json
