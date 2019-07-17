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

import logging
import numpy as np
import nni.parameter_expressions as parameter_expressions

from nni.mtsmac_advisor.util import ei

logger = logging.getLogger("MTSMAC_Advisor_AutoML")

# pylint:disable=invalid-name


class TargetSpace():
    """
    Holds the param-space coordinates (X) and target values (Y)
    """

    def __init__(self, search_space, max_epochs=20, random_state=None):
        """
        Parameters
        ----------
        search_space : dict
                example: search_space = {
                        "dropout_rate":{"_type":"uniform","_value":[0.5,0.9]},q
                        "conv_size":{"_type":"choice","_value":[2,3,5,7]}
                        }

        random_state : int, RandomState, or None
            optionally specify a seed for a random number generator
        """
        self.random_state = random_state

        # Get the name of the parameters
        self._keys = sorted(search_space)
        # Create an array with parameters types/bounds
        self._bounds = np.array([item[1] for item in sorted(
            search_space.items(), key=lambda x: x[0])])

        # used for saving all info about params
        # [{id: 0, params:[], perf: []}, {...}]
        self.hyper_configs = []

        # used for fitting predictor
        self._params = np.empty(shape=(0, self.dim))
        self._target = np.empty(shape=(0), dtype=object)

        self.next_param_id = 0
        self.max_epochs = max_epochs

        self._len_completed = 0
        self._y_max = 0

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

    @property
    def len(self):
        '''length of generated parameters'''
        return len(self.hyper_configs)

    @property
    def len_completed(self):
        '''length of completed trials'''
        return self._len_completed

    def get_train_data(self):
        '''
        params, target: numpy array
        '''
        params = np.empty(shape=(0, self.dim))
        target = np.empty(shape=(0), dtype=object)
        for item in self.hyper_configs:
            if len(item['perf']) >= 0:
                params = np.vstack((params, item['params']))
                target = np.append(target, ['new_serial'])
                target[-1] = item['perf']  # TODO: more pythonic
        logger.info("params:%s", params)
        logger.info("target:%s", target)
        return params, target

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
        self.hyper_configs.append({
            'parameter_id': parameter_id,
            'params': params,  # array
            'perf': [],
            'status': 'RUNNING'
        })

    def register(self, parameter_id, value):
        '''
        insert a result into target space
        '''
        self.hyper_configs[parameter_id]['perf'].append(value)

    def trial_end(self, parameter_id):
        '''
        trial end
        '''
        logger.info("Trial end, parameter_id: %s", parameter_id)
        if len(self.hyper_configs[parameter_id]['perf']) >= self.max_epochs:
            self.hyper_configs[parameter_id]['status'] = 'FINISH'
            # update internal flag variables
            self._len_completed += 1
            if self.hyper_configs[parameter_id]['perf'][-1] > self._y_max:
                self._y_max = self.hyper_configs[parameter_id]['perf'][-1]

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

    def select_config(self, predictor):
        '''
        function to find the maximum of the acquisition function.
        Step 1: get a basket by 'Expected Improvement'
        Step 2; get a config by 'Information Gain'
        '''

        params_new = self._select_from_new(predictor)
        parameter_id = self.next_param_id
        self.next_param_id += 1
        self.register_new_config(parameter_id, params_new)
        logger.info("New config proposed")

        self._select_from_old(predictor)

        parameter_json = self.array_to_params(params_new)
        logger.info("Generate paramageter :\n %s", parameter_json)

        return parameter_id, parameter_json

    def _select_from_new(self, predictor, num_warmup=50):
        # select from new configs
        # Warm up with random points
        x_tries = [self.random_sample()
                   for _ in range(int(num_warmup))]
        mean_tries, std_tries = predictor.predict(x_tries, final_only=True)
        ys = ei(mean_tries, std_tries, y_max=self._y_max)
        params = x_tries[ys.argmax()]
        max_acq = ys.max()

        return params

    def _select_from_old(self, predictor):
        # select from running configs
        # step 1: get vertor of running configs
        params_running = np.empty(shape=(0, self.dim))
        for item in self.hyper_configs:
            if item['status'] == 'RUNNING':
                params_running = np.vstack((params_running, item['params']))
        if params_running.shape[0] > 0:
            # step 2: predict
            mean_tries, std_tries = predictor.predict(
                params_running, final_only=True)
            ys = ei(mean_tries, std_tries, y_max=self._y_max)
            params = params_running[ys.argmax()]
            max_acq = ys.max()
            # find the original parameter_id TODO:more pythonic
            parameter_id = 0
            for item in self.hyper_configs:
                if np.array_equal(item['params'], params):
                    parameter_id = item['parameter_id']
                    break
            logger.info("Old configs proposed, parameter_id: %s", parameter_id)

        return parameter_id, params

    def select_config_warmup(self):
        '''
        select configs for random warmup
        '''
        params = self.random_sample()
        parameter_id = self.next_param_id
        self.next_param_id += 1

        self.register_new_config(parameter_id, params)

        parameter_json = self.array_to_params(params)
        logger.info("Generate paramageter for warm up :\n %s", parameter_json)

        return parameter_id, parameter_json
