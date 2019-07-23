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

from nni.ftbo_advisor.util import ei
from nni.ftbo_advisor.predictor import Predictor

logger = logging.getLogger("FREEZE_THAW_Advisor_AutoML")

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
        i.e. : search_space = {
                "dropout_rate":{"_type":"uniform","_value":[0.5,0.9]},
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
        logger.debug("get_train_data:")
        logger.debug("params:%s", params)
        logger.debug("target:%s", target)
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
            # TODO: consider inter result
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

    def select_config(self, predictor):
        '''
        Algorithm 1 of Freeze-Thaw
        Step 1: get a basket by 'Expected Improvement'
        Step 2: get a config by 'Information Gain'
        '''
        # construct the basket of all configs
        num_new = 3
        num_old = 10
        basket_new = self._get_basket_new(predictor, num_new)
        basket_old = self._get_basket_old(predictor, num_old)
        basket = np.append(basket_new, basket_old)

        logger.debug("basket_new %s", basket_new)
        logger.debug("basket_old %s", basket_old)
        logger.debug("basket %s", basket)

        # vector format of params in the basket
        params = np.empty((0, len(basket[0]['param'])))
        for i, item in enumerate(basket):
            params = np.vstack((params, item['param']))
        logger.debug("params %s", params)

        # get P_min and current entropy
        P_min = self._get_P_min_basket(basket)
        H = self._cal_entropy(P_min)
        logger.debug("P_min %s", P_min)
        logger.debug("H %s", H)

        # get information gain
        logger.debug("------------fantasize period---------------")
        a = np.zeros(len(basket))
        n_fant = 5
        X, y = self.get_train_data()
        for i, item in enumerate(basket):
            logger.debug("fantasize element %s in the basket", i)
            for j in range(n_fant):
                logger.debug("fantasize round %s", j)
                mean_new_fant, std_new_fant = [], []
                mean_old_fant, std_old_fant = [], []
                if i < num_new:
                    # fantasize an observation of a old point
                    mean, std = predictor.predict_point_new([item['param']])
                    obs = self.random_state.normal(mean[0], abs(std[0]))
                    # add fantsized point to fake training data
                    X_fant = np.vstack((X, item['param']))
                    y_fant = np.append(y, ['new_serial'])
                    y_fant[-1] = [obs]
                    # conditioned on the observation, re-compute P_min and H
                    # fit a new predictor with fantsized point added in training data
                    predictor_fant = Predictor()
                    predictor_fant.fit(X_fant, y_fant)
                    # re-calculate P_min, H
                    mean_new_fant, std_new_fant = predictor_fant.predict_asymptote_new(
                        params[:num_new])
                else:
                    # fantasize an observation of a old point
                    cur_epoch = len(item['perf'])
                    mean, std = predictor.predict_point_old([item['param']])
                    obs = self.random_state.normal(
                        mean[cur_epoch], abs(std[cur_epoch]))
                    # add fantsized point to fake training data
                    X_fant = X.copy()
                    y_fant = y.copy()
                    for k in range(X_fant.shape[0]):
                        if np.array_equal(item['param'], X_fant[k]):
                            y_fant[k] = y_fant[k].copy()
                            y_fant[k].append(obs)
                            break
                    # conditioned on the observation, re-compute P_min and H
                    # fit a new predictor with fantsized point added in training data
                    predictor_fant = Predictor()
                    predictor_fant.fit(X_fant, y_fant)
                    # re-calculate P_min, H
                    mean_old_fant, std_old_fant = predictor_fant.predict_asymptote_old(
                        params[num_new:])

                mean_fant = np.append(mean_new_fant, mean_old_fant)
                std_fant = np.append(std_new_fant, std_old_fant)

                logger.debug("mean fantasize %s", mean_fant)
                logger.debug("std fantasize %s", std_fant)
                P_min_fant = self._get_P_min(mean_fant, std_fant)
                H_fant = self._cal_entropy(P_min_fant)
                logger.debug("P_min_fant %s", P_min)
                logger.debug("H_fant %s", H)
                # average over n_fant
                a[i] += (H_fant / n_fant)

        param_selected = basket[a.argmax()]
        logger.debug("a %s", a)
        logger.debug("param_selected %s", param_selected)

        param = param_selected['param']
        if 'parameter_id' not in param_selected:
            parameter_id = self.next_param_id
            self.next_param_id += 1
            self.register_new_config(parameter_id, param)
        else:
            parameter_id = param_selected['parameter_id']

        parameter_json = self.array_to_params(param)
        logger.info("Generate paramageter :\n %s", parameter_json)

        return parameter_id, parameter_json

    def _get_basket_new(self, predictor, num, num_warmup=10000):
        '''
        select a basket from new configs

        Returns
        -------
        basket_new: list, i.e. [{'parameter_id','param':[], 'mean':, 'std': , 'ei': }, {...}]
        '''
        # Warm up with random points
        x_tries = [self.random_sample()
                   for _ in range(int(num_warmup))]
        mean, std = predictor.predict_asymptote_new(x_tries)
        ys = ei(mean, std, y_max=self._y_max)

        x_tries = [x for x, _ in sorted(
            zip(x_tries, ys), key=lambda pair: pair[1], reverse=True)]
        sorted(ys, reverse=True)

        # local search with the 10 top random points
        start_points = x_tries[:10]
        acq_val_incumbent = ys[:10]

        logger.debug("start_points: %s", start_points)
        logger.debug("acq_val_incumbent: %s", acq_val_incumbent)
        max_steps = 10
        for i, start_point in enumerate(start_points):
            logger.debug("start_point : %s", start_point)
            incumbent = start_point
            acq_val = acq_val_incumbent[i]
            changed_inc = False
            for _ in range(max_steps):
                all_neighbours = self._get_one_exchange_neighbourhoods(
                    incumbent)
                mean, std = predictor.predict_asymptote_new(all_neighbours)
                ys_neighbours = ei(mean, std, y_max=self._y_max)

                if max(ys) >= acq_val_incumbent[i]:
                    # restart from the better neigobour next step
                    incumbent = all_neighbours[ys_neighbours.argmax()]
                    acq_val = max(ys)
                    changed_inc = True
                else:
                    # stop the local search once none of the neighbours of the start point has larger EI
                    break
            if changed_inc:
                logger.debug("------------neighbour found: %s", incumbent)
                x_tries.append(incumbent)
                ys = np.append(ys, acq_val)

        # re-comput mean std for all the 10010 new points
        mean, std = predictor.predict_asymptote_new(x_tries)

        basket_new = []
        for i, x_i in enumerate(x_tries):
            basket_new.append(
                {'param': x_i, 'mean': mean[i], 'std': std[i], 'ei': ys[i]})
        # sort basket by ei, from big to small
        sorted(basket_new, key=lambda item: item['ei'], reverse=True)

        return basket_new[:num]

    def _get_one_exchange_neighbourhoods(self, param):
        '''get neighbours of a parameter_id
        Parameters
        ----------
        param: list

        i.e. search_space = {
                "dropout_rate":{"_type":"uniform","_value":[0.5,0.9]},
                "conv_size":{"_type":"choice","_value":[2,3,5,7]}
                }
        Returns
        -------
        neighbours: list of params
        '''

        neighbours = np.empty(shape=(0, len(param)))
        for i, _bound in enumerate(self._bounds):
            if _bound['_type'] == 'choice':
                idx = _bound['_value'].index(param[i])
                length = len(_bound['_value'])
                # find index of the neighbours
                idx_neighbours = []
                if 0 <= (idx-1) <= (length-1):
                    idx_neighbours.append(idx-1)
                if 0 <= (idx+1) <= (length-1):
                    idx_neighbours.append(idx+1)
                for idx_neighbour in idx_neighbours:
                    neighbour = param.copy()
                    neighbour[i] = _bound['_value'][idx_neighbour]
                    # logger.debug("neighbour found: %s", neighbour)
                    neighbours = np.vstack((neighbours, neighbour))
            elif _bound['_type'] == 'uniform':
                std = (_bound['_value'][1] - _bound['_value'][0]) * 0.2
                for _ in range(4):
                    while True:
                        ele_fant = self.random_state.normal(param[i], std)
                        if _bound['_value'][0] <= ele_fant <= _bound['_value'][1]:
                            break
                    neighbour = param.copy()
                    neighbour[i] = ele_fant
                    # logger.debug("neighbour found: %s", neighbour)
                    neighbours = np.vstack((neighbours, neighbour))
            else:
                raise ValueError(
                    "only choice, uniform suported for the moment")

        logger.debug(
            "neighbours found for param%s: \n %s", param, neighbours)

        return neighbours

    def _get_basket_old(self, predictor, num):
        '''
        select a basket from running configs

        Returns
        -------
        basket_old: [{'parameter_id': ,'param':[], 'perf': , 'mean':, 'std': , 'ei'}, {...}]
        '''
        basket_old = []
        for item in self.hyper_configs:
            if item['status'] == 'RUNNING':
                mean, std = predictor.predict_asymptote_old(
                    [item['params']])
                ys = ei(mean, std, y_max=self._y_max)
                basket_old.append(
                    {'parameter_id': item['parameter_id'], 'param': item['params'],
                     'perf': item['perf'], 'mean': mean[0], 'std': std[0], 'ei': ys[0]})

        # sort basket by ei, from big to small
        sorted(basket_old, key=lambda item: item['ei'], reverse=True)

        if len(basket_old) >= num:
            return basket_old[:num]
        else:
            return basket_old

    def _get_P_min_basket(self, basket):
        '''
        Parameters
        ----------
        basket: list, i.e. [{'params':[], 'mean':, 'std':}, {...}]

        Returns
        -------
        result: P_min, i.e. [0.1, 0.5, 0., 0.4]
        '''
        mean = []
        std = []
        for item in basket:
            mean.append(item['mean'])
            std.append(item['std'])

        return self._get_P_min(mean, std)

    def _get_P_min(self, mean, std):
        '''
        Parameters
        ----------
        mean: list, i.e. [0.4, 0.8 0.1, 0.6]
        std: list, i.e. [0.4, 0.3, 0.2, 0.6]

        Returns
        -------
        result: P_min, i.e. [0.1, 0.5, 0., 0.4]
        '''
        n_monte_carlo = 1000
        n_params = len(mean)
        P_min = np.zeros(n_params)
        for _ in range(n_monte_carlo):
            vals = np.empty(n_params)
            for i in range(n_params):
                vals[i] = self.random_state.normal(mean[i], std[i])
            P_min[vals.argmin()] += 1

        P_min /= n_monte_carlo
        return P_min

    def _cal_entropy(self, P_min):
        '''
        Parameters
        ----------
        params: P_min, i.e. [0.1, 0.5, 0.4]

        Returns
        -------
        result: entropy
        '''
        result = 0
        for p in P_min:
            if p != 0:  # p is not 0
                result -= p*np.log(p)
        return result
