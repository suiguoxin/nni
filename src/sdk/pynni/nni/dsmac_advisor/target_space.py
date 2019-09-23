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

from nni.dsmac_advisor.util import ei
from nni.dsmac_advisor.predictor import Predictor

logger = logging.getLogger("DSMAC_Advisor_AutoML")

# pylint:disable=invalid-name


class TargetSpace():
    """
    Holds the param-space coordinates (X) and target values (Y)
    """

    def __init__(self, search_space, max_epochs, random_state=None, utility='AEI'):
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
        self.utility = utility

        # Get the name of the parameters
        self._keys = sorted(search_space)

        # Create an array with parameters types/bounds
        self._bounds = np.array([item[1] for item in sorted(
            search_space.items(), key=lambda x: x[0])])

        # used for saving all info about params
        # [{id: 0, params:[], perf: []}, {...}]
        self.hyper_configs = []

        # params/target used for fitting predictor
        self._params = np.empty(shape=(0, self.dim))
        self._target = np.empty(shape=(0), dtype=object)

        self.next_param_id = 0
        self.max_epochs = max_epochs

        self._len_completed = 0
        self._len_running = 0
        self._y_max = 0

        self._budget = {}

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

    @property
    def len_running(self):
        '''length of completed trials'''
        return self._len_running

    def get_train_data(self):
        '''
        Returns
        -------
        params, target : numpy array
        '''
        params = np.empty(shape=(0, self.dim))
        target = np.empty(shape=(0), dtype=object)
        for item in self.hyper_configs:
            if len(item['perf']) >= 0:
                params = np.vstack((params, item['params']))
                target = np.append(target, ['placeholder'])
                target[-1] = item['perf']  # TODO: more pythonic
        # logger.debug("get_train_data:")
        # logger.debug("params:%s", params)
        # logger.debug("target:%s", target)
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
        self._len_running += 1

    def register(self, parameter_id, value):
        '''
        insert a result into target space
        '''
        self.hyper_configs[parameter_id]['perf'].append(value)
        if value > self._y_max:
            self._y_max = value

    def trial_end(self, parameter_id):
        '''
        trial end
        '''
        logger.info("Trial end, parameter_id: %s", parameter_id)
        if len(self.hyper_configs[parameter_id]['perf']) >= self.max_epochs:
            self.hyper_configs[parameter_id]['status'] = 'FINISH'
            # update internal flag variables
            self._len_completed += 1
            self._len_running -= 1

    def random_sample(self):
        """
        Creates a random point within the bounds of the space.
        """
        params = np.empty(self.dim, dtype=object)
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
        self._budget[parameter_id] = max(round(self.max_epochs/10), 1)

        parameter_json = self.array_to_params(params)
        logger.info("Generate paramageter for warm up :\n %s", parameter_json)

        parameter_json['TRIAL_BUDGET'] = self._budget[parameter_id]

        return parameter_id, parameter_json

    def select_config(self, predictor):
        '''
        Parameters
        ----------
        utility: string, choice in ['AEI', 'TS', 'ES']
        '''
        if self.utility == 'AEI':
            # construct the basket of configs
            basket_new = self._get_basket_new(predictor, num=1, metric='AEI')
            basket_old = self._get_basket_old(predictor, num=1, metric='AEI')
            # seleted the param with max AEI
            if basket_old and basket_old[0]['acq'] >= basket_new[0]['acq']:
                param_selected = basket_old[0]
                logger.debug("old config selected by AEI")
            else:
                param_selected = basket_new[0]
                logger.debug("new config selected by AEI")
        elif self.utility == 'TS':
            # construct the basket of configs
            basket_new = self._get_basket_new(
                predictor, num=self._len_running, metric='MEI')
            basket_old = self._get_basket_old(
                predictor, num=self._len_running, metric='MEI')
            basket = np.append(basket_new, basket_old)
            # get P_max and current entropy
            P_max = self._get_P_max_basket(basket, num_monte_carlo=1)
            param_selected = basket[P_max.argmax()]
            logger.debug("new config selected by TS")
        elif self.utility == 'ES':
            # entropy search
            # Step 1: get a basket by 'Expected Improvement'
            # Step 2: get a config by 'Information Gain'
            num_new = 10
            num_old = 10
            basket_new = self._get_basket_new(predictor, num_new, metric='MEI')
            basket_old = self._get_basket_old(predictor, num_old, metric='MEI')
            basket = np.append(basket_new, basket_old)

            logger.debug("basket_new %s", basket_new)
            logger.debug("basket_old %s", basket_old)
            logger.debug("basket %s", basket)
            # vector format of params in the basket
            params = np.empty((0, len(basket[0]['param'])))
            for i, item in enumerate(basket):
                params = np.vstack((params, item['param']))
            logger.debug("params %s", params)

            # get P_max and current entropy
            P_max = self._get_P_max_basket(basket)
            H = self._cal_entropy(P_max)
            logger.debug("P_max %s", P_max)
            logger.debug("H %s", H)

            # get information gain
            logger.debug("------------fantasize period---------------")
            a = np.zeros(len(basket))
            n_fant = 5
            X, y = self.get_train_data()
            mean, std = predictor.predict(params, final_only=False)
            #logger.debug("mean predicted %s", mean)
            #logger.debug("std predicted %s", std)
            for i, item in enumerate(basket):
                # logger.debug("fantasize element %s in the basket", i)
                for _ in range(n_fant):
                    # logger.debug("fantasize round %s", j)
                    if i < num_new:
                        # fantasize observations of a new point in given budget
                        budget = max(round(self.max_epochs/10), 1)
                        obs = [self.random_state.normal(
                            mean[i][k], std[i][k]) for k in range(budget)]
                        # add fantsized points to fake training data
                        X_fant = np.vstack((X, item['param']))
                        y_fant = np.append(y, ['place_holder'])
                        y_fant[-1] = obs
                    else:
                        # fantasize observations of a old point in given budget
                        cur_epoch = len(item['perf'])
                        parameter_id = item['parameter_id']
                        budget = min(max(round(self.max_epochs/10),
                                         1), self.max_epochs - cur_epoch)
                        obs = [self.random_state.normal(
                            mean[i][cur_epoch+k], std[i][cur_epoch+k]) for k in range(budget)]

                        # add fantsized points to fake training data
                        X_fant = X.copy()
                        y_fant = y.copy()
                        for k in range(X_fant.shape[0]):
                            if np.array_equal(item['param'], X_fant[k]):
                                y_fant[k] = y_fant[k].copy()
                                y_fant[k].extend(obs)
                                break

                    # conditioned on the observation, re-compute P_max and H
                    # fit a new predictor with fantsized point added in training data
                    predictor_fant = Predictor(
                        max_epochs=self.max_epochs, multi_task=True)
                    predictor_fant.fit(X_fant, y_fant)
                    # re-calculate P_max, H
                    mean_fant, std_fant = predictor_fant.predict(
                        params, final_only=False)

                    # logger.debug("mean fantasize %s", mean)
                    # logger.debug("std fantasize %s", std)
                    P_max_fant = self._get_P_max(mean_fant, std_fant)
                    H_fant = self._cal_entropy(P_max_fant)
                    # logger.debug("P_max_fant %s", P_max_fant)
                    # logger.debug("H_fant %s", H_fant)
                    # average over n_fant
                    a[i] += (H_fant / n_fant)
            param_selected = basket[a.argmin()]
            logger.debug("P_max %s", P_max)
            logger.debug("a %s", a)
            logger.debug(
                "param index %s in the basket is selected by H", a.argmin())

        logger.debug("param_selected %s", param_selected)
        param = param_selected['param']
        budget = max(round(self.max_epochs/10), 1)
        if 'parameter_id' not in param_selected:  # new config is selected
            parameter_id = self.next_param_id
            self.next_param_id += 1
            self.register_new_config(parameter_id, param)
        else:  # old config is selected
            parameter_id = param_selected['parameter_id']
            budget = min(budget, self.max_epochs -
                         len(self.hyper_configs[parameter_id]['perf']))

        self._budget[parameter_id] = budget

        parameter_json = self.array_to_params(param)
        parameter_json['TRIAL_BUDGET'] = self._budget[parameter_id]
        logger.info("Generate paramageter :\n %s", parameter_json)

        return parameter_id, parameter_json

    def _get_acquisition_values_new_config(self, predictor, X, metric):
        '''
        calculate acquisition value according to the given metric

        Parameters
        ----------
        X: parameters
        metric: a string in ['AEI', 'MEI', 'FEI']
            AEI: Average Expected Improvement in the following epochs
            MEI: Max Expected Improvement in the following epochs
            FEI: Final Expected Improvement(last epoch)

        Returns
        -------
        ys: one-dimensional list
        '''
        if metric == 'AEI':
            mean, std = predictor.predict(X, final_only=False)
            ys = np.zeros(len(X))
            for t in range(self.max_epochs):
                ys += ei(mean[:, t], std[:, t], y_max=self._y_max * 0.98)
            ys /= self.max_epochs
            ys.tolist()
            logger.debug("_get_basket_new")
            logger.debug("mean.shape: %s", mean.shape)
            logger.debug("ys : %s", len(ys))
        elif metric == 'MEI':
            mean, std = predictor.predict(X, final_only=False)
            ys = []
            for i in range(len(X)):
                # np.max(ei_i)/(np.argmax(ei_i) + 1)
                ei_i = ei(mean[i, :], std[i, :], y_max=self._y_max*0.98)
                ys.append(np.max(ei_i))
        elif metric == 'FEI':
            mean, std = predictor.predict(X, final_only=True)
            ys = ei(mean, std, y_max=self._y_max)

        return ys

    def _get_basket_new(self, predictor, num, num_warmup=10000, metric=None):
        '''
        select a basket from new configs

        Parameters
        ----------
        num: the amount of configurations returned
        metric: a string in ['AEI', 'MEI', 'FEI']
            AEI: Average Expected Improvement in the following epochs
            MEI: Max Expected Improvement in the following epochs
            FEI: Final Expected Improvement(last epoch)

        Returns
        -------
        basket_new: list, i.e. [{'parameter_id','param':[], 'mean':, 'std': , 'acq': }, {...}]
        '''
        # Warm up with random points
        x_tries = [self.random_sample()
                   for _ in range(int(num_warmup))]
        ys = self._get_acquisition_values_new_config(
            predictor, x_tries, metric)

        x_tries = [x for x, _ in sorted(
            zip(x_tries, ys), key=lambda pair: pair[1], reverse=True)]
        ys = sorted(ys, reverse=True)

        # local search with the 10 top random points
        start_points = x_tries[:10]
        acq_val_incumbent = ys[:10]

        logger.debug("start_points: %s", start_points)
        logger.debug("acq_val_incumbent: %s", acq_val_incumbent)
        max_steps = 10
        for i, start_point in enumerate(start_points):
            incumbent = start_point
            acq_val = acq_val_incumbent[i]
            changed_inc = False
            for _ in range(max_steps):
                all_neighbours = self._get_one_exchange_neighbourhoods(
                    incumbent)
                ys_neighbours = self._get_acquisition_values_new_config(
                    predictor, all_neighbours, metric)
                # logger.info("ys_neighbours :", len(ys_neighbours))
                if max(ys_neighbours) > acq_val:  # TODO: check
                    # restart from the better neigobour next step
                    acq_val = max(ys_neighbours)
                    incumbent = all_neighbours[ys_neighbours.index(acq_val)]
                    changed_inc = True
                else:  # stop the local search once none of the neighbours of the start point has larger acquisition value
                    break
            if changed_inc:
                logger.debug(
                    "For start point : %s, best neighbour found: %s, with ei : %s", start_point, incumbent, acq_val)
                x_tries.append(incumbent)
                ys = np.append(ys, acq_val)

        # re-comput mean std for all the 10010(max) new points
        mean, std = predictor.predict(x_tries, final_only=False)
        basket_new = []
        for i, x_i in enumerate(x_tries):
            basket_new.append(
                {'param': x_i, 'mean': mean[i], 'std': std[i], 'acq': ys[i]})
        # sort basket by acquisition value, from big to small
        basket_new = sorted(
            basket_new, key=lambda item: item['acq'], reverse=True)

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

        # logger.debug("neighbours found for param%s: \n %s", param, neighbours)

        return neighbours

    def _get_basket_old(self, predictor, num, metric=None):
        '''
        select a basket from running configs

        Parameters
        ----------
        num: the amount of configurations returned
        metric: a string in ['AEI', 'MEI', 'FEI]

        Returns
        -------
        basket_old: [{'parameter_id': ,'param':[], 'perf': , 'mean':, 'std': , 'acq'}, {...}]
        '''
        basket_old = []
        if metric == 'AEI':
            for item in self.hyper_configs:
                if item['status'] == 'RUNNING':
                    mean, std = predictor.predict(
                        [item['params']], final_only=False)
                    ys = []
                    logger.debug("get_basket_old")
                    logger.debug("mean: %s", mean)
                    logger.debug("std: %s", std)
                    for t in range(len(item['perf']), self.max_epochs):
                        ys.append(ei(mean[:, t], std[:, t],
                                     y_max=self._y_max*0.98)[0])
                    logger.debug("ys: %s", ys)
                    logger.debug("ei: %s", np.mean(ys))
                    basket_old.append(
                        {'parameter_id': item['parameter_id'], 'param': item['params'],
                         'perf': item['perf'], 'mean': mean[0], 'std': std[0], 'acq': np.mean(ys)})
        elif metric == 'MEI':
            for item in self.hyper_configs:
                if item['status'] == 'RUNNING':
                    mean, std = predictor.predict(
                        [item['params']], final_only=False)
                    ys = []
                    for t in range(len(item['perf']), self.max_epochs):
                        ys.append(ei(mean[:, t], std[:, t],
                                     y_max=self._y_max*0.98)[0])
                    logger.debug("ys: %s", ys)
                    logger.debug("ei: %s", max(ys))
                    basket_old.append(
                        {'parameter_id': item['parameter_id'], 'param': item['params'],
                         'perf': item['perf'], 'mean': mean[0], 'std': std[0], 'acq': max(ys)})
        elif metric == 'FEI':
            for item in self.hyper_configs:
                if item['status'] == 'RUNNING':
                    mean, std = predictor.predict(
                        [item['params']], final_only=True)
                    # discount = 1-0.2*(self.max_epochs -
                    #                 len(item['perf']))/self.max_epochs
                    ys = ei(mean, std, y_max=self._y_max*0.98)
                    basket_old.append(
                        {'parameter_id': item['parameter_id'], 'param': item['params'],
                         'perf': item['perf'], 'mean': mean[0], 'std': std[0], 'acq': ys[0]})

        # sort basket by acq, from big to small
        basket_old = sorted(
            basket_old, key=lambda item: item['acq'], reverse=True)

        return basket_old[:num]

    def _get_P_max(self, mean, std, num_monte_carlo=1000):
        '''
        Parameters
        ----------
        mean : numpy array of type object, length of sublist maybe different.
            i.e.[[0.4, 0.8, 0.1, 0.6], [0.1, 0.6]]
        std : numpy array of type object, length of sublist maybe different.
            i.e. [[0.4, 0.3, 0.2, 0.6], [0.2, 0.6]]

        Returns
        -------
        P_max: numpy array, i.e. [0.1, 0.5, 0., 0.4]
        '''
        n_monte_carlo = num_monte_carlo
        n_params = len(mean)
        P_max = np.zeros(n_params)
        for _ in range(n_monte_carlo):
            while True:
                vals = np.empty(n_params)
                for i in range(n_params):
                    k = self.random_state.randint(len(mean[i]))
                    vals[i] = self.random_state.normal(mean[i][k], std[i][k])
                if np.max(vals) >= self._y_max * 0.95:
                    P_max[vals.argmax()] += 1
                    break

        P_max /= n_monte_carlo
        return P_max

    def _get_P_max_basket(self, basket, num_monte_carlo=1000):
        '''
        Parameters
        ----------
        basket: list, i.e. [{'params':[], 'mean':, 'std':}, {...}]

        Returns
        -------
        P_max: numpy array i.e. [0.1, 0.5, 0., 0.4]
        '''
        mean = np.empty(shape=(
            len(basket)), dtype=object)  # TODO: delete shape=(0),check the whole file
        std = np.empty(shape=(len(basket)), dtype=object)
        for i, item in enumerate(basket):
            logger.info("item %s", item)
            logger.info("item['mean'] %s", item['mean'])
            if 'perf' in item:
                idx = len(item['perf'])
            else:
                idx = 0
            mean[i] = item['mean'][idx:]
            std[i] = item['std'][idx:]

        return self._get_P_max(mean, std, num_monte_carlo)

    def _cal_entropy(self, P_max):
        '''
        Parameters
        ----------
        params: P_max, i.e. [0.1, 0.5, 0.4]

        Returns
        -------
        result: entropy
        '''
        result = 0
        for p in P_max:
            if p != 0:  # p is not 0
                result -= p*np.log(p)
        return result
