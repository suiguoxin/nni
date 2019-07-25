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
mtsmac_advisor.py
"""

import logging
import json_tricks
import numpy as np

from nni.protocol import CommandType, send
from nni.msg_dispatcher_base import MsgDispatcherBase
from nni.utils import OptimizeMode, extract_scalar_reward

from nni.mtsmac_advisor.predictor import Predictor
from nni.mtsmac_advisor.target_space import TargetSpace

logger = logging.getLogger("MTSMAC_Advisor_AutoML")


class MTSMAC(MsgDispatcherBase):
    '''
    Multi-Task SMAC
    '''

    def __init__(self, optimize_mode='maximize', cold_start_num=5, max_epochs=27):
        """
        Parameters
        ----------
        optimize_mode: str
            optimize mode, 'maximize' or 'minimize'
        """
        super(MTSMAC, self).__init__()
        self.optimize_mode = OptimizeMode(optimize_mode)

        self._predictor = Predictor(multi_task=True)
        # num of random evaluations before GPR
        self._cold_start_num = cold_start_num
        self._max_epochs = max_epochs

        # target space
        self._space = None
        self._random_state = np.random.RandomState()  # pylint: disable=no-member

    def load_checkpoint(self):
        pass

    def save_checkpoint(self):
        pass

    def handle_initialize(self, data):
        """
        Parameters
        ----------
        data: JSON object
            search space
        """
        self.handle_update_search_space(data)
        send(CommandType.Initialized, '')

    def handle_request_trial_jobs(self, data):
        """
        Parameters
        ----------
        data: int
            number of trial jobs
        """
        params, target = self._space.get_train_data()
        if target.shape[0] > 0:
            logger.info("target shape:%s", target.shape[0])
            self._predictor.fit(params, target)
        for _ in range(data):
            self._request_one_trial_job()

    def _request_one_trial_job(self):
        """
        get one trial job, i.e., one hyperparameter configuration.
        If the number of trial result is lower than cold start number,
        gp will first randomly generate some parameters.
        Otherwise, choose the parameters by the Gussian Process Model

        Returns
        -------
        result : dict
        """
        logger.info("requst_one_trial_job called, len_completed: %s",
                    self._space.len_completed)
        if self._space.len < self._cold_start_num:  # TODO: support parallisim
            parameter_id, parameters = self._space.select_config_warmup()
            parameters['TRIAL_BUDGET'] = self._max_epochs
            parameters['PARAMETER_ID'] = parameter_id
        else:
            # generate one trial
            parameter_id, parameters = self._space.select_config(
                self._predictor)
            parameters['TRIAL_BUDGET'] = 1
            parameters['PARAMETER_ID'] = parameter_id
        res = {
            'parameter_id': parameter_id,
            'parameter_source': 'algorithm',
            'parameters': parameters
        }
        logger.info("Generate paramageters for trial job:\n %s", res)
        send(CommandType.NewTrialJob, json_tricks.dumps(res))

    def handle_update_search_space(self, data):
        """
        Parameters
        ----------
        data: JSON object
            search space
        """
        self._space = TargetSpace(
            data, random_state=self._random_state, max_epochs=self._max_epochs)

    def handle_trial_end(self, data):
        """
        Parameters
        ----------
        data: dict()
            it has 3 keys: 'trial_job_id', 'event', 'hyper_params'
            trial_job_id: the id generated by training service
            event: the job's state
            hyper_params: the hyperparameters (a string) generated and returned by tuner
        """
        # TODO: accept only SUCCEED trials
        hyper_params = json_tricks.loads(data['hyper_params'])
        parameter_id = hyper_params['parameter_id']
        self._space.trial_end(parameter_id)

    def handle_report_metric_data(self, data):
        """
        Parameters
        ----------
        data: dict()
            it has 5 keys 'parameter_id', 'value', 'trial_job_id', 'type', 'sequence'.

        Raises
        ------
        ValueError
            Data type not supported
        """
        # update target space and fit predictor
        value = extract_scalar_reward(data['value'])
        if self.optimize_mode == OptimizeMode.Minimize:
            value = -value
        if data['type'] == 'FINAL' or data['type'] == 'PERIODICAL':
            self._space.register(data['parameter_id'], value)
        else:
            raise ValueError(
                'Data type not supported: {}'.format(data['type']))

    def handle_add_customized_trial(self, data):
        pass

    def handle_import_data(self, data):
        pass
