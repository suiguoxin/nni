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
freezethaw_advisor.py
"""

import logging
import numpy as np

from nni.protocol import CommandType, send
from nni.msg_dispatcher_base import MsgDispatcherBase
from nni.utils import NodeType, OptimizeMode, extract_scalar_reward

from .target_space import TargetSpace
from .predictor import Predictor

logger = logging.getLogger("FreezeThaw_Advisor_AutoML")


class FreezeThaw(MsgDispatcherBase):
    """
    Parameters
    ----------
    R: int
        the maximum amount of resource that can be allocated to a single configuration
    eta: int
        the variable that controls the proportion of configurations discarded in each round of SuccessiveHalving
    optimize_mode: str
        optimize mode, 'maximize' or 'minimize'
    """

    def __init__(self, optimize_mode='maximize'):
        self.optimize_mode = OptimizeMode(optimize_mode)

        # target space
        self._space = None

        self._random_state = np.random.RandomState()  # pylint: disable=no-member

        # nu, alpha are GPR related params
        self._predictor = Predictor()

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
        for _ in range(data):
            self._request_one_trial_job()

    def _request_one_trial_job(self):
        """
        get one trial job, i.e., one hyperparameter configuration.
        """
        # generate one trial

    def handle_update_search_space(self, data):
        """
        Parameters
        ----------
        data: JSON object
            search space
        """
        self._space = TargetSpace(data, self._random_state)

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
        # update _space
        # update _gp

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
        # update _space
        # update _gp

    def handle_add_customized_trial(self, data):
        pass

    def handle_import_data(self, data):
        pass
