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

import json
import matplotlib.pyplot as plt

# pylint:disable=import-error
from nni.mtsmac_advisor.test.util import PATH, COLORS


# pylint:disable=missing-docstring
# pylint:disable=no-member
# pylint:disable=invalid-name


def plot_rfr():
    with open('{}/result/experiment.json'.format(PATH)) as json_file:
        result = json.load(json_file)
        trials = result['trialMessage']

        num_epochs = 0
        metric_best = [0]

        for trial in trials:
            for inter in trial['intermediate']:
                val = inter['data']
                num_epochs += 1
                if float(val) > float(metric_best[-1]):
                    metric_best.append(val)
                else:
                    metric_best.append(metric_best[-1])

        plt.plot(range(len(metric_best)), metric_best, label='best metric')

    plt.title('Learning curve MNIST')
    plt.legend()
    plt.savefig('{}/image/res.png'.format(PATH))
    plt.close()


plot_rfr()
