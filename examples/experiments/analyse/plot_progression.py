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
plot_progression.py
"""

import json
import matplotlib.pyplot as plt
from nni.mtsmac_advisor.test.util import COLORS

PATH = './examples/experiments'


def plot_progression():
    '''
    Fig 4
    '''
    file_name = 'mnist_lr/mtsmac_710.json'

    with open('{}/result/{}'.format(PATH, file_name)) as json_file:
        result = json.load(json_file)
        trials = result['trialMessage']

        perf_last = {}
        colors_next = {}
        counts_param = {}

        parameter_id_prev = -1
        vals = []
        for trial in trials:
            parameter_id = trial['hyperParameters']['parameter_id']
            if parameter_id != parameter_id_prev:
                if vals:
                    if parameter_id_prev in colors_next:
                        idx_color = colors_next[parameter_id_prev]
                    else:
                        idx_color = 0
                    colors_next[parameter_id_prev] = (idx_color + 1) % 5
                    #if(parameter_id_prev < 00): # draw only first 30 
                    plt.plot(range(counts_param[parameter_id_prev] - len(
                            vals), counts_param[parameter_id_prev]), vals, color=COLORS[idx_color])
                if parameter_id in perf_last:
                    vals = [perf_last[parameter_id]]
                else:
                    vals = [0]
                parameter_id_prev = parameter_id
            for inter in trial['intermediate']:
                val = float(inter['data'])
                vals.append(val)
                if parameter_id not in counts_param:
                    counts_param[parameter_id] = 1
                else:
                    counts_param[parameter_id] += 1
            perf_last[parameter_id] = val

    plt.xlabel('Epochs')
    plt.ylabel('Default Metric')
    plt.ylim(0, 1)
    plt.title('MNIST Progression')
    plt.savefig('{}/analyse/image/progression_mtsmac710.png'.format(PATH))
    plt.close()

plot_progression()
