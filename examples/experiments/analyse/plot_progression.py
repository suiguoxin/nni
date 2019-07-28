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
import numpy as np
import matplotlib.pyplot as plt
import imageio

from nni.mtsmac_advisor.test.util import COLORS

PATH = './examples/experiments'


def plot_progression_png(experiment, tuner):
    '''
    Fig 4
    '''
    with open('{}/result/{}/{}.json'.format(PATH, experiment, tuner)) as json_file:
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
    plt.savefig('{}/analyse/image/{}/progression_{}.png'.format(PATH, experiment, tuner))
    plt.close()


def _plot_progression_limit(experiment, tuner, num_trials):
    '''
    Fig 4
    '''
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set(xlabel='Epochs', ylabel='Default Metric', title='MNIST Progression', ylim=(0,1))

    with open('{}/result/{}/{}.json'.format(PATH, experiment, tuner)) as json_file:
        result = json.load(json_file)
        trials = result['trialMessage']

        perf_last = {}
        colors_next = {}
        counts_param = {}

        parameter_id_prev = -1
        vals = []
        for trial in trials[:num_trials]:
            parameter_id = trial['hyperParameters']['parameter_id']
            if parameter_id != parameter_id_prev:
                if vals:
                    if parameter_id_prev in colors_next:
                        idx_color = colors_next[parameter_id_prev]
                    else:
                        idx_color = 0
                    colors_next[parameter_id_prev] = (idx_color + 1) % 5
                    ax.plot(range(counts_param[parameter_id_prev] - len(
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

    # Used to return the plot as an image rray
    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()

    return image


def plot_progression_gif(experiment, tuner, num_trials):
    '''plot gif'''
    imageio.mimsave('{}/analyse/image/{}/progression_{}.gif'.format(PATH, experiment, tuner),
        [_plot_progression_limit(experiment, tuner, i) for i in range(1, num_trials)], fps=10)

plot_progression_png('mnist_lr', 'mtsmac_plus2')
plot_progression_gif('mnist_lr', 'mtsmac_plus2', 600)

