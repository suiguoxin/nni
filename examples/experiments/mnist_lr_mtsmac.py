"""A deep MNIST classifier using convolutional layers."""

import argparse
import logging
import math
import tempfile
import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import nni

FLAGS = None

logger = logging.getLogger('mnist_AutoML')


class MnistNetwork(object):
    '''
    MnistNetwork is for initializing and building basic network for mnist.
    '''

    def __init__(self,
                 learning_rate,
                 constraints_weights,
                 l2_norm,
                 x_dim=784,
                 y_dim=10):

        self.constraints_weights = constraints_weights
        self.l2_norm = l2_norm
        self.learning_rate = learning_rate
        self.x_dim = x_dim
        self.y_dim = y_dim

        self.images = tf.placeholder(
            tf.float32, [None, self.x_dim], name='input_x')
        self.labels = tf.placeholder(
            tf.float32, [None, self.y_dim], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.train_step = None
        self.accuracy = None

    def build_network(self):
        '''
        Building network for mnist
        '''

        # Dropout
        with tf.name_scope('dropout'):
            layer_1_out = tf.nn.dropout(self.images, self.keep_prob)

        # Linear Regression layer
        with tf.name_scope('lr'):
            W = weight_variable([self.x_dim, self.y_dim])
            b = bias_variable([10])

            W_clip = tf.clip_by_norm(W, self.constraints_weights)
            y = tf.nn.softmax(tf.matmul(layer_1_out, W_clip) + b)

        with tf.name_scope('loss'):
            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=y))
            l2_loss = tf.multiply(self.l2_norm, tf.nn.l2_loss(W))
        with tf.name_scope('adam_optimizer'):
            self.train_step = tf.train.AdamOptimizer(
                self.learning_rate).minimize(cross_entropy + l2_loss)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(
                tf.argmax(y, 1), tf.argmax(self.labels, 1))
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32))


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def download_mnist_retry(data_dir, max_num_retries=20):
    """Try to download mnist dataset and avoid errors"""
    for _ in range(max_num_retries):
        try:
            return input_data.read_data_sets(data_dir, one_hot=True)
        except tf.errors.AlreadyExistsError:
            time.sleep(1)
    raise Exception("Failed to download MNIST.")


def main(params):
    '''
    Main function, build mnist network, run and send result to NNI.
    '''
    # Import data
    mnist = download_mnist_retry(params['data_dir'])
    print('Mnist download data done.')
    logger.debug('Mnist download data done.')

    # Create the model
    # Build the graph for the deep net
    mnist_network = MnistNetwork(constraints_weights=params['constraints_weights'],
                                 l2_norm=params['l2_norm'],
                                 learning_rate=params['learning_rate'])
    mnist_network.build_network()
    logger.debug('Mnist build network done.')

    # Write log
    graph_location = tempfile.mkdtemp()
    logger.debug('Saving graph to: %s', graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    test_acc = 0.0
    saver = tf.train.Saver()
    checkpoint_file = '{0}/{1}/model_{2}.ckpt'.format(
        params['model_dir'], params['experiment_id'], params['parameter_id'])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(params['batch_num']):
            batch = mnist.train.next_batch(params['batch_size'])
            mnist_network.train_step.run(feed_dict={mnist_network.images: batch[0],
                                                    mnist_network.labels: batch[1],
                                                    mnist_network.keep_prob: 1 - params['dropout_rate']}
                                         )

            if i % 100 == 0:
                test_acc = mnist_network.accuracy.eval(
                    feed_dict={mnist_network.images: mnist.test.images,
                               mnist_network.labels: mnist.test.labels,
                               mnist_network.keep_prob: 1.0})

                nni.report_intermediate_result(test_acc)
                logger.debug('test accuracy %g', test_acc)
                logger.debug('Pipe send intermediate result done.')

        save_path = saver.save(sess, checkpoint_file)
        print("Model saved in path: %s" % save_path)
        test_acc = mnist_network.accuracy.eval(
            feed_dict={mnist_network.images: mnist.test.images,
                       mnist_network.labels: mnist.test.labels,
                       mnist_network.keep_prob: 1.0})

        nni.report_final_result(test_acc)
        logger.debug('Final result is %g', test_acc)
        logger.debug('Send final result done.')


def get_params():
    ''' Get parameters from command line '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default='/tmp/tensorflow/mnist/input_data', help="data directory")
    parser.add_argument("--model_dir", type=str,
                        default='/tmp/tensorflow/mnist/model_data', help="data directory")
    parser.add_argument("--constraints_weights", type=float,
                        default=1, help="dropout rate")
    parser.add_argument("--l2_norm", type=float,
                        default=0.5, help="dropout rate")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dropout_rate", type=float,
                        default=0.5, help="dropout rate")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_num", type=int, default=2700)

    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    try:
        # get parameters form tuner
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        tuner_params['batch_num'] = tuner_params['TRIAL_BUDGET'] * 100
        tuner_params['parameter_id'] = tuner_params['PARAMETER_ID']
        tuner_params['experiment_id'] = nni.get_experiment_id()
        params = vars(get_params())
        params.update(tuner_params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
