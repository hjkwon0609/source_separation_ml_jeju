#  Compatibility imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import argparse
import math
import random
import os
# uncomment this line to suppress Tensorflow warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from six.moves import xrange as range

from util import *
import pdb
from time import gmtime, strftime

from config import Config

class SeparationModel():

    def create_layer(self, name, X, output_len):
        xavier = tf.contrib.layers.xavier_initializer()

        with tf.name_scope(name):
            W = tf.get_variable(name + 'W', shape=[X.get_shape()[2], output_len, 3], initializer=xavier, dtype=tf.float32)
            b = tf.get_variable(name + 'b', shape=[output_len, 3], dtype=tf.float32)

            # vector_product_matrix defined in util.py
            tf.summary.histogram(name + 'weight_histogram', W)
            tf.summary.histogram(name + 'bias_histogram', b)
            return tf.nn.tanh(vector_product_matrix(X, W) + b)

    def add_prediction_op(self, input_spec):
        curr = input_spec
        
        for i in xrange(Config.num_layers):
            layer_name = 'hidden%d' % (i + 1)
            curr = self.create_layer(layer_name, curr, Config.num_hidden)

        output = self.create_layer('output', curr, Config.output_size)
        curr_frame = output[:,:,:,1]

        # soft masking
        song_out, voice_out = tf.split(curr_frame, [Config.num_freq_bins, Config.num_freq_bins], 2)
        song_mask = tf.abs(song_out) / (tf.abs(song_out) + tf.abs(voice_out))
        voice_mask = 1 - song_mask

        input_spec_curr = input_spec[:,:,:,1]  # current frame of input spec
        song_output = tf.multiply(input_spec_curr, song_mask)
        voice_output = tf.multiply(input_spec_curr, voice_mask)

        self.output = tf.concat([song_output, voice_output], axis=2)

    def add_loss_op(self, target):
        delta = self.output - target  # only compare the current frame
        squared_error = tf.norm(delta, ord=2)

        l2_cost = tf.reduce_sum([tf.norm(v) for v in tf.trainable_variables() if len(v.get_shape().as_list()) == 3])

        self.loss = Config.l2_lambda * l2_cost + squared_error

        tf.summary.scalar("squared_error", squared_error)
        tf.summary.scalar("loss", self.loss)

    def add_training_op(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=Config.lr)
        grads = optimizer.compute_gradients(self.loss)
        for grad, var in grads:
            tf.summary.scalar('gradient_norm_%s' % (var), tf.norm(grad))
        self.optimizer = optimizer.apply_gradients(grads)
        tf.summary.scalar('learning_rate', optimizer._lr)

    def add_summary_op(self):
        self.merged_summary_op = tf.summary.merge_all()

    def train_on_batch(self, train_inputs_batch, train_targets_batch):
        self.add_prediction_op(train_inputs_batch)
        self.add_loss_op(train_targets_batch)
        self.add_training_op()
        self.add_summary_op()


    def print_results(self, train_inputs_batch, train_targets_batch):
        train_feed = self.create_feed_dict(train_inputs_batch, train_targets_batch)
        train_first_batch_preds = session.run(self.decoded_sequence, feed_dict=train_feed)
        compare_predicted_to_true(train_first_batch_preds, train_targets_batch)        

    def __init__(self, freq_weighted=None):
        pass

    

