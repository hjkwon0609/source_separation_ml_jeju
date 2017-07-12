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

    def create_layer(self, name, X, output_len, activation_fn=None):
        xavier = tf.contrib.layers.xavier_initializer()

        with tf.name_scope(name):
            W = tf.get_variable(name + 'W', shape=[X.get_shape()[1], output_len, 3], initializer=xavier, dtype=tf.float32)
            # W = tf.get_variable(name + 'W', shape=[X.get_shape()[1], output_len], initializer=xavier, dtype=tf.float32)

            b = tf.get_variable(name + 'b', shape=[output_len, 3], dtype=tf.float32)
            # b = tf.get_variable(name + 'b', shape=[output_len], dtype=tf.float32)

            tf.summary.histogram(name + 'weight_histogram', W)
            tf.summary.histogram(name + 'bias_histogram', b)

            # vector_product_matrix defined in util.py
            output = vector_product_matrix(X, W) + b
            # output = tf.matmul(X, W) + b
            if activation_fn:
                output = activation_fn(output)
            tf.summary.histogram(name + 'output', output)

            return output


    def add_prediction_op(self, input_spec):
        self.input = input_spec
        curr = tf.abs(self.input)  # use real component for training

        for i in xrange(Config.num_layers):
            layer_name = 'hidden%d' % (i + 1)
            activation_fn = tf.nn.relu #if i < Config.num_layers - 1 else tf.nn.relu
            curr = self.create_layer(layer_name, curr, Config.num_hidden, activation_fn)

        song_out = self.create_layer('output_song', curr, Config.num_freq_bins, tf.nn.relu)
        voice_out = self.create_layer('output_voice', curr, Config.num_freq_bins, tf.nn.relu)

        song_out, voice_out = song_out[:,:,1], voice_out[:,:,1]

        curr_frame = tf.concat([song_out, voice_out], axis=1)
        # curr_frame = output
        self.output = curr_frame

        # hard masking
        hard_song_mask = tf.cast(tf.greater(tf.abs(song_out), tf.abs(voice_out)), tf.int32)  #tf.abs(song_out)  (tf.abs(song_out) + tf.abs(voice_out))
        hard_voice_mask = 1 - hard_song_mask
        # soft masking
        soft_song_mask = song_out / (song_out + voice_out + 1e-10) # tf.abs(song_out) / (tf.abs(song_out) + tf.abs(voice_out))
        soft_voice_mask = 1 - soft_song_mask

        input_spec_curr = input_spec[:,:,1]  # current frame of input spec
        # input_spec_curr = input_spec  # current frame of input spec
        hard_song_output = tf.multiply(input_spec_curr, tf.cast(hard_song_mask, tf.complex64))
        hard_voice_output = tf.multiply(input_spec_curr, tf.cast(hard_voice_mask, tf.complex64))
        soft_song_output = tf.multiply(input_spec_curr, tf.cast(soft_song_mask, tf.complex64))
        soft_voice_output = tf.multiply(input_spec_curr, tf.cast(soft_voice_mask, tf.complex64))
        # song_output = np.multiply(input_spec_curr, song_mask)
        # voice_output = np.multiply(input_spec_curr, voice_mask)

        self.hard_masked_output = tf.concat([hard_song_output, hard_voice_output], axis=1)
        self.soft_masked_output = tf.concat([soft_song_output, soft_voice_output], axis=1)


    def add_loss_op(self, target):
        self.target = target  # for outputting later
        real_target = tf.abs(self.target)

        delta = self.output - real_target  # only compare the current frame
        # squared_error = tf.norm(delta, ord=2)
        squared_error = tf.reduce_sum(tf.pow(delta, 2)) 
        mean_error = squared_error / self.output.get_shape().as_list()[0]

        l2_cost = tf.reduce_sum([tf.norm(v) for v in tf.trainable_variables() if len(v.get_shape().as_list()) == 3])

        self.loss = Config.l2_lambda * l2_cost + mean_error
        # self.squared_error = squared_error

        tf.summary.scalar("squared_error", mean_error)
        tf.summary.scalar("loss", self.loss)

        # song_target, voice_target = tf.split(self.target, [Config.num_freq_bins, Config.num_freq_bins], axis=1)

        masked_loss = tf.abs(self.soft_masked_output) - real_target
        self.masked_loss = Config.l2_lambda * l2_cost + tf.reduce_sum(tf.pow(masked_loss, 2)) / self.output.get_shape().as_list()[0]
        tf.summary.scalar('masked_loss', self.masked_loss)


    def add_training_op(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=Config.lr, beta1=Config.beta1, beta2=Config.beta2)
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=Config.lr)
        grads = optimizer.compute_gradients(self.masked_loss)
        for grad, var in grads:
            tf.summary.histogram('gradient_norm_%s' % (var), grad)
        self.optimizer = optimizer.apply_gradients(grads, global_step=self.global_step)


    def add_summary_op(self):
        self.merged_summary_op = tf.summary.merge_all()


    def run_on_batch(self, train_inputs_batch, train_targets_batch):
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

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

    

