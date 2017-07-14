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
            b = tf.get_variable(name + 'b', shape=[output_len, 3], dtype=tf.float32)

            # tf.summary.histogram(name + 'weight_histogram', W)
            # tf.summary.histogram(name + 'bias_histogram', b)

            # vector_product_matrix defined in util.py
            output = vector_product_matrix(X, W) + b
            if activation_fn:
                output = activation_fn(output)
            # tf.summary.histogram(name + 'output', output)

            return output


    def add_prediction_op(self, input_spec):
        self.input = input_spec
        print(input_spec.get_shape())
        curr = tf.abs(self.input)  # only use magnitude for training

        activation_fn = tf.nn.relu if Config.use_relu else tf.nn.sigmoid
        for i in xrange(Config.num_layers):
            layer_name = 'hidden%d' % (i + 1)    
            curr = self.create_layer(layer_name, curr, Config.num_hidden, activation_fn)

        song_out = self.create_layer('output_song', curr, Config.num_freq_bins, activation_fn)
        voice_out = self.create_layer('output_voice', curr, Config.num_freq_bins, activation_fn)

        song_out, voice_out = song_out[:,:,1], voice_out[:,:,1]

        curr_frame = tf.concat([song_out, voice_out], axis=1)
        self.output = curr_frame

        # soft masking
        soft_song_mask = tf.abs(song_out) / (tf.abs(song_out) + tf.abs(voice_out) + 1e-10) # tf.abs(song_out) / (tf.abs(song_out) + tf.abs(voice_out))
        soft_voice_mask = 1 - soft_song_mask

        input_spec_curr = self.input[:,:,1]  # current frame of input spec
        soft_song_output = apply_mask(input_spec_curr, soft_song_mask)
        soft_voice_output = apply_mask(input_spec_curr, soft_voice_mask)

        self.soft_masked_output = tf.concat([soft_song_output, soft_voice_output], axis=1)


    def add_loss_op(self, target):
        self.target = target  # for outputting later
        real_target = tf.abs(self.target)

        delta = self.output - real_target 
        squared_error = tf.reduce_mean(tf.pow(delta, 2)) 

        l2_cost = tf.reduce_mean([tf.norm(v) for v in tf.trainable_variables() if len(v.get_shape().as_list()) == 3])

        self.loss = Config.l2_lambda * l2_cost + squared_error

        tf.summary.scalar("loss", self.loss)

        masked_loss = tf.abs(self.soft_masked_output) - real_target
        self.masked_loss = Config.l2_lambda * l2_cost + tf.reduce_mean(tf.pow(masked_loss, 2))
        tf.summary.scalar('masked_loss', self.masked_loss)


    def add_training_op(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=Config.lr, beta1=Config.beta1, beta2=Config.beta2)
        grads = optimizer.compute_gradients(self.masked_loss) if Config.use_mask_loss else optimizer.compute_gradients(self.loss)
        # for grad, var in grads:
        #     tf.summary.histogram('gradient_norm_%s' % (var), grad)
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
