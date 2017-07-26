#  Compatibility imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import argparse
import math
import random
import os
import sys
# uncomment this line to suppress Tensorflow warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from six.moves import xrange as range

import pdb
from time import gmtime, strftime

from embedding_config import EmbeddingConfig
from tensorflow.contrib.tensorboard.plugins import projector

from util import *

class SeparationModel():

    def create_layer(self, name, X, output_len, activation_fn=None):
        xavier = tf.contrib.layers.xavier_initializer()

        with tf.name_scope(name):
            W = tf.get_variable(name + 'W', shape=[X.get_shape()[1], output_len, 3], initializer=xavier, dtype=tf.float32)
            b = tf.get_variable(name + 'b', shape=[output_len, 3], dtype=tf.float32)
            # vector_product_matrix defined in util.py
            output = vector_product_matrix(X, W) + b
            if activation_fn:
                output = activation_fn(output)
            return output


    def add_prediction_op(self, input_spec):
        self.input = input_spec
        curr = tf.abs(self.input)  # use real component for training
        print('input shape: %s' % (str(curr.get_shape())))
        if EmbeddingConfig.use_vpnn:
            curr = tf.transpose(curr, [2, 0, 1])
            curr -= self.stats[0][2]
            curr /= self.stats[1][2]
            curr = tf.transpose(curr, [1, 2, 0])
        else:
            curr -= self.stats[0][2]  # subtract mean for each freq bin
            curr /= self.stats[1][2]  # divide by variance of mixed

        activation_fn = tf.nn.tanh
        # initializer = tf.contrib.layers.xavier_initializer()
        initializer = tf.truncated_normal_initializer(stddev=0.0001)

        if EmbeddingConfig.use_vpnn:
            for i in xrange(EmbeddingConfig.num_layers):
                layer_name = 'hidden%d' % (i + 1)
                curr = self.create_layer(layer_name, curr, EmbeddingConfig.num_hidden, activation_fn)
            curr_frame = curr[:,:,1]
        else:
            if EmbeddingConfig.use_gru:
                cell_fw = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(EmbeddingConfig.num_hidden, kernel_initializer=initializer), output_keep_prob=EmbeddingConfig.keep_prob) for _ in xrange(EmbeddingConfig.num_layers)])
                cell_bw = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(EmbeddingConfig.num_hidden, kernel_initializer=initializer), output_keep_prob=EmbeddingConfig.keep_prob) for _ in xrange(EmbeddingConfig.num_layers)])
            else:
                cell_fw = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(EmbeddingConfig.num_hidden, initializer=initializer), output_keep_prob=EmbeddingConfig.keep_prob) for _ in xrange(EmbeddingConfig.num_layers)])
                cell_bw = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(EmbeddingConfig.num_hidden, initializer=initializer), output_keep_prob=EmbeddingConfig.keep_prob) for _ in xrange(EmbeddingConfig.num_layers)])

            output, state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, curr, dtype=tf.float32)
            fw_output, bw_output = output
            final_output = tf.concat([fw_output, bw_output], axis=2)
            curr_frame = tf.reshape(final_output, [-1, EmbeddingConfig.num_hidden * 2])

        embedding_list = []
        for i in xrange(EmbeddingConfig.num_freq_bins):
            embedding = tf.layers.dense(curr_frame, EmbeddingConfig.embedding_dim, activation_fn, kernel_initializer=initializer)
            # unit norm
            embedding = tf.divide(embedding, tf.norm(embedding, ord=2, axis=1, keep_dims=True) + 1e-6)
            # embedding = tf.divide(embedding, tf.reduce_max(embedding, axis=1, keep_dims=True) + 1e-10)
            embedding_list.append(embedding)

        self.embedding = tf.transpose(tf.stack(embedding_list), [1, 0, 2])  # shape: [batch_size, frequency bins, embedding dim]  

    def add_loss_op(self, voice_spec, song_spec):
        if not EmbeddingConfig.use_vpnn:
            # concatenate all batches into one axis  [num_batches * time_frames, freq_bins]
            voice_spec = tf.reshape(voice_spec, [-1, EmbeddingConfig.num_freq_bins])
            song_spec = tf.reshape(song_spec, [-1, EmbeddingConfig.num_freq_bins])

        self.voice_spec = voice_spec  # for output
        self.song_spec = song_spec

        song_spec_mask = tf.cast(tf.abs(song_spec) > tf.abs(voice_spec), tf.float32)
        voice_spec_mask =  tf.ones(song_spec_mask.get_shape()) - song_spec_mask

        V = self.embedding
        Y = tf.transpose([song_spec_mask, voice_spec_mask], [1, 2, 0])  # [num_batch, num_freq_bins, 2]

        # A_pred = tf.matmul(V, tf.transpose(V, [0, 2, 1]))
        # A_target = tf.matmul(Y, tf.transpose(Y, [0, 2, 1]))
        error = tf.reduce_mean(tf.square(tf.matmul(V, tf.transpose(V, [0, 2, 1])) - tf.matmul(Y, tf.transpose(Y, [0, 2, 1]))))  # average error per TF bin

        # tf.summary.histogram('a_same cluster embedding distribution', A_pred * A_target)
        # tf.summary.histogram('a_different cluster embedding distribution', A_pred * (1 - A_target))

        # tf.summary.histogram('V', V)
        # tf.summary.histogram('V V^T', A_pred)

        l2_cost = tf.reduce_sum([tf.norm(v) for v in tf.trainable_variables() if len(v.get_shape().as_list()) == 2])

        self.loss = EmbeddingConfig.l2_lambda * l2_cost + error
        
        # tf.summary.scalar("avg_loss", self.loss)
        # tf.summary.scalar('regularizer cost', EmbeddingConfig.l2_lambda * l2_cost)

    def add_training_op(self):
        # learning_rate = tf.train.exponential_decay(EmbeddingConfig.lr, self.global_step, 50, 0.96)
        optimizer = tf.train.AdamOptimizer(learning_rate=EmbeddingConfig.lr, beta1=EmbeddingConfig.beta1, beta2=EmbeddingConfig.beta2)
        # optimizer = tf.train.MomentumOptimizer(learning_rate=EmbeddingConfig.lr, momentum=0.9)
        # optimizer = tf.train.RMSPropOptimizer(learning_rate=EmbeddingConfig.lr, epsilon=1e-6)
        grads = optimizer.compute_gradients(self.loss)
        grads = [(tf.clip_by_norm(grad, 100000), var) for grad, var in grads if grad is not None]
        # grads = [(grad + tf.random_normal(shape=grad.get_shape(), stddev=0.6), var) for grad, var in grads if grad is not None]
        # for grad, var in grads:
            # if grad is not None:
                # tf.summary.scalar('gradient_%s' % (var), tf.norm(grad))
                # tf.summary.histogram('gradient_%s' % (var), grad)
        self.optimizer = optimizer.apply_gradients(grads, global_step=self.global_step)


    def add_summary_op(self):
        # for v in tf.global_variables():
            # if len(v.get_shape().as_list()) != 0 and ('lstm' in v.name or 'gru' in v.name):
                # tf.summary.histogram(v.name, v)
        self.merged_summary_op = tf.summary.merge_all()


    def run_on_batch(self, train_inputs_batch, voice_spec_batch, song_spec_batch):
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        self.add_prediction_op(train_inputs_batch)
        self.add_loss_op(voice_spec_batch, song_spec_batch)
        self.add_training_op()
        self.add_summary_op()

    def __init__(self, stats, freq_weighted=None):
        self.stats = stats  # stats for normalizing input
