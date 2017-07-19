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

        activation_fn = tf.nn.sigmoid
        for i in xrange(EmbeddingConfig.num_layers):
            layer_name = 'hidden%d' % (i + 1)
            curr = self.create_layer(layer_name, curr, EmbeddingConfig.num_hidden, activation_fn)

        curr_frame = curr[:,:,1]
        embedding_list = []
        for i in xrange(EmbeddingConfig.num_freq_bins):
            embedding = tf.layers.dense(curr_frame, EmbeddingConfig.embedding_dim, tf.nn.tanh)
            
            # unit norm
            embedding = tf.divide(embedding, tf.norm(embedding, axis=1, keep_dims=True))
            embedding_list.append(embedding)

        self.embedding = tf.transpose(tf.stack(embedding_list), [1, 0, 2])  # shape: [batch_size, frequency bins, embedding dim]  


    def add_loss_op(self, voice_spec, song_spec):
        song_spec_mask = tf.cast(tf.abs(song_spec) > tf.abs(voice_spec), tf.float32)
        voice_spec_mask =  tf.ones(song_spec_mask.get_shape()) - song_spec_mask

        V = self.embedding
        Y = tf.transpose([song_spec_mask, voice_spec_mask], [1, 2, 0])  # [num_batch, num_freq_bins, 2]

        # D = diag(Y Y^T 1)
        D = tf.matrix_diag(tf.squeeze(tf.matmul(tf.matmul(Y, tf.transpose(Y, [0, 2, 1])), tf.ones([EmbeddingConfig.batch_size, EmbeddingConfig.num_freq_bins, 1], tf.float32))))
        # D_inv_sqrt: D^(-1/2)
        D_inv_sqrt = tf.sqrt(tf.matrix_inverse(D))

        # cost = Frobenius|(V^T D^(-1/2) V)|^2 - 2 * Frobenius|(V^T D^(-1/2) Y)|^2 + Frobenius|(Y^T D^(-1/2) Y)|^2
        error = tf.reduce_sum(tf.square(tf.matmul(tf.matmul(tf.transpose(V, [0, 2, 1]), D_inv_sqrt), V))) + \
                    tf.reduce_sum(tf.square(tf.matmul(tf.matmul(tf.transpose(V, [0, 2, 1]), D_inv_sqrt), Y))) + \
                    tf.reduce_sum(tf.square(tf.matmul(tf.matmul(tf.transpose(V, [0, 2, 1]), D_inv_sqrt), Y)))

        average_error = error / V.get_shape().as_list()[0]

        l2_cost = tf.reduce_sum([tf.norm(v) for v in tf.trainable_variables() if len(v.get_shape().as_list()) == 3])

        self.loss = EmbeddingConfig.l2_lambda * l2_cost + average_error
        
        tf.summary.scalar("loss", self.loss)

    def add_training_op(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=EmbeddingConfig.lr, beta1=EmbeddingConfig.beta1, beta2=EmbeddingConfig.beta2)
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=EmbeddingConfig.lr)
        grads = optimizer.compute_gradients(self.loss)
        for grad, var in grads:
            tf.summary.histogram('gradient_norm_%s' % (var), grad)
        self.optimizer = optimizer.apply_gradients(grads, global_step=self.global_step)


    def add_summary_op(self):
        self.merged_summary_op = tf.summary.merge_all()


    def run_on_batch(self, train_inputs_batch, voice_spec_batch, song_spec_batch):
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        self.add_prediction_op(train_inputs_batch)
        self.add_loss_op(voice_spec_batch, song_spec_batch)
        self.add_training_op()
        self.add_summary_op()


    def print_results(self, train_inputs_batch, train_targets_batch):
        train_feed = self.create_feed_dict(train_inputs_batch, train_targets_batch)
        train_first_batch_preds = session.run(self.decoded_sequence, feed_dict=train_feed)
        compare_predicted_to_true(train_first_batch_preds, train_targets_batch)        

    def __init__(self, freq_weighted=None):
        pass
