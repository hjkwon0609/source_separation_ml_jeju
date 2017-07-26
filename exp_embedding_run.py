import time
import argparse
import math
import random
import os
import sys
import distutils.util
# uncomment this line to suppress Tensorflow warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from six.moves import xrange as range
from scipy.io import wavfile

import pdb
from time import gmtime, strftime

from exp_embedding_config import EmbeddingConfig
from embedding_model import SeparationModel
import h5py

from scipy.cluster.vq import whiten
from sklearn.cluster import KMeans

from evaluate import bss_eval_sources
import pickle
import librosa

import copy

from util import *

from tensorflow.contrib.tensorboard.plugins import projector

TRAIN_DIR = 'data/experiment_test'
TEST_DIR = 'data/test'
PREPROCESSING_STATS = 'data/preprocessing_stat/stats.npy'

EMBEDDING_RESULT_DIR = 'data/clustering_experiment/'

sequence_model = 'vpnn' if EmbeddingConfig.use_vpnn else 'gru' if EmbeddingConfig.use_gru else 'lstm'
model_name = sequence_model + '_embedding_lr%f_layer%d_hidden_unit%dkeep_prob%fembedding_dim%d' % (EmbeddingConfig.lr, EmbeddingConfig.num_layers, EmbeddingConfig.num_hidden, EmbeddingConfig.keep_prob, EmbeddingConfig.embedding_dim)

def read_and_decode(filename_queue, train):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example, features={
        'song_spec': tf.FixedLenFeature([], tf.string),
        'voice_spec': tf.FixedLenFeature([], tf.string),
        'mixed_spec': tf.FixedLenFeature([], tf.string)
        })

    song_spec = transform_spec_from_raw(features['song_spec'])
    voice_spec = transform_spec_from_raw(features['voice_spec'])
    mixed_spec = transform_spec_from_raw(features['mixed_spec'])

    if EmbeddingConfig.use_vpnn:
        stacked_mixed_spec = stack_spectrograms(mixed_spec)
        return stacked_mixed_spec, voice_spec, song_spec
    else:
        # create shorter segments for lstm
        # song_specs = tf.stack(tf.split(song_spec, num_or_size_splits=EmbeddingConfig.num_segments, axis=0))
        # voice_specs = tf.stack(tf.split(voice_spec, num_or_size_splits=EmbeddingConfig.num_segments, axis=0))
        # mixed_specs = tf.stack(tf.split(mixed_spec, num_or_size_splits=EmbeddingConfig.num_segments, axis=0))

        return mixed_spec, voice_spec, song_spec

def transform_spec_from_raw(raw):
    '''
    Read raw features from TFRecords and shape them into spectrograms
    '''
    spec = tf.decode_raw(raw, tf.float32)
    spec.set_shape([EmbeddingConfig.num_time_frames * EmbeddingConfig.num_freq_bins * 2])
    spec = tf.reshape(spec, [-1, EmbeddingConfig.num_freq_bins * 2])
    real, imag = tf.split(spec, [EmbeddingConfig.num_freq_bins, EmbeddingConfig.num_freq_bins], axis=1)
    orig_spec = tf.complex(real, imag)
    # orig_spec = librosa.feature.melspectrogram(S=orig_spec, n_mels=150)
    return orig_spec  # shape: [time_frames, num_freq_bins]

def stack_spectrograms(spec):
    '''
    Stack spectrograms so that each element in the spectrogram now has 3 elements (prev_frame, curr_frame, next_frame)
    For the first(last) frame, prev_frame(next_frame) is zero vector
    '''
    stacked = tf.stack([tf.pad(spec,[[0, 2], [0, 0]], "CONSTANT"),
        tf.pad(spec,[[1, 1], [0, 0]], "CONSTANT"),
        tf.pad(spec,[[0, 2], [0, 0]], "CONSTANT")], axis=2)
    padding_removed = stacked[1:-1,:,:]
    return padding_removed

def prepare_data(train):
    # data_dir = TRAIN_DIR if train else TEST_DIR
    data_dir = TRAIN_DIR
    files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f[-10:] == '.tfrecords'])

    with tf.name_scope('input'):
        num_epochs = EmbeddingConfig.num_epochs if train else 1
        batch_size = EmbeddingConfig.batch_size if train else 1

        # enqueue_many = EmbeddingConfig.use_vpnn  # a whole song is one example for lstm. Each frame is one for vpnn

        filename_queue = tf.train.string_input_producer(files, num_epochs=num_epochs)

        input_spec, voice_spec, song_spec = read_and_decode(filename_queue, train)

        if train:
            input_specs, voice_specs, song_specs = tf.train.shuffle_batch(
            [input_spec, voice_spec, song_spec], batch_size=batch_size, num_threads=2, enqueue_many=True,
                capacity=EmbeddingConfig.file_reader_capacity, min_after_dequeue=EmbeddingConfig.file_reader_min_after_dequeue)
        else:
            # for testing, read each song as whole
            input_specs, voice_specs, song_specs = tf.train.batch(
                [input_spec, voice_spec, song_spec], batch_size=batch_size, num_threads=1, enqueue_many=False,
                capacity=EmbeddingConfig.file_reader_capacity)

    return input_specs, voice_specs, song_specs

def model_train(freq_weighted):
    logs_path = "tensorboard/" + strftime("%Y_%m_%d_%H_%M_%S", gmtime()) + model_name + '/'
     
    with tf.Graph().as_default():
        
        train_inputs, voice_specs, song_specs = prepare_data(True)
        stats = np.load(PREPROCESSING_STATS)

        model = SeparationModel(freq_weighted=False, stats=stats)  # don't use freq_weighted for now
        model.run_on_batch(train_inputs, voice_specs, song_specs)
        
        embedding_var = tf.get_variable('embedding', trainable=False, shape=[EmbeddingConfig.num_freq_bins, EmbeddingConfig.embedding_dim])
        # init = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        saver = tf.train.Saver()

        print('train input shape: %s' % (train_inputs.get_shape()))

        with tf.Session() as session:
            ckpt = tf.train.get_checkpoint_state('checkpoints/')
            
            if ckpt:
                print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
                session.run(tf.local_variables_initializer())
                saver.restore(session, ckpt.model_checkpoint_path)
            else:
                session.run(init)

            train_writer = tf.summary.FileWriter(logs_path + 'train', session.graph)
            global_start = time.time()
            
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=session, coord=coord)

            print('num trainable parameters: %s' % (np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
            step_ii = 0

            try:
                while not coord.should_stop():
                    start = time.time()
                    step_ii += 1
                    
                    batch_cost, summary, optimizer, embedding = session.run([model.loss, 
                        model.merged_summary_op, model.optimizer, model.embedding])
                    
                    train_writer.add_summary(summary, step_ii)
                    duration = time.time() - start

                    embedding_var.assign(embedding[0])

                    if step_ii % 5 == 0:
                        print('Step %d: loss = %.5f (%.3f sec)' % (step_ii, batch_cost, duration))

                    if step_ii % 100 == 0:
                        checkpoint_name = logs_path + 'checkpoint'
                        saver.save(session, checkpoint_name, global_step=model.global_step)

            except tf.errors.OutOfRangeError:
                print('Done Training for %d epochs, %d steps' % (EmbeddingConfig.num_epochs, step_ii))
            finally:
                coord.request_stop()

            coord.join(threads)


def model_test():
    logs_path = "tensorboard/" + strftime("%Y_%m_%d_%H_%M_%S", gmtime()) + model_name + '/'
    with tf.Graph().as_default():
        train_inputs, voice_specs, song_specs = prepare_data(False)
        stats = np.load(PREPROCESSING_STATS)
            
        model = SeparationModel(freq_weighted=False, stats=stats)  # don't use freq_weighted for now

        model.run_on_batch(train_inputs, voice_specs, song_specs)
        
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        saver = tf.train.Saver()

        with tf.Session() as session:
            ckpt = tf.train.get_checkpoint_state('checkpoints/')
            if ckpt:
                print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
                session.run(tf.local_variables_initializer())
                saver.restore(session, ckpt.model_checkpoint_path)
            else:
                session.run(init)
            global_start = time.time()

            train_writer = tf.summary.FileWriter(logs_path + 'test', session.graph)
            
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=session, coord=coord)

            print('num trainable parameters: %s' % (np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))

            results = []

            try:
                step_ii = 0
                while not coord.should_stop():
                    start = time.time()

                    batch_cost, summary, embedding, voice_spec, song_spec, mixed_spec = session.run([model.loss, \
                        model.merged_summary_op, model.embedding, model.voice_spec, model.song_spec, model.input])

                    duration = time.time() - start
                    print('Step %d: loss = %.5f (%.3f sec)' % (step_ii, batch_cost, duration))

                    print(embedding.shape)
                    embedding = np.reshape(embedding, [-1, EmbeddingConfig.embedding_dim]).astype(np.float64)

                    k = 3

                    np.savez(os.path.join(EMBEDDING_RESULT_DIR, 'embedding%d' % (step_ii)), embedding)
                    np.savez(os.path.join(EMBEDDING_RESULT_DIR, 'mixed%d' % (step_ii)), mixed_spec)

                    whitened_embedding = whiten(embedding)
                    kmeans = KMeans(n_clusters=k).fit(whitened_embedding)

                    labels = np.reshape(kmeans.labels_, [-1, EmbeddingConfig.num_freq_bins])
                    print(labels.shape)
                    print(labels)

                    step_ii += 1
                    # duration = time.time() - start

                    # mixed_spec = tf.squeeze(mixed_spec)

                    # mixed_audio = create_audio_from_spectrogram(mixed_spec)
                    # song_target_audio = create_audio_from_spectrogram(song_spec)
                    # voice_target_audio = create_audio_from_spectrogram(voice_spec)

                    # result_wav_dir = 'data/results'

                    # writeWav(os.path.join(result_wav_dir, 'song_target_%d.wav' % (step_ii)), 16000, song_target_audio)
                    # writeWav(os.path.join(result_wav_dir, 'voice_target_%d.wav' % (step_ii)), 16000, voice_target_audio)                    

                    
                    # for i in xrange(k):
                    #     src_mask = np.equal(labels, i).astype(np.int8)
                    #     src_spec = apply_mask(mixed_spec, src_mask)

                    #     src_audio = create_audio_from_spectrogram(src_spec)
                    #     writeWav(os.path.join(result_wav_dir, 'src%d_%d.wav' % (i, step_ii)), 16000, src_audio)


                    # soft_gnsdr, soft_gsir, soft_gsar = bss_eval_global(mixed_audio, song_target_audio, voice_target_audio, src1_audio, src2_audio)
                    # results.append([soft_gnsdr[0], soft_gnsdr[1], soft_gsir[0], soft_gsir[1], soft_gsar[0], soft_gsar[1]])

            except tf.errors.OutOfRangeError:
                # results = np.asarray(results)
                # print(np.mean(results, axis=1))
                print('Done')
            finally:
                coord.request_stop()

            coord.join(threads)


def writeWav(fn, fs, data):
    data = data * 1.5 / np.max(np.abs(data))
    wavfile.write(fn, fs, data)


def bss_eval_global(mixed_wav, src1_wav, src2_wav, pred_src1_wav, pred_src2_wav):
    len_cropped = pred_src1_wav.shape[-1]
    src1_wav = src1_wav[:len_cropped]
    src2_wav = src2_wav[:len_cropped]
    mixed_wav = mixed_wav[:len_cropped]
    gnsdr, gsir, gsar = np.zeros(2), np.zeros(2), np.zeros(2)
    total_len = 0
    # for i in range(2):
    sdr, sir, sar, _ = bss_eval_sources(np.array([src1_wav, src2_wav]),
                                        np.array([pred_src1_wav, pred_src2_wav]), True)
    sdr_mixed, _, _, _ = bss_eval_sources(np.array([src1_wav, src2_wav]),
                                          np.array([mixed_wav, mixed_wav]), True)
    nsdr = sdr - sdr_mixed
    gnsdr += len_cropped * nsdr
    gsir += len_cropped * sir
    gsar += len_cropped * sar
    total_len += len_cropped
    gnsdr = gnsdr / total_len
    gsir = gsir / total_len
    gsar = gsar / total_len
    return gnsdr, gsir, gsar

if __name__ == "__main__":
    print(model_name)

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', nargs='?', default=True, type=distutils.util.strtobool)
    parser.add_argument('--test_single_input', nargs='?', default='data/test_combined/combined.wav', type=str)
    parser.add_argument('--freq_weighted', nargs='?', default=True, type=distutils.util.strtobool)
    parser.add_argument('--test_batch', nargs='?', default=False, type=distutils.util.strtobool)
    args = parser.parse_args()

    if args.test_batch:
        model_batch_test()
    elif args.train:
        model_train(args.freq_weighted)
    else:
        model_test()

