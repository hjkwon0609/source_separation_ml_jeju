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

from embedding_config import EmbeddingConfig
from embedding_model import SeparationModel
import h5py

from evaluate import bss_eval_sources
import pickle

import copy

from util import *

from tensorflow.contrib.tensorboard.plugins import projector

TRAIN_DIR = 'data/train'
TEST_DIR = 'data/test'

model_name = 'embedding_lr%f_layer%d' % (EmbeddingConfig.lr, EmbeddingConfig.num_layers)

def read_and_decode(filename_queue):
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

    stacked_mixed_spec = stack_spectrograms(mixed_spec)

    return stacked_mixed_spec, voice_spec, song_spec

def transform_spec_from_raw(raw):
    '''
    Read raw features from TFRecords and shape them into spectrograms
    '''
    spec = tf.decode_raw(raw, tf.float32)
    spec.set_shape([EmbeddingConfig.num_time_frames * EmbeddingConfig.num_freq_bins * 2])
    spec = tf.reshape(spec, [-1, EmbeddingConfig.num_freq_bins * 2])
    real, imag = tf.split(spec, [EmbeddingConfig.num_freq_bins, EmbeddingConfig.num_freq_bins], axis=1)
    orig_spec = tf.complex(real, imag)
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
    data_dir = TRAIN_DIR if train else TEST_DIR
    files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f[-10:] == '.tfrecords'])
    files = files[:4]

    with tf.name_scope('input'):
        num_epochs = EmbeddingConfig.num_epochs if train else 1
        batch_size = EmbeddingConfig.batch_size if train else 1

        filename_queue = tf.train.string_input_producer(files, num_epochs=num_epochs)

        input_spec, voice_spec, song_spec = read_and_decode(filename_queue)

        input_specs, voice_specs, song_specs = tf.train.shuffle_batch(
        [input_spec, voice_spec, song_spec], batch_size=batch_size, num_threads=2, enqueue_many=True,
            capacity=EmbeddingConfig.file_reader_capacity, min_after_dequeue=EmbeddingConfig.file_reader_min_after_dequeue)

    return input_specs, voice_specs, song_specs

def model_train(freq_weighted):
    logs_path = "tensorboard/" + strftime("%Y_%m_%d_%H_%M_%S", gmtime()) + model_name + '/'
     
    with tf.Graph().as_default():
        
        train_inputs, voice_specs, song_specs = prepare_data(True)

        model = SeparationModel(freq_weighted=False)  # don't use freq_weighted for now
        model.run_on_batch(train_inputs, voice_specs, song_specs)
        
        embedding_var = tf.get_variable('embedding', trainable=False, shape=[EmbeddingConfig.num_freq_bins, EmbeddingConfig.embedding_dim])
        init = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
        saver = tf.train.Saver()

        print('train input shape: %s' % (train_inputs.get_shape()))

        with tf.Session() as session:
            ckpt = tf.train.get_checkpoint_state(logs_path)
            
            if ckpt:
                print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
                session.run(tf.initialize_local_variables())
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

                    if step_ii % 1 == 0:
                        print('Step %d: loss = %.5f (%.3f sec)' % (step_ii, batch_cost, duration))

                    if step_ii %10 == 0:
                        checkpoint_name = logs_path + 'checkpoint'
                        saver.save(session, checkpoint_name, global_step=model.global_step)

            except tf.errors.OutOfRangeError:
                print('Done Training for %d epochs, %d steps' % (EmbeddingConfig.num_epochs, step_ii))
            finally:
                coord.request_stop()

            coord.join(threads)


def model_test():

    with tf.Graph().as_default():
        train_inputs, voice_spec, song_spec, train_targets = prepare_data(False)
            
        model = SeparationModel(freq_weighted=False)  # don't use freq_weighted for now

        model.run_on_batch(train_inputs, train_targets)
        print(train_inputs.get_shape())
        
        init = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
        saver = tf.train.Saver()

        with tf.Session() as session:
            ckpt = tf.train.get_checkpoint_state('checkpoints/')
            if ckpt:
                print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
                saver.restore(session, ckpt.model_checkpoint_path)
                session.run(tf.initialize_local_variables())
            else:
                session.run(init)
            global_start = time.time()
            
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=session, coord=coord)

            hard_masked_results = []
            soft_masked_results = []

            print('num trainable parameters: %s' % (np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))

            try:
                step_ii = 0
                while not coord.should_stop():
                    start = time.time()

                    hard_masked_output, soft_masked_output, batch_cost, summary, target, mixed_spec = session.run([model.hard_masked_output, 
                                                                            model.soft_masked_output,
                                                                            model.loss,
                                                                            model.merged_summary_op,
                                                                            model.target,
                                                                            model.input])

                    step_ii += 1
                    duration = time.time() - start

                    print('Step %d: loss = %.5f (%.3f sec)' % (step_ii, batch_cost, duration))

                    hard_song_masked, hard_voice_masked = tf.split(hard_masked_output, [EmbeddingConfig.num_freq_bins, EmbeddingConfig.num_freq_bins], axis=1)
                    soft_song_masked, soft_voice_masked = tf.split(soft_masked_output, [EmbeddingConfig.num_freq_bins, EmbeddingConfig.num_freq_bins], axis=1)
                    song_target, voice_target = tf.split(target, [EmbeddingConfig.num_freq_bins, EmbeddingConfig.num_freq_bins], axis=1)

                    # original spectrum (reverting normalization)
                    mixed_mean, mixed_var, song_mean, song_var, voice_mean, voice_var = stats
                    
                    hard_song_masked = tf.complex(tf.real(hard_song_masked) * tf.sqrt(song_var) + song_mean, tf.imag(hard_song_masked))    
                    soft_song_masked = tf.complex(tf.real(soft_song_masked) * tf.sqrt(song_var) + song_mean, tf.imag(soft_song_masked))
                    song_target = tf.complex(tf.real(song_target) * tf.sqrt(song_var) + song_mean, tf.imag(song_target))

                    hard_voice_masked = tf.complex(tf.real(hard_voice_masked) * tf.sqrt(voice_var) + voice_mean, tf.imag(hard_voice_masked))
                    soft_voice_masked = tf.complex(tf.real(soft_voice_masked) * tf.sqrt(voice_var) + voice_mean, tf.imag(soft_voice_masked))
                    voice_target = tf.complex(tf.real(voice_target) * tf.sqrt(voice_var) + voice_mean, tf.imag(voice_target))                    

                    mixed_spec = mixed_spec[:,:,1]
                    # print('before shape: %s' % (mixed_spec.shape))
                    mixed_spec = mixed_spec.real * tf.sqrt(mixed_var) + mixed_mean + mixed_spec.imag
                    # print('after shape: %s' % (mixed_spec.shape))

                    result_wav_dir = 'data/results'

                    mixed_audio = create_audio_from_spectrogram(mixed_spec)
                    writeWav(os.path.join(result_wav_dir, 'mixed%d.wav' % (step_ii)), sample_rate, mixed_audio)

                    hard_song_masked_audio = create_audio_from_spectrogram(hard_song_masked)
                    hard_voice_masked_audio = create_audio_from_spectrogram(hard_voice_masked)

                    writeWav(os.path.join(result_wav_dir, 'hard_song_masked%d.wav' % (step_ii)), sample_rate, hard_song_masked_audio)
                    writeWav(os.path.join(result_wav_dir, 'hard_voice_masked%d.wav' % (step_ii)), sample_rate, hard_voice_masked_audio)

                    soft_song_masked_audio = create_audio_from_spectrogram(soft_song_masked)
                    soft_voice_masked_audio = create_audio_from_spectrogram(soft_voice_masked)

                    writeWav(os.path.join(result_wav_dir, 'soft_song_masked%d.wav' % (step_ii)), sample_rate, soft_song_masked_audio)
                    writeWav(os.path.join(result_wav_dir, 'soft_voice_masked%d.wav' % (step_ii)), sample_rate, soft_voice_masked_audio)

                    song_target_audio = create_audio_from_spectrogram(song_target)
                    voice_target_audio = create_audio_from_spectrogram(voice_target)

                    writeWav(os.path.join(result_wav_dir, 'song_target%d.wav' % (step_ii)), sample_rate, song_target_audio)
                    writeWav(os.path.join(result_wav_dir, 'voice_target%d.wav' % (step_ii)), sample_rate, voice_target_audio)

                    # hard_sdr, hard_sir, hard_sar, _ = bss_eval_sources(np.array([song_target_audio, voice_target_audio]), np.array([hard_song_masked_audio, hard_voice_masked_audio]), False)
                    # soft_sdr, soft_sir, soft_sar, _ = bss_eval_sources(np.array([song_target_audio, voice_target_audio]), np.array([soft_song_masked_audio, soft_voice_masked_audio]), False)
                    hard_gnsdr, hard_gsir, hard_gsar = bss_eval_global(mixed_audio, song_target_audio, voice_target_audio, hard_song_masked_audio, hard_voice_masked_audio)
                    soft_gnsdr, soft_gsir, soft_gsar = bss_eval_global(mixed_audio, song_target_audio, voice_target_audio, soft_song_masked_audio, soft_voice_masked_audio)

                    # out_results.append([hard_sdr[0], hard_sdr[1], hard_sir[0], hard_sir[1], hard_sar[0], hard_sar[1]])
                    # masked_results.append([soft_sdr[0], soft_sdr[1], soft_sir[0], soft_sir[1], soft_sar[0],soft_sar[1]])
                    hard_masked_results.append([hard_gnsdr[0], hard_gnsdr[1], hard_gsir[0], hard_gsir[1], hard_gsar[0], hard_gsar[1]])
                    soft_masked_results.append([soft_gnsdr[0], soft_gnsdr[1], soft_gsir[0], soft_gsir[1], soft_gsar[0], soft_gsar[1]])

            except tf.errors.OutOfRangeError:
                hard_masked_results = np.asarray(hard_masked_results)
                soft_masked_results = np.asarray(soft_masked_results)

                print(np.mean(hard_masked_results, axis=0))
                print(np.mean(soft_masked_results, axis=0))
            finally:
                coord.request_stop()

            coord.join(threads)


def writeWav(fn, fs, data):
    data = data * 1.5 / np.max(np.abs(data))
    wavfile.write(fn, fs, data)


def create_mask(clean_output, noise_output, hard=True):
    clean_mask = np.zeros(clean_output.shape)
    noise_mask = np.zeros(noise_output.shape)

    if hard:
        for i in range(len(clean_output)):
            for j in range(len(clean_output[0])):
                if abs(clean_output[i][j]) < abs(noise_output[i][j]):
                    noise_mask[i][j] = 1.0
                else:
                    clean_mask[i][j] = 1.0
    else:
        for i in range(len(clean_output)):
            for j in range(len(clean_output[0])):
                clean_mask[i][j] = abs(clean_output[i][j]) / (abs(clean_output[i][j]) + abs(noise_output[i][j]))
                noise_mask[i][j] = abs(noise_output[i][j]) / (abs(clean_output[i][j]) + abs(noise_output[i][j]))


    return clean_mask, noise_mask


def bss_eval_global(mixed_wav, src1_wav, src2_wav, pred_src1_wav, pred_src2_wav):
    len_cropped = pred_src1_wav.shape[-1]
    src1_wav = src1_wav[:len_cropped]
    src2_wav = src2_wav[:len_cropped]
    mixed_wav = mixed_wav[:len_cropped]
    gnsdr, gsir, gsar = np.zeros(2), np.zeros(2), np.zeros(2)
    total_len = 0
    # for i in range(2):
    sdr, sir, sar, _ = bss_eval_sources(np.array([src1_wav, src2_wav]),
                                        np.array([pred_src1_wav, pred_src2_wav]), False)
    sdr_mixed, _, _, _ = bss_eval_sources(np.array([src1_wav, src2_wav]),
                                          np.array([mixed_wav, mixed_wav]), False)
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

