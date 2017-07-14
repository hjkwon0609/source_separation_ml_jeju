import time
import argparse
import math
import random
import os
import distutils.util
# uncomment this line to suppress Tensorflow warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from six.moves import xrange as range
from scipy.io import wavfile

from util import *
import pdb
from time import gmtime, strftime

from config import Config
from model import SeparationModel
import h5py

from evaluate import bss_eval_sources
import pickle

import copy

TRAIN_DIR = 'data/train'
TEST_DIR = 'data/test'
PREPROCESSING_STAT_DIR = 'data/preprocessing_stat'

model_name = 'vpnn_lr%f_masking_layer_layer%d_num_hidden%d_reg%f' % (Config.lr, Config.num_layers, Config.num_hidden, Config.l2_lambda)

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

    input_spec = stack_spectrograms(mixed_spec)  # this will be the input

    target_spec = tf.concat([song_spec, voice_spec], axis=1) # target spec is going to be a concatenation of song_spec and voice_spec

    # reshape so each frame becomes one example, instead of each song being an example
    # input: [freq_bins, freq_bins, 3] ==> [num_frames, freq_bins, 3]
    # input_shape = input_spec.get_shape().as_list()
    # input_spec = tf.reshape(input_spec, [-1, input_shape[1], input_shape[2]])

    return input_spec, target_spec


def transform_spec_from_raw(raw):
    '''
    Read raw features from TFRecords and shape them into spectrograms
    '''
    spec = tf.decode_raw(raw, tf.float32)
    spec.set_shape([Config.num_time_frames * Config.num_freq_bins * 2])
    spec = tf.reshape(spec, [-1, Config.num_freq_bins * 2])
    real, imag = tf.split(spec, [Config.num_freq_bins, Config.num_freq_bins], axis=1)
    orig_spec = tf.complex(real, imag)
    return orig_spec  # [num_time_frames, num_freq_bin]


def stack_spectrograms(spec):
    '''
    Stack spectrograms so that each element in the spectrogram now has 3 elements (prev_frame, curr_frame, next_frame)
    For the first(last) frame, prev_frame(next_frame) is zero vector
    '''
    stacked = tf.stack([tf.pad(spec,[[0, 2], [0, 0]], "CONSTANT"),
        tf.pad(spec,[[1, 1], [0, 0]], "CONSTANT"),
        tf.pad(spec,[[0, 2], [0, 0]], "CONSTANT")], axis=2) # axis=0    
    
    padding_removed = stacked[1:-1,:,:]
    return padding_removed


def prepare_data(train):
    # data_dir = TRAIN_DIR if train else TEST_DIR
    data_dir = TRAIN_DIR
    files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f[-10:] == '.tfrecords'])
    # filename = ['10161_chorus.tfrecords', '10161_verse.tfrecords', '10164_chorus.tfrecords', '10170_chorus.tfrecords']
    # files = [os.path.join(data_dir, f) for f in filename]
    if not train:
        files = files[:10]
        print files
    
    with tf.name_scope('input'):
        num_epochs = Config.num_epochs if train else 1
        batch_size = Config.batch_size if train else 1
        capacity = Config.file_reader_capacity if train else 5

        filename_queue = tf.train.string_input_producer(files, num_epochs=num_epochs)

        input_spec, target_spec = read_and_decode(filename_queue)

        if train:
            input_specs, target_specs = tf.train.shuffle_batch(
            [input_spec, target_spec], batch_size=batch_size, num_threads=2, enqueue_many=True,
            capacity=Config.file_reader_capacity, min_after_dequeue=Config.file_reader_min_after_dequeue)
        else:
            input_specs, target_specs = input_spec, target_spec

    return input_specs, target_specs 


def model_train(freq_weighted):
    logs_path = "tensorboard/" + strftime("%Y_%m_%d_%H_%M_%S", gmtime()) + model_name 
    
    with tf.Graph().as_default():
        
        train_inputs, train_targets = prepare_data(True)
        model = SeparationModel(freq_weighted=False)  # don't use freq_weighted for now
        model.run_on_batch(train_inputs, train_targets)
        
        init = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
        saver = tf.train.Saver()

        with tf.Session() as session:
            ckpt = tf.train.get_checkpoint_state('checkpoints/')
            
            if ckpt:
                print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
                session.run(tf.initialize_local_variables())
                saver.restore(session, ckpt.model_checkpoint_path)
            else:
                session.run(init)

            train_writer = tf.summary.FileWriter(logs_path + '/train', session.graph)
            global_start = time.time()
            
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=session, coord=coord)

            print('num trainable parameters: %s' % (np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
            step_ii = 0

            try:
                
                while not coord.should_stop():
                    start = time.time()
                    step_ii += 1
                    
                    output, batch_cost, masked_loss, summary, optimizer = session.run([model.output, model.loss, model.masked_loss, model.merged_summary_op, model.optimizer])
                    
                    # total_train_cost += batch_cost * curr_batch_size
                    train_writer.add_summary(summary, step_ii)

                    duration = time.time() - start

                    if step_ii % 10 == 0:
                        print('Step %d: loss = %.5f masked_loss = %.5f (%.3f sec)' % (step_ii, batch_cost, masked_loss, duration))

                    if step_ii % 500 == 0:
                        checkpoint_name = 'checkpoints/' + model_name
                        saver.save(session, checkpoint_name, global_step=model.global_step)

            except tf.errors.OutOfRangeError:
                print('Done Training for %d epochs, %d steps' % (Config.num_epochs, step_ii))
            finally:
                coord.request_stop()

            coord.join(threads)


def model_test():
    with tf.Graph().as_default():
        train_inputs, train_targets = prepare_data(False)
            
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

            soft_masked_results = []

            print('num trainable parameters: %s' % (np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))

            try:
                step_ii = 0
                while not coord.should_stop():
                    start = time.time()

                    soft_masked_output, batch_cost, masked_loss, summary, target, mixed_spec = session.run([model.soft_masked_output,
                                                                            model.loss,
                                                                            model.masked_loss,
                                                                            model.merged_summary_op,
                                                                            model.target,
                                                                            model.input])

                    step_ii += 1
                    duration = time.time() - start

                    print('Step %d: loss = %.5f masked_loss = %.5f (%.3f sec)' % (step_ii, batch_cost, masked_loss, duration))

                    soft_song_masked, soft_voice_masked = tf.split(soft_masked_output, [Config.num_freq_bins, Config.num_freq_bins], axis=1)
                    song_target, voice_target = tf.split(target, [Config.num_freq_bins, Config.num_freq_bins], axis=1)

                    mixed_spec = mixed_spec[:,:,1]

                    result_wav_dir = 'data/results'

                    mixed_audio = create_audio_from_spectrogram(mixed_spec)
                    writeWav(os.path.join(result_wav_dir, 'mixed%d.wav' % (step_ii)), Config.sample_rate, mixed_audio)

                    soft_song_masked_audio = create_audio_from_spectrogram(soft_song_masked)
                    soft_voice_masked_audio = create_audio_from_spectrogram(soft_voice_masked)

                    writeWav(os.path.join(result_wav_dir, 'soft_song_masked%d.wav' % (step_ii)), Config.sample_rate, soft_song_masked_audio)
                    writeWav(os.path.join(result_wav_dir, 'soft_voice_masked%d.wav' % (step_ii)), Config.sample_rate, soft_voice_masked_audio)

                    song_target_audio = create_audio_from_spectrogram(song_target)
                    voice_target_audio = create_audio_from_spectrogram(voice_target)

                    writeWav(os.path.join(result_wav_dir, 'song_target%d.wav' % (step_ii)), Config.sample_rate, song_target_audio)
                    writeWav(os.path.join(result_wav_dir, 'voice_target%d.wav' % (step_ii)), Config.sample_rate, voice_target_audio)

                    # soft_sdr, soft_sir, soft_sar, _ = bss_eval_sources(np.array([song_target_audio, voice_target_audio]), np.array([soft_song_masked_audio, soft_voice_masked_audio]), False)
                    soft_gnsdr, soft_gsir, soft_gsar = bss_eval_global(mixed_audio, song_target_audio, voice_target_audio, soft_song_masked_audio, soft_voice_masked_audio)

                    # masked_results.append([soft_sdr[0], soft_sdr[1], soft_sir[0], soft_sir[1], soft_sar[0],soft_sar[1]])
                    soft_masked_results.append([soft_gnsdr[0], soft_gnsdr[1], soft_gsir[0], soft_gsir[1], soft_gsar[0], soft_gsar[1]])

            except tf.errors.OutOfRangeError:
                soft_masked_results = np.asarray(soft_masked_results)
                print(np.mean(soft_masked_results, axis=0))
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

    print model_name
    if args.test_batch:
        model_batch_test()
    elif args.train:
        model_train(args.freq_weighted)
    else:
        model_test()

