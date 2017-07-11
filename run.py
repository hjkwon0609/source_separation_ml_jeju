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
from stft.types import SpectrogramArray
import stft

from evaluate import bss_eval_sources
import pickle

import copy

TRAIN_DIR = 'data/train'
TEST_DIR = 'data/test'
PREPROCESSING_STAT_DIR = 'data/preprocessing_stat'

def read_and_decode(filename_queue, stats):
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

    # normalize data with training preprocessing mean, var
    mixed_mean, mixed_var, song_mean, song_var, voice_mean, voice_var = stats
    
    # normalize magnitude of spectrums
    normalized_mixed_spec = tf.complex(tf.real(mixed_spec - mixed_mean) / tf.sqrt(mixed_var), tf.imag(mixed_spec))
    normalized_song_spec = tf.complex(tf.real(song_spec - song_mean) / tf.sqrt(song_var), tf.imag(song_spec))
    normalized_voice_spec = tf.complex(tf.real(voice_spec - voice_mean) / tf.sqrt(voice_var), tf.imag(voice_spec))

    # stacked_song_spec = stack_spectrograms(song_spec)
    # stacked_voice_spec = stack_spectrograms(voice_spec)
    stacked_mixed_spec = stack_spectrograms(normalized_mixed_spec)  # this will be the input

    target_spec = tf.concat([normalized_song_spec, normalized_voice_spec], axis=1) # axis=2  # target spec is going to be a concatenation of song_spec and voice_spec

    # return mixed_spec, target_spec
    return stacked_mixed_spec, target_spec

def transform_spec_from_raw(raw):
    '''
    Read raw features from TFRecords and shape them into spectrograms
    '''
    spec = tf.decode_raw(raw, tf.float32)
    spec.set_shape([Config.num_time_frames * Config.num_freq_bins * 2])
    spec = tf.reshape(spec, [Config.num_time_frames, Config.num_freq_bins * 2])
    real, imag = tf.split(spec, [Config.num_freq_bins, Config.num_freq_bins], axis=1)
    orig_spec = tf.complex(real, imag)
    return orig_spec  # just use magnitude (ignore phase) for training mask

def stack_spectrograms(spec):
    '''
    Stack spectrograms so that each element in the spectrogram now has 3 elements (prev_frame, curr_frame, next_frame)
    For the first(last) frame, prev_frame(next_frame) is zero vector
    '''
    stacked = tf.stack([tf.pad(spec,[[0, 2], [0, 0]], "CONSTANT"),
        tf.pad(spec,[[1, 1], [0, 0]], "CONSTANT"),
        tf.pad(spec,[[0, 2], [0, 0]], "CONSTANT")], axis=2) # axis=0
    return tf.slice(stacked, [1, 0, 0], [Config.num_time_frames, -1, -1]) # get rid of padding
    # return tf.slice(stacked, [0, 1, 0], [-1, Config.num_time_frames, -1]) # get rid of padding

def prepare_data(train):
    # data_dir = TRAIN_DIR if train else TEST_DIR
    data_dir = TRAIN_DIR
    # files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f[-10:] == '.tfrecords']
    filename = ['10161_chorus.tfrecords', '10161_verse.tfrecords', '10164_chorus.tfrecords', '10170_chorus.tfrecords']
    files = [os.path.join(data_dir, f) for f in filename]
    # if not train:
    #     files = files[:5]

    # normalize data
    stats = np.load(os.path.join(PREPROCESSING_STAT_DIR, 'stats.npy'))

    with tf.name_scope('input'):
        num_epochs = Config.num_epochs if train else 1
        batch_size = Config.batch_size if train else 1

        filename_queue = tf.train.string_input_producer(files, num_epochs=num_epochs)

        input_spec, target_spec = read_and_decode(filename_queue, stats)

        input_specs, target_specs = tf.train.batch(
        [input_spec, target_spec], batch_size=batch_size, num_threads=2,
        capacity= 2 + 3 * batch_size)

    # reshape so each frame becomes one example, instead of each song being an example
    # input: [num_batch, num_frames, freq_bins, 3] ==> [num_batch * num_frames, freq_bins, 3]
    # target: [num_batch, num_frames, freq_bins] ==> [num_batch * num_frames, freq_bins]
    input_shape = input_specs.get_shape().as_list()
    input_specs = tf.reshape(input_specs, [-1, input_shape[2], input_shape[3]])
    # train_inputs = tf.reshape(train_inputs, [-1, input_shape[2]])
    target_shape = target_specs.get_shape().as_list()
    target_specs = tf.reshape(target_specs, [-1, target_shape[2]])

    return input_specs, target_specs, stats  # return stats for loss calculation later

def model_train(freq_weighted):
    logs_path = "tensorboard/" + strftime("%Y_%m_%d_%H_%M_%S", gmtime()) + 'small_vpn_lr%f' % (Config.lr) 
    
    with tf.Graph().as_default():
        
        train_inputs, train_targets, stats = prepare_data(True)
        model = SeparationModel(freq_weighted=False)  # don't use freq_weighted for now
        model.run_on_batch(train_inputs, train_targets)
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        
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
            # total_train_cost = 0

            print('num trainable parameters: %s' % (np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))

            try:
                
                while not coord.should_stop():
                    start = time.time()
                    global_step += 1
                    step_ii = global_step.eval()
                    
                    output, batch_cost, summary, optimizer = session.run([model.output, model.loss, model.merged_summary_op, model.optimizer])
                    
                    # total_train_cost += batch_cost * curr_batch_size
                    train_writer.add_summary(summary, step_ii)

                    

                    duration = time.time() - start

                    if step_ii % 1 == 0:
                        print('Step %d: loss = %.5f (%.3f sec)' % (step_ii, batch_cost, duration))

                    if (step_ii + 1) % 100 == 0:
                        checkpoint_name = 'checkpoints/small_vpnn%dlayer_%flr_model' % (Config.num_layers, Config.lr)
                        saver.save(session, checkpoint_name, global_step=step_ii + 1)

            except tf.errors.OutOfRangeError:
                print('Done Training for %d epochs, %d steps' % (Config.num_epochs, step_ii))
            finally:
                coord.request_stop()

            coord.join(threads)


def model_test():

    sample_rate, sample_audio = wavfile.read('data/raw_data/10161_chorus.wav')
    sample_spectrogram = create_spectrogram_from_audio(sample_audio)  # saves spectrogram setting for ispectrogram later

    with tf.Graph().as_default():
        train_inputs, train_targets, stats = prepare_data(False)
            
        model = SeparationModel(freq_weighted=False)  # don't use freq_weighted for now

        model.run_on_batch(train_inputs, train_targets)
        
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

            out_results = []
            masked_results = []

            print('num trainable parameters: %s' % (np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))

            try:
                step_ii = 0
                while not coord.should_stop():
                    start = time.time()

                    output, masked_output, batch_cost, summary, optimizer = session.run([model.output, 
                                                                            model.masked_output,
                                                                            model.loss,
                                                                            model.merged_summary_op, 
                                                                            model.optimizer])

                    step_ii += 1
                    duration = time.time() - start

                    print('Step %d: loss = %.5f (%.3f sec)' % (step_ii, batch_cost, duration))

                    song_out, voice_out = tf.split(output, [Config.num_freq_bins, Config.num_freq_bins], axis=1)
                    song_masked, voice_masked = tf.split(masked_output, [Config.num_freq_bins, Config.num_freq_bins], axis=1)
                    song_target, voice_target = tf.split(train_targets, [Config.num_freq_bins, Config.num_freq_bins], axis=1)

                    result_wav_dir = 'data/results'

                    song_out_audio = create_audio_from_spectrogram(song_out)
                    voice_out_audio = create_audio_from_spectrogram(voice_out)

                    writeWav(os.path.join(result_wav_dir, 'song_out%d.wav' % (step_ii)), sample_rate, song_out_audio)
                    writeWav(os.path.join(result_wav_dir, 'voice_out%d.wav' % (step_ii)), sample_rate, voice_out_audio)

                    song_masked_audio = create_audio_from_spectrogram(song_masked)
                    voice_masked_audio = create_audio_from_spectrogram(voice_masked)

                    writeWav(os.path.join(result_wav_dir, 'song_masked%d.wav' % (step_ii)), sample_rate, song_out_audio)
                    writeWav(os.path.join(result_wav_dir, 'voice_masked%d.wav' % (step_ii)), sample_rate, voice_out_audio)

                    song_target_audio = create_audio_from_spectrogram(song_target)
                    voice_target_audio = create_audio_from_spectrogram(voice_target)

                    writeWav(os.path.join(result_wav_dir, 'song_target%d.wav' % (step_ii)), sample_rate, song_target_audio)
                    writeWav(os.path.join(result_wav_dir, 'voice_target%d.wav' % (step_ii)), sample_rate, voice_target_audio)

                    out_sdr, out_sir, out_sar, _ = bss_eval_sources(np.array([song_target_audio, voice_target_audio]), np.array([song_out_audio, voice_out_audio]), False)
                    masked_sdr, masked_sir, masked_sar, _ = bss_eval_sources(np.array([song_target_audio, voice_target_audio]), np.array([song_out_audio, voice_out_audio]), False)

                    out_results.append([out_sdr[0], out_sdr[1], out_sir[0], out_sir[1], out_sar[0], out_sar[1]])
                    masked_results.append([masked_sdr[0], masked_sdr[1], masked_sir[0], masked_sir[1], masked_sar[0], masked_sar[1]])

            except tf.errors.OutOfRangeError:
                out_results = np.asarray(out_results)
                masked_results = np.asarray(masked_results)

                print(np.mean(out_results, axis=0))
                print(np.mean(masked_results, axis=0))
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

def createSpectrogram(arr, settings):
    x = SpectrogramArray(arr, stft_settings={
                                'framelength': settings['framelength'],
                                'hopsize': settings['hopsize'],
                                'overlap': settings['overlap'],
                                'centered': settings['centered'],
                                'window': settings['window'],
                                'halved': settings['halved'],
                                'transform': settings['transform'],
                                'padding': settings['padding'],
                                'outlength': settings['outlength'],
                                }
                        )
    return x

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

