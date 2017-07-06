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

# def create_batch(input_data, target_data, batch_size):
#     input_batches = []
#     target_batches = []
    
#     for i in xrange(0, len(target_data), batch_size):
#         input_batches.append(input_data[i:i + batch_size])
#         target_batches.append(target_data[i:i + batch_size])
    
#     return input_batches, target_batches

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

    # stacked_song_spec = stack_spectrograms(song_spec)
    # stacked_voice_spec = stack_spectrograms(voice_spec)
    stacked_mixed_spec = stack_spectrograms(mixed_spec)  # this will be the input

    target_spec = tf.concat([song_spec, voice_spec], axis=1) # axis=2  # target spec is going to be a concatenation of song_spec and voice_spec

    return stacked_mixed_spec, target_spec

def transform_spec_from_raw(raw):
    '''
    Read raw features from TFRecords and shape them into spectrograms
    '''
    spec = tf.decode_raw(raw, tf.float32)
    spec.set_shape([Config.num_time_frames * Config.num_freq_bins])
    return tf.reshape(spec, [Config.num_time_frames, Config.num_freq_bins])

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

def model_train(freq_weighted):
    logs_path = "tensorboard/" + strftime("%Y_%m_%d_%H_%M_%S", gmtime())
    
    with tf.Graph().as_default():
        files = [os.path.join(TRAIN_DIR, f) for f in os.listdir(TRAIN_DIR) if f[-10:] == '.tfrecords']
        test_files = [os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR) if f[-10:] == '.tfrecords']
        with tf.name_scope('input'):
            filename_queue = tf.train.string_input_producer(files, num_epochs=Config.num_epochs)

            train_input, train_target = read_and_decode(filename_queue)

            train_inputs, train_targets = tf.train.batch(
                [train_input, train_target], batch_size=Config.batch_size, num_threads=2,
                capacity= 2 + 3 * Config.batch_size)
        
        model = SeparationModel(freq_weighted=False)  # don't use freq_weighted for now
        model.train_on_batch(train_inputs, train_targets)
        
        # init = tf.global_variables_initializer()
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

            # if args.load_from_file is not None:
            #     new_saver = tf.train.import_meta_graph('%s.meta' % args.load_from_file, clear_devices=True)
            #     new_saver.restore(session, args.load_from_file)

            train_writer = tf.summary.FileWriter(logs_path + '/train', session.graph)
            global_start = time.time()
            
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=session, coord=coord)
            total_train_cost = 0

            try:
                step_ii = 0
                while not coord.should_stop():
                    start = time.time()

                    curr_batch_size = train_inputs.get_shape().as_list()[0]
                    output, batch_cost, summary, optimizer = session.run([model.output, model.loss, model.merged_summary_op, model.optimizer])
                    
                    total_train_cost += batch_cost * curr_batch_size
                    train_writer.add_summary(summary, step_ii)

                    step_ii += 1
                    duration = time.time() - start

                    if step_ii % 1 == 0:
                        print('Step %d: loss = %.2f (%.3f sec)' % (step_ii, batch_cost, duration))

                    if (step_ii + 1) % 100 == 0:
                        checkpoint_name = 'checkpoints/%dlayer_%flr_model' % (Config.num_layers, Config.lr)
                        saver.save(session, checkpoint_name, global_step=step_ii + 1)

            except tf.errors.OutOfRangeError:
                print('Done Training for %d epochs, %d steps' % (Config.num_epochs, step_ii))
            finally:
                coord.request_stop()

            coord.join(threads)


            # for curr_epoch in range(Config.num_epochs):
            #     total_train_cost = 0
            #     total_train_examples = 0

            #     start = time.time()

            #     for batch in random.sample(range(num_batches_per_epoch), num_batches_per_epoch):
            #         cur_batch_size = len(train_target_batch[batch])
            #         total_train_examples += cur_batch_size

            #         _, batch_cost, summary = model.train_on_batch(session, 
            #                                                 train_input_batch[batch],
            #                                                 train_target_batch[batch], 
            #                                                 train=True)

            #         total_train_cost += batch_cost * cur_batch_size
            #         train_writer.add_summary(summary, step_ii)

            #         step_ii += 1

            #     train_cost = total_train_cost / total_train_examples

            #     num_dev_batches = len(dev_target_batch)
            #     total_batch_cost = 0
            #     total_batch_examples = 0

            #     # val_batch_cost, _ = model.train_on_batch(session, dev_input_batch[0], dev_target_batch[0], train=False)
            #     for batch in random.sample(range(num_dev_batches_per_epoch), num_dev_batches_per_epoch):
            #         cur_batch_size = len(dev_target_batch[batch])
            #         total_batch_examples += cur_batch_size

            #         _, _val_batch_cost, _ = model.train_on_batch(session, dev_input_batch[batch], dev_target_batch[batch], train=False)

            #         total_batch_cost += cur_batch_size * _val_batch_cost


            #     val_batch_cost = None
            #     try:
            #         val_batch_cost = total_batch_cost / total_batch_examples
            #     except ZeroDivisionError:
            #         val_batch_cost = 0

            #     log = "Epoch {}/{}, train_cost = {:.3f}, val_cost = {:.3f}, time = {:.3f}"
            #     print(
            #     log.format(curr_epoch + 1, Config.num_epochs, train_cost, val_batch_cost, time.time() - start))

            #     # if args.print_every is not None and (curr_epoch + 1) % args.print_every == 0:
            #     #     batch_ii = 0
            #     #     model.print_results(train_feature_minibatches[batch_ii], train_labels_minibatches[batch_ii])

            #     if (curr_epoch + 1) % 10 == 0:
            #         checkpoint_name = 'checkpoints/%dlayer_%flr_model' % (Config.num_layers, Config.lr)
            #         if freq_weighted:
            #             checkpoint_name = checkpoint_name + '_freq_weighted'
            #         saver.save(session, checkpoint_name, global_step=curr_epoch + 1)


def model_test(test_input):

    test_rate, test_audio = wavfile.read(test_input)
    clean_rate, clean_audio = wavfile.read(CLEAN_FILE)
    noise_rate, noise_audio = wavfile.read(NOISE_FILE)

    length = len(clean_audio)
    noise_audio = noise_audio[:length]

    clean_spec = stft.spectrogram(clean_audio)
    noise_spec = stft.spectrogram(noise_audio)
    test_spec = stft.spectrogram(test_audio)

    reverted_clean = stft.ispectrogram(clean_spec)
    reverted_noise = stft.ispectrogram(noise_spec)

    test_data = np.array([test_spec.transpose() / 100000])  # make data a batch of 1

    with tf.Graph().as_default():
        model = SeparationModel()
        saver = tf.train.Saver(tf.trainable_variables())
        
        with tf.Session() as session:
            ckpt = tf.train.get_checkpoint_state('checkpoints/')
            if ckpt:
                print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
                saver.restore(session, ckpt.model_checkpoint_path)
            else:
                print("Created model with fresh parameters.")
                session.run(tf.initialize_all_variables())

            test_data_shape = np.shape(test_data)
            dummy_target = np.zeros((test_data_shape[0], test_data_shape[1], 2 * test_data_shape[2]))

            output, _, _ = model.train_on_batch(session, test_data, dummy_target, train=False)

            num_freq_bin = output.shape[2] / 2
            clean_output = output[0,:,:num_freq_bin]
            noise_output = output[0,:,num_freq_bin:]

            clean_mask, noise_mask = create_mask(clean_output, noise_output)

            clean_spec = createSpectrogram(np.multiply(clean_mask.transpose(), test_spec), test_spec.stft_settings) 
            noise_spec = createSpectrogram(np.multiply(noise_mask.transpose(), test_spec), test_spec.stft_settings)

            clean_wav = stft.ispectrogram(clean_spec)
            noise_wav = stft.ispectrogram(noise_spec)

            sdr, sir, sar, _ = bss_eval_sources(np.array([reverted_clean, reverted_noise]), np.array([clean_wav, noise_wav]), False)
            print(sdr, sir, sar)

            writeWav('data/test_combined/output_clean.wav', 44100, clean_wav)
            writeWav('data/test_combined/output_noise.wav', 44100, noise_wav)

def model_batch_test():

    test_batch = h5py.File('%stest_batch_amplified' % (DIR))
    data = test_batch['data'].value

    with open('%stest_settings.pkl' % (DIR), 'rb') as f:
        settings = pickle.load(f)

    # print(settings[:2])
    
    combined, clean, noise = zip(data)
    combined = combined[0]
    clean = clean[0]
    noise = noise[0]
    target = np.concatenate((clean,noise), axis=2)

    # test_rate, test_audio = wavfile.read('data/test_combined/combined.wav')
    # test_spec = stft.spectrogram(test_audio)

    combined_batch, target_batch = create_batch(combined, target, 50)

    original_combined_batch = [copy.deepcopy(batch) for batch in combined_batch]

    with tf.Graph().as_default():
        model = SeparationModel()
        saver = tf.train.Saver(tf.trainable_variables())
        
        with tf.Session() as session:
            ckpt = tf.train.get_checkpoint_state('checkpoints/')
            if ckpt:
                print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
                saver.restore(session, ckpt.model_checkpoint_path)
            else:
                print("Created model with fresh parameters.")
                session.run(tf.initialize_all_variables())

            prev_mask_array = None
            diff = float('inf')
            iters = 0

            while True:
                curr_mask_array = []
                iters += 1
                output, _, _ = model.train_on_batch(session, combined_batch[0], target_batch[0], train=False)

                num_freq_bin = output.shape[2] / 2
                clean_outputs = output[:,:,:num_freq_bin]
                noise_outputs = output[:,:,num_freq_bin:]

                # clean = [target[:,:num_freq_bin] for target in target_batch]
                # noise = [target[:,num_freq_bin:] for target in target_batch]

                num_outputs = len(clean_outputs)

                results = []

                for i in xrange(num_outputs):
                    orig_clean_output = clean_outputs[i]
                    orig_noise_output = noise_outputs[i]

                    stft_settings = copy.deepcopy(settings[i])
                    orig_length = stft_settings['orig_length']
                    stft_settings.pop('orig_length', None)
                    clean_output = orig_clean_output[-orig_length:]
                    noise_output = orig_noise_output[-orig_length:]

                    clean_mask, noise_mask = create_mask(clean_output, noise_output)
                    orig_clean_mask, orig_noise_mask = create_mask(orig_clean_output, orig_noise_output)

                    curr_mask_array.append(clean_mask)
                    # if i == 0:
                        # print clean_mask[10:20,10:20]
                    curr_mask_array.append(noise_mask)

                    clean_spec = createSpectrogram(np.multiply(clean_mask.transpose(), original_combined_batch[0][i][-orig_length:].transpose()), settings[i])
                    noise_spec = createSpectrogram(np.multiply(noise_mask.transpose(), original_combined_batch[0][i][-orig_length:].transpose()), settings[i])

                    # print '-' * 20
                    # print original_combined_batch[0][i]
                    # print '=' * 20
                    combined_batch[0][i] += np.multiply(orig_clean_mask, original_combined_batch[0][i]) * 0.1
                    # print combined_batch[0][i]
                    # print '=' * 20
                    # print original_combined_batch[0][i]
                    # print '-' * 20

                    estimated_clean_wav = stft.ispectrogram(clean_spec)
                    estimated_noise_wav = stft.ispectrogram(noise_spec)

                    reference_clean_wav = stft.ispectrogram(SpectrogramArray(clean[i][-orig_length:], stft_settings).transpose())
                    reference_noise_wav = stft.ispectrogram(SpectrogramArray(noise[i][-orig_length:], stft_settings).transpose())

                    try:
                        sdr, sir, sar, _ = bss_eval_sources(np.array([reference_clean_wav, reference_noise_wav]), np.array([estimated_clean_wav, estimated_noise_wav]), False)
                        results.append((sdr[0], sdr[1], sir[0], sir[1], sar[0], sar[1]))
                        # print('%f, %f, %f, %f, %f, %f' % (sdr[0], sdr[1], sir[0], sir[1], sar[0], sar[1]))
                    except ValueError:
                        print('error')
                        continue
                # break
                
                diff = 1
                if prev_mask_array is not None:
                    # print curr_mask_array[0]
                    # print prev_mask_array[0]
                    diff = sum(np.sum(np.abs(curr_mask_array[i] - prev_mask_array[i])) for i in xrange(len(prev_mask_array)))
                    print('Changes after iteration %d: %d' % (iters, diff))

                sdr_cleans, sdr_noises, sir_cleans, sir_noises, sar_cleans, sar_noises = zip(*results)
                print('Avg sdr_cleans: %f, sdr_noises: %f, sir_cleans: %f, sir_noises: %f, sar_cleans: %f, sar_noises: %f' % (np.mean(sdr_cleans), np.mean(sdr_noises), np.mean(sir_cleans), np.mean(sir_noises), np.mean(sar_cleans), np.mean(sar_noises)))
                with open('data/results/masking_iteration_amplified' + '.csv', 'a+') as f:
                    f.write('%d,%f,%f,%f,%f,%f,%f\n' % (diff,np.mean(sdr_cleans), np.mean(sdr_noises), np.mean(sir_cleans), np.mean(sir_noises), np.mean(sar_cleans), np.mean(sar_noises)))

                prev_mask_array = [np.copy(mask) for mask in curr_mask_array]

                # if diff == 0:
                #     break

            results_filename = '%sresults_%d_%f' % ('data/results/', Config.num_layers, Config.lr)
            # results_filename += 'freq_weighted'

            with open(results_filename + '.csv', 'w+') as f:
                for sdr_1, sdr_2, sir_1, sir_2, sar_1, sar_2 in results:
                    f.write('%f,%f,%f,%f,%f,%f\n' % (sdr_1, sdr_2, sir_1, sir_2, sar_1, sar_2))

            # f = h5py.File(results_filename, 'w')
            # f.create_dataset('result', data=results, compression="gzip", compression_opts=9)


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
        model_test(args.test_single_input)

