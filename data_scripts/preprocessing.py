from scipy.io import wavfile
import os
import numpy as np
# import stft
import h5py
import sys
import tensorflow as tf
# import librosa

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from util import *

DATA_DIR = '../data'
INPUT_DIR = os.path.join(DATA_DIR, 'raw_data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')

PREPROCESSING_STAT_DIR = os.path.join(DATA_DIR, 'preprocessing_stat')
TRAINING_DATA = 200


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

if __name__ == '__main__':

	for i, f in enumerate(os.listdir(INPUT_DIR)):
		if f[-4:] == '.wav':
			print i
			filename = os.path.join(INPUT_DIR, f)
			data, rate = librosa.load(filename, mono=False)

			song = data[0,:]
			voice = data[1,:]

			song = librosa.resample(song, rate, 16000)
			voice = librosa.resample(voice, rate, 16000)

			mixed = song / 2 + voice / 2

			song_spec, voice_spec, mixed_spec = create_spectrogram_from_audio(song), \
												create_spectrogram_from_audio(voice), \
												create_spectrogram_from_audio(mixed)

			output_dir = None
			print song_spec.shape
			if i < TRAINING_DATA:
				output_dir = TRAIN_DIR
			else:
				output_dir = TEST_DIR

			writer_filename = os.path.join(output_dir, '%s.tfrecords' % f[:-4])
			writer = tf.python_io.TFRecordWriter(writer_filename)

			example = tf.train.Example(features=tf.train.Features(feature={
				'song_spec': _bytes_feature(song_spec.tostring()),
				'voice_spec': _bytes_feature(voice_spec.tostring()),
				'mixed_spec': _bytes_feature(mixed_spec.tostring())
				}))

			writer.write(example.SerializeToString())
			writer.close()

	# np.save(os.path.join(PREPROCESSING_STAT_DIR, 'per_freq_stats'),
	# 	[np.mean(mixed_specs,axis=0), np.var(mixed_specs,axis=0), np.mean(song_specs,axis=0), np.var(song_specs,axis=0), np.mean(voice_specs,axis=0), np.var(voice_specs,axis=0)])
	# np.save(os.path.join(PREPROCESSING_STAT_DIR, 'total_stats'),
	# 	[np.mean(mixed_specs,axis=0), np.var(mixed_specs,axis=0), np.mean(song_specs,axis=0), np.var(song_specs,axis=0), np.mean(voice_specs,axis=0), np.var(voice_specs,axis=0)])




			
			




