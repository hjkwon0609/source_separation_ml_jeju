from scipy.io import wavfile
import os
import numpy as np
# import stft
import h5py
import sys
import tensorflow as tf
import librosa

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from util import *

'''
Script to try out splitting music on random music
'''

DATA_DIR = '../data'
INPUT_DIR = os.path.join(DATA_DIR, 'experiment_data')
OUT_DIR = os.path.join(DATA_DIR, 'experiment_test')

def _bytes_feature(value): return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

if __name__ == '__main__':
	train_spectrograms = []
	j = 0

	for i, f in enumerate(os.listdir(INPUT_DIR)):
		if f[-4:] == '.wav':
			print(f)
			filename = os.path.join(INPUT_DIR, f)
			try:
				data, rate = librosa.load(filename, mono=False, sr=44100)
				mono_data = data[0] / 2 + data[1] / 2
			except:
				mono_data, rate = librosa.load(filename, sr=44100)

			mono_data = librosa.resample(mono_data, rate, 16000)

			mixed_spec = create_spectrogram_from_audio(mono_data)
			print(mixed_spec.shape)

			mixed_spec = spectrogram_split_real_imag(mixed_spec)

			writer_filename = os.path.join(OUT_DIR, '%s.tfrecords' % f[:-4])
			writer = tf.python_io.TFRecordWriter(writer_filename)

			dummy_song_spec = np.zeros_like(mixed_spec)
			dummy_voice_spec = np.zeros_like(mixed_spec)

			example = tf.train.Example(features=tf.train.Features(feature={
				'song_spec': _bytes_feature(dummy_song_spec.tostring()),
				'voice_spec': _bytes_feature(dummy_voice_spec.tostring()),
				'mixed_spec': _bytes_feature(mixed_spec.tostring())
				}))

			writer.write(example.SerializeToString())
			writer.close()
