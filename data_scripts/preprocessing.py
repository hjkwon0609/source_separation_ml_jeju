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
TRAIN_FILES = set(['45381_chorus.wav','45391_verse.wav','66556_verse.wav','54219_verse.wav','54211_chorus.wav','54243_verse.wav','54186_verse.wav','54242_verse.wav','31112_chorus.wav','45406_chorus.wav','21069_verse.wav','31134_verse.wav','31126_chorus.wav','45389_chorus.wav','31099_verse.wav','66566_verse.wav','31100_chorus.wav','31116_chorus.wav','21064_verse.wav','21061_verse.wav','45435_chorus.wav','21068_verse.wav','45412_chorus.wav','10161_chorus.wav','45393_verse.wav','45387_verse.wav','45390_verse.wav','31143_chorus.wav','45435_verse.wav','21071_verse.wav','45421_verse.wav','21032_verse.wav','21073_chorus.wav','21062_verse.wav','21061_chorus.wav','71719_chorus.wav','45398_verse.wav','31109_chorus.wav','66559_chorus.wav','21031_verse.wav','31092_chorus.wav','45382_verse.wav','54236_chorus.wav','31092_verse.wav','66560_verse.wav','90586_verse.wav','54246_chorus.wav','45378_verse.wav','21038_chorus.wav','21038_verse.wav','31083_chorus.wav','54205_verse.wav','54243_chorus.wav','45422_chorus.wav','21075_chorus.wav','31083_verse.wav','54227_verse.wav','31081_verse.wav','45382_chorus.wav','31139_verse.wav','45423_verse.wav','31075_chorus.wav','21075_verse.wav','21060_verse.wav','61671_chorus.wav','45418_verse.wav','45381_verse.wav','45429_verse.wav','31143_verse.wav'])

def _bytes_feature(value): return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

if __name__ == '__main__':
	train_spectrograms = []
	j = 0

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

			if f in TRAIN_FILES:
				output_dir = TRAIN_DIR
				train_spectrograms.append([np.abs(song_spec), np.abs(voice_spec), np.abs(mixed_spec)])
				print 'Train'
			else:
				output_dir = TEST_DIR
				print 'Test'

			song_spec, voice_spec, mixed_spec = spectrogram_split_real_imag(song_spec), \
												spectrogram_split_real_imag(voice_spec), \
												spectrogram_split_real_imag(mixed_spec)

			writer_filename = os.path.join(output_dir, '%s.tfrecords' % f[:-4])
			writer = tf.python_io.TFRecordWriter(writer_filename)

			example = tf.train.Example(features=tf.train.Features(feature={
				'song_spec': _bytes_feature(song_spec.tostring()),
				'voice_spec': _bytes_feature(voice_spec.tostring()),
				'mixed_spec': _bytes_feature(mixed_spec.tostring())
				}))

			writer.write(example.SerializeToString())
			writer.close()

	train_spectrograms = np.asarray(train_spectrograms)
	print(train_spectrograms.shape)
	means = np.mean(train_spectrograms, axis=(0,2))
	std = np.std(train_spectrograms, axis=(0,2))
	print(means.shape)
	print(std.shape)
	np.save(os.path.join(PREPROCESSING_STAT_DIR, 'stats'), [means, std])
