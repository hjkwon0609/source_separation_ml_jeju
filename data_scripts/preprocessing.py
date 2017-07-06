from scipy.io import wavfile
import os
import numpy as np
import stft
import h5py
import sys
import tensorflow as tf

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from util import *

DATA_DIR = '../data'
INPUT_DIR = os.path.join(DATA_DIR, 'raw_data')
OUTPUT_DIR = os.path.join(DATA_DIR, 'tf_records')

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

if __name__ == '__main__':

	for i, f in enumerate(os.listdir(INPUT_DIR)):
		if f[-4:] == '.wav':
			print i
			filename = os.path.join(INPUT_DIR, f)
			rate, data = wavfile.read(filename)
			
			length = data.shape[0]

			song = data[:,0]
			voice = data[:,1]

			mixed = song / 2 + voice / 2

			song_spec = create_spectrogram_from_audio(song)
			voice_spec = create_spectrogram_from_audio(voice)
			mixed_spec = create_spectrogram_from_audio(mixed)

			writer_filename = os.path.join(OUTPUT_DIR, '%s.tfrecords' % f[:-4])
			# writer = tf.python_io.TFRecordWriter(writer_filename, options=tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB))
			writer = tf.python_io.TFRecordWriter(writer_filename)

			example = tf.train.Example(features=tf.train.Features(feature={
				'song_spec': _bytes_feature(song_spec.tostring()),
				'voice_spec': _bytes_feature(voice_spec.tostring()),
				'mixed_spec': _bytes_feature(mixed_spec.tostring())
				}))

			writer.write(example.SerializeToString())
			writer.close()

	# print processed_data, procesed_data.shape



			
			




