import numpy as np
import tensorflow as tf
from config import Config
import librosa

############################################################################
##  Spectrogram util functions
############################################################################

def create_spectrogram_from_audio(data):
	global setting
	spectrogram = librosa.stft(data, n_fft=Config.n_fft, hop_length=Config.hop_length).transpose()

	# divide the real and imaginary components of each element 
	# concatenate the matrix with the real components and the matrix with imaginary components
	# (DataCorruptionError when saving complex numbers in TFRecords)
	concatenated = np.concatenate([np.real(spectrogram), np.imag(spectrogram)], axis=1)
	return concatenated

def create_audio_from_spectrogram(spec):
	spec_transposed = tf.transpose(spec).eval()
	print spec_transposed.shape
	return librosa.istft(spec_transposed, Config.hop_length)

############################################################################
##  Vector Product Functions
############################################################################

def vector_product_matrix(X, W):
	return tf.transpose([tf.matmul(X[:,:,1], W[:,:,2]) - tf.matmul(X[:,:,2], W[:,:,1]),
		tf.matmul(X[:,:,2], W[:,:,0]) - tf.matmul(X[:,:,0], W[:,:,2]),
		tf.matmul(X[:,:,0], W[:,:,1]) - tf.matmul(X[:,:,1], W[:,:,0])], (1, 2, 0))