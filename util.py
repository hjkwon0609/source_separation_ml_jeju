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
	concatenated = np.concatenate([np.real(spectrogram), np.imag(spectrogram)], axis=1)
	print concatenated.shape
	return concatenated

def create_audio_from_spectrogram(spec):
	# reconstruct complex number from real and imaginary components
	spec_transposed = np.transpose(spec.eval())
	return stft.istft(spec_transposed, hop_length)

############################################################################
##  Vector Product Functions
############################################################################

# def tf_broadcast_matrix_mult(a, b):
# 	'''
# 	a is 3-d and b is 2-d
# 	'''
# 	orig_shape = a.get_shape().as_list()
# 	a_ = tf.reshape(a, [-1, orig_shape[-1]])
# 	mul = tf.matmul(a_, b)
# 	new_shape = orig_shape[:-1] + b.get_shape().as_list()[-1:]
# 	return tf.reshape(mul, new_shape)

def vector_product_matrix(X, W):
	return tf.transpose([tf.matmul(X[:,:,1], W[:,:,2]) - tf.matmul(X[:,:,2], W[:,:,1]),
		tf.matmul(X[:,:,2], W[:,:,0]) - tf.matmul(X[:,:,0], W[:,:,2]),
		tf.matmul(X[:,:,0], W[:,:,1]) - tf.matmul(X[:,:,1], W[:,:,0])], (1, 2, 0))