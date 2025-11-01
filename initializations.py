import numpy as np

def he(n_out, n_in):
	return np.random.randn(n_out, n_in) * np.sqrt(2.0 / n_in)

def zeros(n_out, n_in):
	return np.zeros((n_out, n_in))

def standard_guassian(n_out, n_in):
	return np.random.randn(n_out, n_in)