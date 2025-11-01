import numpy as np

class CostFunction():
	def __init__(self, name, function, derivative):
		self.name = name
		self.function = function
		self.derivative = derivative
	
	def apply(self, y, a):
		return self.function(y, a)
	
	def apply_derivative(self, y, a):
		"""
		derivative wrt a
		"""
		return self.derivative(y, a)

###########################################################################

def least_squares(y, a):
	return 0.5 * np.sum((y-a) ** 2)

def least_squares_derivative(y, a):
	return (a - y)

LeastSquares = CostFunction("Least Squares", least_squares, least_squares_derivative)

###########################################################################

def least_squares_mean(y, a):
	return 1 / len(y[0]) * 0.5 * np.sum((y-a) ** 2)

def least_squares_mean_derivative(y, a):
	return 1 / len(y[0]) *(a - y)

LeastSquaresMean = CostFunction("Least Squares Mean", least_squares_mean, least_squares_mean_derivative)

###########################################################################

def cross_entropy(y, a):
	return - np.sum(y * np.log(a + 1e-12))

def cross_entropy_derivative(y, a):
	return - y / (a + 1e-12)

CrossEntropy = CostFunction("Cross Entropy", cross_entropy, cross_entropy_derivative)