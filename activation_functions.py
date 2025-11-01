import numpy as np

class ActivationFunction():
	def __init__(self, name, function, derivative, jacobian):
		self.name = name
		self.function = function
		self.derivative = derivative
		self.jacobian = jacobian
	
	def apply(self, z):
		return self.function(z)
	
	def apply_derivative(self, z):
		return self.derivative(z)
	
	def apply_jacobian(self, z):
		return self.jacobian(z)

###########################################################################

def rel_u(z):
	return np.maximum(0, z)

def rel_u_derivative(z):
	return np.where(z > 0, 1, 0)

def rel_u_jacobian(z):
	return np.diagflat(rel_u_derivative(z))

RelU = ActivationFunction("RelU", rel_u, rel_u_derivative, rel_u_jacobian)

###########################################################################

LEAKY_REL_U_SLOPE = 0.01
def leaky_rel_u(z):
	return np.maximum(LEAKY_REL_U_SLOPE * z, z)

def leaky_rel_u_derivative(z):
	return np.where(z > 0, 1, LEAKY_REL_U_SLOPE)

def leaky_rel_u_jacobian(z):
	return np.diagflat(leaky_rel_u_derivative(z))

LeakyRelU = ActivationFunction("Leaky RelU", leaky_rel_u, leaky_rel_u_derivative, leaky_rel_u_jacobian)

###########################################################################

def identity(z):
    return z

def identity_derivative(z):
	return np.ones_like(z)

def identity_jacobian(z):
	return np.diagflat(identity_derivative(z))

Identity = ActivationFunction("Identity", identity, identity_derivative, identity_jacobian)

###########################################################################

def sigmoid(z):
    z = np.asarray(z)
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1 / (1 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1 + exp_z)
    return out

def sigmoid_derivative(z):
	s = sigmoid(z)
	return s*(1 - s)

def sigmoid_jacobian(z):
	return np.diagflat(sigmoid_derivative(z))

Sigmoid = ActivationFunction("Sigmoid", sigmoid, sigmoid_derivative, sigmoid_jacobian)

###########################################################################

def softmax(z):
    # subtract max(z) along the axis to stabilize
    z_max = np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(z - z_max)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def softmax_derivative(_):
	raise ValueError("Softmax does not have a derivative in vector form as the Jacobian contains non zero terms outside the diagonal")

def softmax_jacobian(z):
	s = softmax(z)
	return np.diagflat(s) - np.dot(s, s.T)

Softmax = ActivationFunction("Softmax", softmax, softmax_derivative, softmax_jacobian)
