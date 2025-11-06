import numpy as np
import initializations
import activation_functions as af
import cost_functions as cf

class Network():
	def __init__(self, sizes: list[int], 
			  activation_functions: list[af.ActivationFunction], 
			  cost_function: cf.CostFunction = cf.LeastSquaresMean,
			  learning_rate: float = 0.01,
			  debug = False,
			  weights_initialization = initializations.he,
		):
		"""
		Initializing a network with pre-determined number of layers and sizes of the layers\n
		Layer 0 is the input layer with size set to self.input_size\n
		Layer -1 is the output layer with size set to self.output_size
		"""
		self.debug = debug

		if not len(sizes) - len(activation_functions) == 1:
			raise ValueError("Number of layers should be just 1 more than number of activations")
		self.num_layers = len(sizes)
		self.num_hidden_layers = len(sizes) - 1

		self.activation_functions = activation_functions
		self.cost_function = cost_function

		self.learning_rate = learning_rate

		self.sizes = sizes
		self.input_size = sizes[0]
		self.output_size = sizes[-1]
		
		# weights[0] converts x to z_1. So size of w_i will be len(z_1) x len(z_0). x is z_0
		# weights[i] converts a_(i-1) to z_i. So size of w_i will be len(z_i) x len(a_(i-1))
		self.weights = [weights_initialization(sizes[i], sizes[i - 1]) for i in range(1, self.num_layers)]

		# bias[0] is the bias from x to z_1. size will be len(z_1)
		# bias[i] is the bias from a_(i-1) to z_i. size will be len(z_i)
		self.biases = [np.zeros((sizes[i], 1)) for i in range(1, self.num_layers)]
	
	def forward_propagation(self, x, round_number):
		"""
		x has to have the shape self.input_size x N, where N is the number of data points.\n
		Outputs the network output of size self.output_size x N
		"""
		if not len(x) == self.input_size:
			raise ValueError(f"Cannot take a vector of size {len(x)}. Expected size: {self.input_size}")
		
		z_values = [x]
		activations = [x]
		for i in range(self.num_hidden_layers):
			w_i = self.weights[i]
			b_i = self.biases[i]
			func = self.activation_functions[i]
			a_prev = activations[-1]
			z_i = np.matmul(w_i, a_prev) + b_i
			a_i = func.apply(z_i)
			z_values.append(z_i)
			activations.append(a_i)
		return z_values, activations
	
	def get_predictions(self, x):
		_, activations =  self.forward_propagation(x, 0)
		return activations[-1]
	
	def get_accuracy(self, x, y):
		predictions = self.get_predictions(x)
		num_points = len(x[0])
		correct = 0
		for i in range(num_points):
			prediction_i = predictions[:, i]
			actual_i = y[:, i]
			prediction = np.argmax(prediction_i)
			actual = np.argmax(actual_i)
			if prediction == actual:
				correct += 1
		return correct / num_points
	
	def one_round_one_point(self, x, y, round_number, total_rounds):
		if not len(x[0]) == 1 and not len(y[0]) == 1:
			raise ValueError("This trains only one point at a time")
		z_values, activations = self.forward_propagation(x, round_number)

		deltas = [None] * self.num_hidden_layers
		## Calculate delta_L
		delta_from_cost_function = self.cost_function.apply_derivative(y, activations[-1])
		delta_from_activation = self.activation_functions[-1].apply_jacobian(z_values[-1]).T
		deltas[-1] = np.matmul(delta_from_activation, delta_from_cost_function)

		## Calculate all deltas
		for l in range(2,self.num_layers):
			delta_l_plus_1 = deltas[-l + 1]
			w_l_plus_1 = self.weights[-l + 1]
			delta_from_l_plus_1 = np.matmul(w_l_plus_1.T, delta_l_plus_1)
			delta_from_activation = self.activation_functions[-l].apply_jacobian(z_values[-l])
			deltas[-l] = np.matmul(delta_from_activation, delta_from_l_plus_1)
		
		delta_biases = [np.sum(delta_i, axis=1, keepdims=True) for delta_i in deltas]
		delta_weights = [None] * self.num_hidden_layers
		for l in range(1, self.num_layers):
			delta_l = deltas[-l]
			activations_l_1 = activations[-l - 1].T
			delta_weights[-l] = np.matmul(delta_l, activations_l_1)
		
		for i in range(self.num_hidden_layers):
			self.weights[i] -= self.learning_rate * delta_weights[i]
			self.biases[i] -= self.learning_rate * delta_biases[i]

	#### this does not work
	def one_round(self, x, y, round_number, total_rounds):
		z_values, activations = self.forward_propagation(x, round_number)

		deltas = [None] * self.num_hidden_layers

		## Calculate delta_L
		delta_L = np.zeros_like(y)
		for i in range(y.shape[1]):
			y_curr = y[:, [i]]
			z_curr = z_values[-1][:, [i]]
			a_curr = activations[-1][:, [i]]
			delta_from_cost_function = self.cost_function.apply_derivative(y_curr, a_curr)
			delta_from_activation = self.activation_functions[-1].apply_jacobian(z_curr).T
			delta_L[:, [i]] = np.matmul(delta_from_activation, delta_from_cost_function)
		deltas[-1] = delta_L
		## Calculate all deltas
		for l in range(2,self.num_layers):
			delta_l_plus_1 = deltas[-l + 1]
			w_l_plus_1 = self.weights[-l + 1]
			delta_from_l_plus_1 = np.matmul(w_l_plus_1.T, delta_l_plus_1)
			delta_from_activation = self.activation_functions[-l].apply_jacobian(z_values[-l])
			deltas[-l] = np.matmul(delta_from_activation, delta_from_l_plus_1)
		
		delta_biases = [np.sum(delta_i, axis=1, keepdims=True) for delta_i in deltas]
		delta_weights = [None] * self.num_hidden_layers
		for l in range(1, self.num_layers):
			delta_l = deltas[-l]
			activations_l_1 = activations[-l - 1].T
			delta_weights[-l] = np.matmul(delta_l, activations_l_1)
		
		for i in range(self.num_hidden_layers):
			self.weights[i] -= self.learning_rate * delta_weights[i]
			self.biases[i] -= self.learning_rate * delta_biases[i]

	def learn(self, x, y, epochs = 10000):
		for i in range(epochs):
			random_idx = np.random.randint(1, len(x[0]))
			self.one_round_one_point(x[:, [random_idx]], y[:, [random_idx]], i, epochs)
			if i % (epochs // 100) == 0:
				self.learning_rate *= 0.9
			if i % (epochs // 10) == 0 and self.debug:
				print(f"Accuracy after {i} rounds: {self.get_accuracy(x, y)}")
				print("\n")
		print(f"Accuracy on training set: {self.get_accuracy(x, y)}")
	
	def validate(self, x_validation, y_validation):
		return self.get_accuracy(x_validation, y_validation)