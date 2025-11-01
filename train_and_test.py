import warnings
warnings.simplefilter("error", RuntimeWarning) # this is to catch overflows as errors in vs code debug mode

import pandas as pd
import numpy as np
from network import Network
import initializations
import activation_functions as af
import cost_functions as cf

def read_training_data(filename):
	data_frame = pd.read_csv(filename, header=0) # header = 0 is the default value

	features = data_frame.iloc[:,1:]
	labels = data_frame.iloc[:,0]

	x_train = features.values
	y_train = labels.values

	return x_train.T, y_train

def read_test_data(filename):
	data_frame = pd.read_csv(filename, header=0)

	features = data_frame.iloc[:,1:]
	labels = data_frame.iloc[:,0]

	x_test = features.values
	y_test = labels.values

	return x_test.T, y_test

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def build_training_and_validation_sets(x, y, split_ratio = 0.8):
    num_samples = x.shape[1]

    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    x_shuffled = x[:, indices]
    y_shuffled = y[:, indices]
    
    split = int(split_ratio * num_samples)
    x_train, x_validation = x_shuffled[:, :split], x_shuffled[:, split:]
    y_train, y_validation = y_shuffled[:, :split], y_shuffled[:, split:]

    return x_train, y_train, x_validation, y_validation

def train_and_test(train_filename, test_filename, layer_sizes, activations, cost_function, weights_initialization, epochs):
    x_train, y_train = read_training_data(train_filename)

    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)

    x_train_normalized = (x_train - mean) / (std + 1e-8)  # 1e-8 avoids divide-by-zero
    one_hot_y_train = one_hot(y_train)

    mean = np.mean(x_train_normalized, axis=0)
    std = np.std(x_train_normalized, axis=0)

    x_train_set, y_train_set, x_validation_set, y_validation_set = build_training_and_validation_sets(x_train_normalized, one_hot_y_train)

    network = Network(layer_sizes, activations, cost_function, debug=False, weights_initialization=weights_initialization)
    network.learn(x_train_set, y_train_set, epochs=epochs)

    validation_accuracy = network.validate(x_validation_set, y_validation_set)
    print(f"Accuracy on validation set {validation_accuracy}")

    x_test, y_test = read_test_data(test_filename)
    test_accuracy = network.validate(x_test, one_hot(y_test))
    print(f"Accuracy on test set {test_accuracy}")