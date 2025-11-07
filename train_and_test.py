import warnings
warnings.simplefilter("error", RuntimeWarning) # this is to catch overflows as errors in vs code debug mode

import pandas as pd
import numpy as np
from network import Network
import time

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
    Y = Y.ravel()
    one_hot_Y = pd.get_dummies(Y, dtype=np.float32).to_numpy().T
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

def read_train_test(train_filename, test_filename, layer_sizes, activations, cost_function, weights_initialization, batch_size = 1000, epochs = 10, debug = False):
    x_train, y_train = read_training_data(train_filename)
    x_test, y_test = read_test_data(test_filename)
    train_and_test(x_train, y_train, x_test, y_test, layer_sizes, activations, cost_function, weights_initialization, batch_size, epochs, debug)


def train_and_test(x_train, y_train, x_test, y_test, layer_sizes, activations, cost_function, weights_initialization, batch_size = 1000, epochs = 10, debug = False):
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)

    x_train_normalized = (x_train - mean) / (std + 1e-8)  # 1e-8 avoids divide-by-zero
    if debug:
        print("Creating one-hot matrix for y_train...")

    start_time = time.perf_counter()
    one_hot_y_train = one_hot(y_train)
    end_time = time.perf_counter()

    if debug:
        elapsed = end_time - start_time
        print(f"One-hot matrix of y_train created. shape: {one_hot_y_train.shape}")
        print(f"Time taken: {elapsed:.3f} seconds\n")

    mean = np.mean(x_train_normalized, axis=0)
    std = np.std(x_train_normalized, axis=0)

    x_train_set, y_train_set, x_validation_set, y_validation_set = build_training_and_validation_sets(x_train_normalized, one_hot_y_train)

    network = Network(layer_sizes, activations, cost_function, debug=debug, weights_initialization=weights_initialization)
    network.learn(x_train_set, y_train_set,batch_size=batch_size, epochs=epochs)

    validation_accuracy = network.validate(x_validation_set, y_validation_set)
    print(f"Accuracy on validation set {validation_accuracy}\n")

    one_hot_y_test = one_hot(y_test)
    if debug:
         print(f"One hot matrix of y_test created. shape: { one_hot_y_test.shape}\n")
    test_accuracy = network.validate(x_test, one_hot_y_test)
    print(f"Accuracy on test set {test_accuracy}\n")    