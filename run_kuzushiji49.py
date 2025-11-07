## vv does not work

import numpy as np
import initializations
import activation_functions as af
import cost_functions as cf
from train_and_test import train_and_test

train_images = np.load("data/kuzushiji-49/train_imgs.npz")["arr_0"]
train_labels = np.load("data/kuzushiji-49/train_labels.npz")["arr_0"]

x_train = train_images.reshape(train_images.shape[0], -1).T
y_train = train_labels.reshape(-1, 1)

test_images = np.load("data/kuzushiji-49/test_imgs.npz")["arr_0"]
test_labels = np.load("data/kuzushiji-49/test_labels.npz")["arr_0"]

x_test = test_images.reshape(test_images.shape[0], -1).T
y_test = test_labels.reshape(-1, 1)

train_and_test(
	x_train = x_train,
	y_train = y_train,
	x_test = x_test,
	y_test = y_test,
	layer_sizes = [784, 256, 128, 49],
	activations = [af.LeakyRelU, af.LeakyRelU, af.Softmax],
	cost_function = cf.CrossEntropy,
	weights_initialization= initializations.he,
	batch_size=5,	
	epochs = 2,
	debug=True,
)