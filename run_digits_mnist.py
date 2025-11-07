import initializations
import activation_functions as af
import cost_functions as cf
from train_and_test import read_train_test

read_train_test(
	train_filename="data/digits-mnist/train.csv", 
	test_filename="data/digits-mnist/test.csv",
	layer_sizes = [784, 16, 10],
	activations = [af.LeakyRelU, af.Softmax],
	cost_function = cf.CrossEntropy,
	weights_initialization= initializations.he,
	batch_size=10,
	epochs = 5,
	debug=True,
)