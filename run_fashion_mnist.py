import initializations
import activation_functions as af
import cost_functions as cf
from train_and_test import train_and_test

train_and_test(
	train_filename="data/fashion-mnist/train.csv", 
	test_filename="data/fashion-mnist/test.csv",
	layer_sizes = [784, 16, 10],
	activations = [af.LeakyRelU, af.Softmax],
	cost_function = cf.CrossEntropy,
	weights_initialization= initializations.he,
	epochs = 200000,
)