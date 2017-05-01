# coding: utf-8
# Source from https://github.com/oreilly-japan/deep-learning-from-scratch
import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.optimizer import *

class Train:
	def __init__(self, model, x_train, t_train, x_test, t_test,
	 epochs=20, mini_batch_size=100, optimizer='SGD',
	 optimizer_param={'lr':0.01},
	 evaluate_sample_num_per_epoch=None, verbose=True):
		self.model = model
		self.verbose = verbose
		self.x_train = x_train
		self.x_test = x_test
		self.t_train = t_train
		self.t_test = t_test
		self.epochs = epochs
		self.batch_size = mini_batch_size
		self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch

		#optimizer:
		optimizer_class_dict = {'sgd': SGD, 'momentum': Momentum, 'nesterov': Nesterov,
			'adagrad': AdaGrad, 'rmsprpo': RMSprop, 'adam': Adam}
		self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)

		#initiate train size and iterate number
		self.train_size = x_train.shape[0]
		self.iter_per_epoch = max(self.train_size/ mini_batch_size, 1)
		self.max_iter = int(epochs * self.iter_per_epoch)
		self.current_iter = 0
		self.current_epoch = 0

		#initiate training result list for graph
		self.train_loss_list = []
		self.train_accuracy_list = []
		self.test_accuracy_list = []

	def train_one_step(self):
		# use only batch_size number of data for each step
		
		batch_mask = np.random.choice(self.train_size, self.batch_size)
		x_batch = self.x_train[batch_mask]
		t_batch = self.t_train[batch_mask]

		grads = self.model.gradient(x_batch, t_batch)
		self.optimizer.update(self.model.weights, grads)

		loss = self.model.loss(x_batch, t_batch)
		self.train_loss_list.append(loss)
		if self.verbose: print("train loss: " + str(loss))

		if self.current_iter % self.iter_per_epoch == 0:
			self.current_epoch += 1

			x_train_sample, t_train_sample = self.x_train, self.t_train
			x_test_sample, t_test_sample = self.x_test, self.t_test
			if not self.evaluate_sample_num_per_epoch is None:
				t = self.evaluate_sample_num_per_epoch
				x_train_sample, t_train_sample = self.x_train[:t], self.t_train[:t]
				x_test_sample, t_test_sample = self.x_test[:t], self.t_test[:t]

			train_accuracy = self.model.accuracy(x_train_sample, t_train_sample)
			test_accuracy = self.model.accuracy(x_test_sample, t_test_sample)
			self.train_accuracy_list.append(train_accuracy)
			self.test_accuracy_list.append(test_accuracy)

			if self.verbose: print("=== epoch:" + str(self.current_epoch) + ", train accuracy: " + str(train_accuracy) + ", test accuracy: " + str(test_accuracy) + "===")
		self.current_iter += 1
	def train(self):
		for i in range(self.max_iter):
			self.train_one_step()
		test_accuracy = self.model.accuracy(self.x_test, self.t_test)
		if self.verbose:
			print("====================== Final Test Accuracy ======================")
			print("test accuracy: " + str(test_accuracy))