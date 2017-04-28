# coding: utf-8
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
	self.mini_batch_size = mini_batch_size
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
	
	batch_mask