import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict

class Model:
	def  __init__(self, input_size, hidden_size_list, output_size, weight_init_std=0.01, activation='Relu', weight_decay_lambda=0,
                 use_weight_init_std=True, use_dropout = False, dropout_ratio = 0.5, use_batchnorm=False):
		self.weights = {}
		self.input_size = input_size
		self.output_size = output_size
		self.hidden_size_list = hidden_size_list
		self.hidden_layer_num = len(hidden_size_list)
		self.use_weight_init_std = use_weight_init_std
		self.weight_init_std = weight_init_std
		self.activation_func = activation
		self.use_batchnorm = use_batchnorm
		self.use_dropout = use_dropout
		self.weight_decay_lambda = weight_decay_lambda

        #Initialize all weight:
		self.__init_weight(weight_init_std)
        #Create layers based on the hidden_size_list
		activation_functions = {"sigmoid": Sigmoid, "Relu": Relu}
		self.layers = OrderedDict()
		for i in range(1, self.hidden_layer_num + 1):
            #Create Affine layer transpose(W) * X + b
			self.layers["Affine" + str(i)] = Affine(self.weights["W" + str(i)], self.weights["b" + str(i)])

            #Create batch normalization layer for each hidden layer
			if self.use_batchnorm:
				self.weights["gamma" + str(i)] = np.ones(hidden_size_list[i - 1])
				self.weights["beta" + str(i)] = np.zeros(hidden_size_list[i - 1])
				self.layers["BatchNorm" + str(i)] = BatchNormalization(self.weights["gamma" + str(i)], self.weights["beta" + str(i)])

            #Activation
			self.layers["Activation_function" + str(i)] = activation_functions[activation]()

			if self.use_dropout:
				self.layers["Dropout" + str(i)] = Dropout(dropout_ratio)
		idx = self.hidden_layer_num + 1
		self.layers["Affine" + str(idx)] = Affine(self.weights["W" + str(idx)], self.weights["b" + str(idx)])

		self.last_layer = SoftmaxWithLoss()
	def __init_weight(self, weight_init_std):
		"""Initialize weights
		Parameters:
			weight_init_std: standard deviation of gaussian distribution for weight N(0, weight_init_std^2) default is 0.01
				if use standard corresponding to activation function then
					activation : Relu => std = sqrt(2/Number of hidden unit)
					activation : Sigmoid, Tanh => std = sqrt(1/Number of hidden unit)
			If batch_norm is used then weight init is not so much neccesary
		"""
		all_layer_size = [self.input_size] + self.hidden_size_list + [self.output_size]
		for i in range(1, len(all_layer_size)):
			scale = weight_init_std
			if not self.use_weight_init_std:
				if str(activation).lower() in ("relu"):
					scale = np.sqrt(2.0 / all_layer_size[i - 1])
				elif str(activation).lower() in ("sigmoid", "tanh"):
					scale = np.sqrt(1.0 / all_layer_size[i - 1])
			self.weights["W" + str(i)] = scale * np.random.randn(all_layer_size[i - 1], all_layer_size[i])
			self.weights["b" + str(i)] = np.zeros(all_layer_size[i])
	
	def predict(self, x, train_flg=False):
		for key, layer in self.layers.items():
			if "Dropout" in key or "BatchNorm" in key:
				x = layer.forward(x, train_flg)
			else:
				x = layer.forward(x)

		return x
    
	def loss(self, x, t, train_flg=False):
		"""
		Calculate loss by feed forward until the layer before softmax layer
		At softmax layer, the forward function return the loss by cross entropy loss function
		Weight decay is calculated through all layers
		"""
		y = self.predict(x, train_flg)

		weight_decay = 0
		for idx in range(1, self.hidden_layer_num + 2):
			W = self.weights['W' + str(idx)]
			weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)

		return self.last_layer.forward(y, t) + weight_decay

	def accuracy(self, X, T):
		"""
		Feed forward toward the last layer
		apply argmax to predict output
		compare output with real label to get accuracy
 		"""
		Y = self.predict(X, train_flg=False)
		Y = np.argmax(Y, axis=1)
		if T.ndim != 1 : T = np.argmax(T, axis=1)

		accuracy = np.sum(Y == T) / float(X.shape[0])
		return accuracy

	def gradient(self, x, t):
		"""
		Calculate the gradient of each weight through all layers
 		First: feed forward to get the final predict y
		At softmax layer: gradient = (y - t)/ batch_size
		At other layers: dW at affine layer is transpose(x).dot(dout)
		weights is update
		"""
    	#Forward to all layers
		self.loss(x, t, train_flg=True)

    	#Error back propagation
		dout = 1
		dout = self.last_layer.backward(dout)

		layers = list(self.layers.values())
		layers.reverse() # for backprop
		for layer in layers:
			dout = layer.backward(dout)

		grads = {}
    	#calculate gradients
		for i in range(1, self.hidden_layer_num + 2):
			grads["W" + str(i)] = self.layers["Affine" + str(i)].dW + self.weight_decay_lambda * self.weights["W" + str(i)]
			grads["b" + str(i)] = self.layers["Affine" + str(i)].db
			if self.use_batchnorm and i != self.hidden_layer_num + 1:
				grads["gamma" + str(i)] = self.layers["BatchNorm" + str(i)].dgamma
				grads["beta" + str(i)] = self.layers["beta" + str(i)].dbeta
		return grads