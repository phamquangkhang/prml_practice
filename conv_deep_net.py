# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import pickle
import numpy as numpy
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient

class DeepCNN:
	"""
	Network architecture
		conv - relu - conv - relu - pool -
		conv - relu - conv - relu - pool -
		conv - relu - conv - relu - pool -
		affine - relu - dropout - affine - dropout - softmax
		Each pooling decrease the size of image to half
		By this, the size of each layer is for default setting is as below:
		Input: 1x28x28
		Conv1: 16x28x28
		Conv2: 16x28x28
		Pool1: 16x14x14
		Conv3: 32x14x14
		Conv4: 32x16x16 padding change to 2 so that calculate for size is easier
		Pool2: 32x8x8
		Conv5: 64x8x8
		Conv6: 64x8x8
		Pool3: 64x4x4
		FC: 50
	"""

	def __init__(self, input_dim=(1, 28, 28),
				conv_param_1= {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1},
				conv_param_2= {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1},
				conv_param_3= {'filter_num':32, 'filter_size':3, 'pad':1, 'stride':1},
				conv_param_4= {'filter_num':32, 'filter_size':3, 'pad':2, 'stride':1},
				conv_param_5= {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
				conv_param_6= {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
				fc_layer_size=50, output_size=10):
		#initialize weights:
		# calculate number of node num in each filter: number of channel * filter_size^2
		# the last filter before fully connected has size: number of channel * last_layer_size^2
		input_channel_no = input_dim[0]
		connection_list = ()

		pre_node_nums = np.array([1*3*3, 16*3*3, 16*3*3, 32*3*3, 32*3*3, 64*3*3, 64*4*4, fc_layer_size])
		weight_init_scales = np.sqrt(2.0 / pre_node_nums)

		self.weights = {}
		pre_channel_num = input_channel_no
		for idx, conv_param in enumerate([conv_param_1, conv_param_2, conv_param_3, conv_param4, conv_param_5, conv_param_6]):
			self.weights['W' + str(idx + 1)] = weight_init_scales[idx] * np.random.randn(conv_param['filter_num'], pre_channel_num, conv_param['filter_size'], conv_param['filter_size'])
			self.weights['b' + str(idx + 1)] = np.zeros(conv_param['filter_num'])
			pre_channel_num = conv_param['filter_num']
		self.weights['W7'] = weight_init_scales[6] * np.random.randn(64*4*4, fc_layer_size)
		self.weights['b7'] = np.zeros(fc_layer_size)
		self.weights['W8'] = weight_init_scales[7] * np.random.randn(fc_layer_size, output_size)
		self.weights['b8'] = np.zeros(output_size)

		# Initiate layers
		self.layers = []
		self.layers.append(Convolution(self.weights['W1'], self.weights['b1'],
							conv_param_1['stride'], conv_param_1['pad']))
		self.layers.append(Relu())
		self.layers.append(Convolution(self.weights['W2'], self.weights['b2'],
							conv_param_2['stride'], conv_param_2['pad']))
		self.layers.append(Relu())
		self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
		self.layers.append(Convolution(self.weights['W3'], self.weights['b3'],
							conv_param_3['stride'], conv_param_3['pad']))
		self.layers.append(Relu())
		self.layers.append(Convolution(self.weights['W4'], self.weights['b4'],
							conv_param_4['stride'], conv_param_4['pad']))
		self.layers.append(Relu())
		self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
		self.layers.append(Convolution(self.weights['W5'], self.weights['b5'],
							conv_param_5['stride'], conv_param_5['pad']))
		self.layers.append(Relu())
		self.layers.append(Convolution(self.weights['W6'], self.weights['b6'],
							conv_param_6['stride'], conv_param_6['pad']))
		self.layers.append(Relu())
		self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
		self.layers.append(Affine(self.weights['W7'], self.weights['b7']))
		self.layers.append(Relu())
		self.layers.append(Dropout(0.5))
		self.layers.append(Affine(self.weights['W8'], self.weights['b8']))
		self.layers.append(Dropout(0.5))

		self.last_layer = SoftmaxWithLoss()
	def predict(self, x, train_flg=False):
		for layer in self.layers:
			if isinstance(layer, Dropout):
				x = layer.forward(x, train_flg)
			else:
				x = layer.forward(x)
		return x
	def loss(self, x, t):
		y = self.predict(x, train_flg=True)
		return self.last_layer.forward(y, t)
	def accuracy(self, x, t, batch_size=100):
		if t.ndim != 1 : t = np.argmax(t, axis=1)

		acc = 0.0

		for i in range(int(x.shape[0] / batch_size)):
			tx = x[i*batch_size:(i+1)*batch_size]
			tt = t[i*batch_size:(i+1)*batch_size]
			y = self.predict(tx, train_flg=False)
			y = np.argmax(y, axis=1)
			acc += np.sum(y == tt)
		return ac / x.shape[0]
	def gradient(self, x, t):
		#forward
		self.loss(x, t)

		#backward
		dout = 1
		dout = self.last_layer.backward(dout)

		tmp_layers = self.layers.copy()
		tmp_layers.reverse()
		for layer in tmp_layers:
			dout = layer.backward(dout)

		#setting
		grads = {}
		for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 18)):
			grads['W' + str(i + 1)] = self.layers[layer_idx].dW
			grads['b' + str(i + 1)] = self.layers[layer_idx].db
		return grads
	def save_params(self, file_name="params.pkl"):
		params = {}
		for key, val in self.weights.items():
			params[key] = val
		with open(file_name, 'wb') as f:
			pickle.dump(params, f)

	def load_params(self, file_name="params.pkl"):
		with open(file_name, 'rb') as f:
			params = pickle.load(f)
		for key, val in params.items():
			self.weights[key] = val

		for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 18)):
			self.layers[layer_idx].W = self.weights['W' + str(i+1)]
			self.layers[layer_idx].b = self.weights['b' + str(i+1)]
	def number_of_connection_conv_layer(conv_param):
		return conv_param.filter_num * conv_param.filter_size * conv_param.filter_size