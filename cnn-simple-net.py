# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import pickle
import numpy as numpy
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient

class SimpleCnn:
	""" Simple Conv net
	conv - relu - pool - affine - relu - affine - softmax

	Hyper-params:
	=============
	input_size: size of imput = image height * width
	hidden_size_list: hidden layers number of unit (e.g [100, 100, 100])
	output_size: size of output. MNIST: 10
	activation: 'relu' or 'sigmoid'
	weight_init_std: standard deviation for weight initialize (e.g: 0.01)
	"""
	def __init__(self, input_dim=(1, 28, 28), conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},
				hidden_size=100, ouput_size=10, weight_init_std=0.01):
		filter_num = conv_param['filter_num']
		filter_size = conv_param['filter_size']
		filter_pad = conv_param['pad']
		filter_stride = conv_param['stride']
		input_size = input_dim[1]
		conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
		pool_ouput_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))

		#initialize weights
		self.params = {}
		self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
		self.params['b1'] = np.zeros(filter_num)

		self.params['W2'] = weight_init_std * np.random.randn(pool_ouput_size, hidden_size)
		self.params['b2'] = np.zeros(hidden_size)

		self.params['W2'] = weight_init_std * np.random.randn(hidden_size, ouput_size)
		self.params['b2'] = np.zeros(output_size)

		#generate layers
		
