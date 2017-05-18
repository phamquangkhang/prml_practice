# coding:  utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from mnist_load import *
from cnn_simple_net import SimpleCnn
from train import Train

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

max_epochs = 20

network = SimpleCnn(input_dim=(1,28,28), 
                    conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                    hidden_size=100, output_size=10, weight_init_std=0.01)

train = Train(network, x_train, t_train, x_test, t_test,
              epochs=max_epochs, mini_batch_size=100,
              optimizer='Adam', optimizer_param={'lr': 0.001},
              evaluate_sample_num_per_epoch=1000)

train.train()

# save weights
network.save_params("simple_convnet_params.pkl")
print("Saved Network Parameters")
