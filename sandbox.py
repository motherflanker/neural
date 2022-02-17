import numpy as np
import nnfs

nnfs.init()

'''SOFTMAX ACTIVATION FUNCTION'''

'''exponentiation'''
layers_outputs = [[4.8, 1.21, 2.385],
                  [8.9, -1.81, 0.2],
                  [1.41, 1.051, 0.026]]


exp_values = np.exp(layers_outputs)
#print(np.sum(layers_outputs, axis=1, keepdims=True))

'''normalization'''
norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)

print(norm_values)
