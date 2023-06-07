#!/usr/bin/env python3

import itertools
import numpy as np
from itertools import product
from itertools import permutations
import matplotlib.pyplot as plt

def convolution(I, W):
	rows, cols = I.shape[0]-W.shape[0]+1, I.shape[1]-W.shape[1]+1
	output = np.zeros((rows, cols))
	corresponding_weights = np.zeros((rows, cols, 2, 2))
	for i in range(rows):
		for j in range(cols):
			output[i, j] = np.sum(I[i:i+2, j:j+2]*W)
			corresponding_weights[i, j] = W*(I[i:i+2, j:j+2]==1)
	return output, corresponding_weights

def max_value_and_weights(I, W):
	conv, weights_array = convolution(I, W)
	max_value = np.max(conv)
	max_location = np.unravel_index(conv.argmax(), conv.shape)
	corresponding_weights = weights_array[max_location]
	corresponding_weights = corresponding_weights[corresponding_weights != 0]
	return max_value, corresponding_weights, max_location

def main(I_array, W):
	results = []
	for I in I_array:
		max_value, weights, location = max_value_and_weights(I, W)
		print(max_value)
		eles = []
		w4 = np.float64(-1.0)
		w3 = np.float64(-0.5)
		w2 = np.float64(0.5)
		w1 = np.float64(1.0)
		ele = ''
		for weight in weights:
			if weight == w4:
				eles.append('w4')
				ele += 'w4 + '
			if weight == w3:
				eles.append('w3')
				ele += 'w3 + '
			if weight == w2:
				eles.append('w2')
				ele += 'w2 + '
			if weight == w1:
				eles.append('w1')
				ele += 'w1 + '
		ele = ele[:-2]
		results.append((max_value, ele))
	return results


	
def generate_I():
	for bits in product([0, 1], repeat=9):
		yield np.array(bits).reshape(3, 3)
		
def generate_kernel():
	matrix_values = [1, 0.5, -0.5, -1]
	matrices_str = set()
	
	for perm in permutations(matrix_values):
		matrices_str.add(str(perm))
		
	for matrix_str in matrices_str:
		matrix = np.array(eval(matrix_str)).reshape(2, 2)
		yield matrix

I_array = []
for I in generate_I():
	I_array.append(I)
	
W_array = []
for W in generate_kernel():
	W_array.append(W)


results = main(I_array, W_array[0])
for result in results:
	print(f'{result[1]}')
