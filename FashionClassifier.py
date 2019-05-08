#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Github  : https://github.com/danielle811/FashionClassifier

import pickle
import time
import gzip
import zipfile
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from brian2 import *

class Data:
	def __init__(self, class_type, path = 'data'):
		self.type = class_type
		self.path_labels = ''
		self.path_images = ''
		self.get_path(path)
		self.labels = self.read_data(self.path_labels, 8)
		self.images = self.read_data(self.path_images, 16).reshape(len(self.labels), 784)
		self.data = self.reshape_data()
		
	def reshape_data(self):
		images = np.array(self.images)/255
		labels = self.toOneHot(self.labels)
		data = []
		data.append(images)
		data.append(labels)
		return data

	def get_path(self, path):
		if self.type is 'train':
			self.path_labels = path+'/train-labels-idx1-ubyte.gz'
			self.path_images = path+'/train-images-idx3-ubyte.gz'
		elif self.type is 'test':
			self.path_labels = path+'/t10k-labels-idx1-ubyte.gz'
			self.path_images = path+'/t10k-images-idx3-ubyte.gz'
		else:
			print('ERROR! Wrong class_type!')

	def toOneHot(self, arr):
		res = []
		for i in range(0, len(arr)):
			one_hot = np.zeros(10)
			one_hot[arr[i]] = 1
			res.append(one_hot)
		return np.asarray(res)

	def read_data(self, path, offset_no):
		with gzip.open(path, 'rb') as file:
			data = np.frombuffer(file.read(), dtype=np.uint8, offset=offset_no)
		return data

	def arr_to_img(self, arr):
		mat = np.reshape(arr, (28, 28))
		img = Image.fromarray(np.uint8(mat) , 'L')
		img.show()

class STDP:
	def __init__(self):
		self.stdp_incre = 0.002
		self.stdp_decre = 0.0005
		self.trace_incre = 0.01
		self.trace_decay = -0.01

	def getTrace(self, spike_train, t):
		trace = 0.0
		interval = 0
		for i in range(len(spike_train)):
			interval += 1
			if spike_train[i] == 1:
				trace += self.trace_incre
				interval = 0
			trace = trace*math.exp(self.trace_decay*interval)
			if t == i:
				return trace

	def updateSTDP(self, spike_train_i, spike_train_j, weight):
		for i in range(len(spike_train_i)):
			# when there's a postsynaptic spike, find the trace on the pre spike train
			if spike_train_i[i] == 1:
				weight += self.stdp_incre*self.getTrace(spike_train_j, i)
			# when there's a presynaptic spike, find the trace on the post spike train
			if spike_train_j[i] == 1:
				weight -= self.stdp_decre*self.getTrace(spike_train_i, i)
		return weight

class Tools:
	# Convert the spike dictionary from spike_trains() to arrays of 0 and 1
	def dictToSpikeTrains(self, spike_dict):
		# How many time steps in one second
		time_steps = 1000
		all_trains = []
		for i in range(len(spike_dict)):
			spikes = spike_dict.get(i)
			spikes = [int(x/second*time_steps) for x in spikes]
			#print(spikes)
			train = numpy.zeros(100)
			
			for j in range(len(spikes)):
				# index=second-1
				train[spikes[j]-1]=1
			all_trains.append(train)
		return all_trains

	# generate a current value with respect to its gray scale value
	def grayValToCurr(self, gray_val):
		return (gray_val + 0.01)*10

	# Convert a 1-D array image to its corresponding current input
	def imageToCurr(self, img):
		# print('in imageToCurr')
		curr = []
		for i in range(len(img)):
			curr.append(self.grayValToCurr(img[i])*nA)
		return curr

	# Cut down the number of categories
	def filterData(self, label, clothes_index):
		total = 0
		for i in range(len(clothes_index)):
			total += label[clothes_index[i]]
		if total == 1:
			return True
		return False

	# Generate currents for teaching neurons
	def getTeachCurr(self, label):
		# print('in getTeachCurr')
		if label[target_categories[0]] == 1:
			return [8,0,0,0] * nA
		elif label[target_categories[1]] == 1:
			return [0,8,0,0] * nA
		elif label[target_categories[2]] == 1:
			return [0,0,8,0] * nA
		elif label[target_categories[3]] == 1:
			return [0,0,0,8] * nA
		else:
			print('wrong label')
			return [0,0,0,0] * nA

if __name__ == '__main__':
	start_time = time.time()
	# set train to False to use the pre-trained weights
	train = True

	target_categories = [1,2,7,8]

	tools = Tools()
	train_data = Data('train')
	test_data = Data('test')
	stdp = STDP()

	duration = 0.1*second
	num_input_neurons = 784
	num_output_neurons = 4
	start_scope()
	# Parameters
	area = 20000*umetre**2
	Cm = 1*ufarad*cm**-2 * area
	gl = 5e-5*siemens*cm**-2 * area
	El = -65*mV
	EK = -90*mV
	ENa = 50*mV
	g_na = 100*msiemens*cm**-2 * area
	g_kd = 30*msiemens*cm**-2 * area
	VT = -63*mV

	# The model
	eqs = Equations('''
	dv/dt = (gl*(El-v) - g_na*(m*m*m)*h*(v-ENa) - g_kd*(n*n*n*n)*(v-EK) + I)/Cm : volt
	dm/dt = 0.32*(mV**-1)*(13.*mV-v+VT)/
		(exp((13.*mV-v+VT)/(4.*mV))-1.)/ms*(1-m)-0.28*(mV**-1)*(v-VT-40.*mV)/
		(exp((v-VT-40.*mV)/(5.*mV))-1.)/ms*m : 1
	dn/dt = 0.032*(mV**-1)*(15.*mV-v+VT)/
		(exp((15.*mV-v+VT)/(5.*mV))-1.)/ms*(1.-n)-.5*exp((10.*mV-v+VT)/(40.*mV))/ms*n : 1
	dh/dt = 0.128*exp((17.*mV-v+VT)/(18.*mV))/ms*(1.-h)-4./(1+exp((40.*mV-v+VT)/(5.*mV)))/ms*h : 1
	I : amp
	''')
	# Threshold and refractoriness are only used for spike counting
	input_layer = NeuronGroup(num_input_neurons, eqs,
						threshold='v > -40*mV',
						refractory='v > -40*mV',
						method='exponential_euler')
	output_layer = NeuronGroup(num_output_neurons, eqs,
						threshold='v > -40*mV',
						refractory='v > -40*mV',
						method='exponential_euler')

	# monitor is used to keep track of the voltage, not needed for this
	monitor = StateMonitor(input_layer,'v', record=range(num_input_neurons))
	spikeMons = SpikeMonitor(input_layer)
	output_spikeMons = SpikeMonitor(output_layer)

	store()

	print('start training')
	weight=numpy.zeros((4,784))

	if train:
		#1: Trousers 2: Pullover 7: Sneaker 8: Bag
		for epoch in range(1000):
			if epoch%10==0:
				print('complete:' + str(epoch) + '/1000')
				
			# Ignore the data that are not 1,2,7 or 8
			if tools.filterData(train_data.data[1][epoch], target_categories)==False:
				continue
				
			# Restore to the previous checkpoint (reset the time of monitors)
			restore()
			
			input_layer.v = El
			
			# Feed the input layer with image current, output layer with teaching currents 
			input_layer.I = tools.imageToCurr(train_data.data[0][epoch])
			output_layer.I = tools.getTeachCurr(train_data.data[1][epoch])
			
			run(duration)	
			
			# spike_trains() is a dictionary that has neuron index as key and its 
			# corresponding time of spikes as value
			spike_dict = spikeMons.spike_trains()
			output_spike_dict = output_spikeMons.spike_trains()

			# convert the dictionaries to arrays of 0 and 1
			in_trains = tools.dictToSpikeTrains(spike_dict)
			out_trains = tools.dictToSpikeTrains(output_spike_dict)
			
			# update the weights with STDP
			for k in range(num_output_neurons):
				for i in range(len(weight[k])):
					weight[k][i] = stdp.updateSTDP(out_trains[k], in_trains[i], weight[k][i])

		# Save the newly updated weights to weights.txt
		with open('weights.txt','wb') as f:
			pickle.dump(weight, f)
	else:
		# load the pre-trained weights 
		with open('weights.txt','rb') as f:
			weight = pickle.load(f)


	print('done training')

	elapsed_time = time.time() - start_time
	print('Time Used for Training: ', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

	print('start testing')
	count, correct = 0,0
	for i in range(1000):
		if i%50==0:
			print('complete:', i, '/1000')
		if tools.filterData(test_data.data[1][i], target_categories)==False:
			continue
		cur = tools.imageToCurr(test_data.data[0][i])

		# add up all the currents with respect to the weights
		sum=[0,0,0,0]
		x = -1
		for k in range(784):
			sum[0]+=weight[0][k]*cur[k]
			sum[1]+=weight[1][k]*cur[k]
			sum[2]+=weight[2][k]*cur[k]
			sum[3]+=weight[3][k]*cur[k]

		# See which one has the strongest current 
		if sum[0]==max(sum):
			pred = target_categories[0]
		elif sum[1]==max(sum):
			pred = target_categories[1]
		elif sum[2]==max(sum):
			pred = target_categories[2]
		else:
			pred = target_categories[3]
		
		index = -1
		for j in range(10):
			if test_data.data[1][i][j] == 1:
				index = j
				break
		if pred == index:
			correct += 1

		count +=1
	print('done testing')
	
	print('---------')
	print('correct:', correct)
	print('total tested:', count)
	print('accuracy:', '{0:.3f}%'.format(correct/count*100))
