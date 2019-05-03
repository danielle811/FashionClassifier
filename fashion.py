#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gzip
import zipfile
import sys

import numpy as np
import matplotlib.pyplot as plt

import os
import gzip
import numpy as np
from PIL import Image

def toOneHot(arr):
    res = []
    for i in range(0, len(arr)):
        one_hot = np.zeros(10)
        one_hot[arr[i]] = 1
        res.append(one_hot)
    return np.asarray(res)

# load data
# offset = 8 for lables
# offset = 16 for images 
def read_data(path, offset_no):
    with gzip.open(path, 'rb') as file:
        data = np.frombuffer(file.read(), dtype=np.uint8, offset=offset_no)
    return data

train_labels = read_data('data/train-labels-idx1-ubyte.gz', 8)
train_images = read_data('data/train-images-idx3-ubyte.gz', 16).reshape(len(train_labels), 784)

test_labels = read_data('data/t10k-labels-idx1-ubyte.gz', 8)
test_images = read_data('data/t10k-images-idx3-ubyte.gz', 16).reshape(len(test_labels), 784)

# Convert array to images 
def arr_to_img(arr):
    mat = np.reshape(arr, (28, 28))
    img = Image.fromarray(np.uint8(mat) , 'L')
    img.show()
    
test_images = np.array(test_images)/255
train_images = np.array(train_images)/255

train_labels = toOneHot(train_labels)
test_labels = toOneHot(test_labels)

train_data = []
train_data.append(train_images)
train_data.append(train_labels)

test_data = []
test_data.append(test_images)
test_data.append(test_labels)


# In[2]:


# Convert the spike dictionary from spike_trains() to arrays of 0 and 1
def dictToSpikeTrains(spike_dict):
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


# In[3]:


# STDP
stdp_incre = 0.002
stdp_decre = 0.0005
trace_incre=0.01
trace_decay=-0.01
def getTrace(spike_train, t):
	trace = 0.0
	interval = 0
	for i in range(len(spike_train)):
		interval += 1
		if spike_train[i]==1:
			trace += trace_incre
			interval = 0
		trace = trace*math.exp(trace_decay*interval)
		if t == i:
			return trace
# j - presynaptic, i - postsynaptic
def updateSTDP(spike_train_i, spike_train_j, weight):
	for i in range(len(spike_train_i)):
		# when there's a postsynaptic spike, find the trace on the pre spike train
		if spike_train_i[i]==1:
			weight += stdp_incre*getTrace(spike_train_j, i)
		# when there's a presynaptic spike, find the trace on the post spike train
		if spike_train_j[i]==1:
			weight -= stdp_decre*getTrace(spike_train_i, i)
	return weight


# In[4]:


# generate a current value with respect to its gray scale value
def grayValToCurr(gray_val):
    return (gray_val + 0.01)*10

# Convert a 1-D array image to its corresponding current input
def imageToCurr(img):
    curr = []
    for i in range(len(img)):
        curr.append(grayValToCurr(img[i])*nA)
    return curr


# In[5]:


# Initializing neurons
#get_ipython().run_line_magic('matplotlib', 'notebook')
num_input_neurons = 784
num_output_neurons = 4
'''
Input-Frequency curve of a HH model.
Network: 100 unconnected Hodgin-Huxley neurons with an input current I.
The input is set differently for each neuron.

This simulation should use exponential Euler integration.
'''
from brian2 import *
duration = 0.1*second
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


# In[6]:


# Cut down the number of categories
def filterData(label, clothes_index):
    sum=0
    for i in range(len(clothes_index)):
        sum += label[clothes_index[i]]
    if sum == 1:
        return True
    return False


# In[11]:


# Generate currents for teaching neurons
def getTeachCurr(label):
    if label[0] == 1:
        return [8,0,0,0] * nA
    if label[2] == 1:
        return [0,8,0,0] * nA
    if label[7] == 1:
        return [0,0,8,0] * nA
    if label[8] == 1:
        return [0,0,0,8] * nA
    print('wrong label')
    return [0,0,0,0] * nA


# In[12]:

print('start training')
weight=numpy.zeros((4,784))
#1: Trousers 2: Pullover 7: Sneaker 8: Bag
for epoch in range(1000):
    if epoch%10==0:
        print(str(epoch) + '/1000')
        
    # Ignore the data that are not 1,2,7 or 8
    if filterData(train_data[1][epoch], [0,2,7,8])==False:
        continue
        
    # Restore to the previous checkpoint (reset the time of monitors)
    restore()
    
    input_layer.v = El
    
    # Feed the input layer with image current, output layer with teaching currents 
    input_layer.I = imageToCurr(train_data[0][epoch])
    output_layer.I = getTeachCurr(train_data[1][epoch])
    
    run(duration)    
    
    # spike_trains() is a dictionary that has neuron index as key and its 
    # corresponding time of spikes as value
    spike_dict = spikeMons.spike_trains()
    output_spike_dict = output_spikeMons.spike_trains()
    
    #plot(monitor.t/ms, monitor.v[0])

    # convert the dictionaries to arrays of 0 and 1
    in_trains = dictToSpikeTrains(spike_dict)
    out_trains = dictToSpikeTrains(output_spike_dict)
    
    # update the weights with STDP
    for k in range(num_output_neurons):
        for i in range(len(weight[k])):
            weight[k][i] = updateSTDP(out_trains[k], in_trains[i], weight[k][i])
            
print('done training')


# In[13]:


count, correct = 0,0
for i in range(1000):
    if i%50==0:
        print(i)
    if filterData(test_data[1][i], [0,2,7,8])==False:
        continue
    cur = imageToCurr(test_data[0][i])

    sum=[0,0,0,0]
    x = -1
    for k in range(784):
        sum[0]+=weight[0][k]*cur[k]
        sum[1]+=weight[1][k]*cur[k]
        sum[2]+=weight[2][k]*cur[k]
        sum[3]+=weight[3][k]*cur[k]
    if sum[0]==max(sum):
        x = 0
    elif sum[1]==max(sum):
        x = 2
    elif sum[2]==max(sum):
        x = 7
    else:
        x = 8
    index = -1
    for j in range(10):
        if test_data[1][i][j] == 1:
            index = j
            break
    if x == index:
        correct += 1
    count +=1

print('---------')
print(correct)
print(count)


# In[ ]:




