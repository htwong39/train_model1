from __future__ import division

import random as rd
import numpy as np
import tensorflow as tf

class Qnetwork():
    def __init__(self,I_size,O_size,n_control):
        #The network recieves a frame from the game, flattened into an array.
        #It then resizes it and processes it through four convolutional layers.
        self.scalarInput =  tf.placeholder(shape=[None,I_size],dtype=tf.float32)
        
        self.f_connect1 = tf.contrib.layers.fully_connected(
            inputs = self.scalarInput,
            num_outputs = 64,
            activation_fn = tf.nn.relu,
            weights_initializer=tf.random_normal_initializer(),
            biases_initializer=tf.random_normal_initializer()
            )
        self.f_connect2 = tf.contrib.layers.fully_connected(
            inputs = self.f_connect1,
            num_outputs = 64,
            activation_fn = tf.nn.relu,
            weights_initializer=tf.random_normal_initializer(),
            biases_initializer=tf.random_normal_initializer()
            )
        self.f_connect3 = tf.contrib.layers.fully_connected(
            inputs = self.f_connect2,
            num_outputs = 64,
            activation_fn = tf.nn.relu,
            weights_initializer=tf.random_normal_initializer(),
            biases_initializer=tf.random_normal_initializer()
            )
        self.f_connect4 = tf.contrib.layers.fully_connected(
            inputs = self.f_connect3,
            num_outputs = O_size,
            activation_fn = tf.nn.relu,
            weights_initializer=tf.random_normal_initializer(),
            biases_initializer=tf.random_normal_initializer()
            )

        #We take the output from the final convolutional layer and split it into separate advantage and value streams.
        self.streamAC,self.streamVC = tf.split(self.f_connect4, num_or_size_splits=2, axis=1)

        #self.streamA = slim.flatten(self.streamAC)
        #self.streamV = slim.flatten(self.streamVC)

        self.streamA = self.streamAC
        self.streamV = self.streamVC

        xavier_init = tf.contrib.layers.xavier_initializer()
        self.AW = tf.Variable(xavier_init([O_size//2, 5*n_control]))
        self.VW = tf.Variable(xavier_init([O_size//2, 1]))
        self.Advantage = tf.matmul(self.streamA,self.AW)
        self.Value = tf.matmul(self.streamV,self.VW)

        #Then combine them together to get our final Q-values.
        self.Qout = self.Value + tf.subtract(self.Advantage,tf.reduce_mean(self.Advantage,keep_dims=True))
        
        sizeQ = tf.shape(self.Qout)

        self.Qout_reshape = tf.reshape(self.Qout,[ tf.to_int32(sizeQ[0]*n_control) , tf.to_int32(sizeQ[1]/n_control) ])

        self.predict = tf.argmax(self.Qout_reshape,1)
        
        #network generate all the action-value pair for the input state, we sample some action-value pair from memory, we just need to min the different between o and out 

        #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None,n_control],dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None,n_control],dtype=tf.int32)

        self.actions_onehot = tf.one_hot(self.actions, 5, dtype=tf.float32) 

        hotsize = tf.shape(self.actions_onehot)

        self.reshape_hot = tf.reshape(self.actions_onehot,[hotsize[0]*n_control,5])

        self.sum = tf.reduce_sum(tf.multiply(self.Qout_reshape , self.reshape_hot),axis=1)
        self.Q = tf.reshape(self.sum,[hotsize[0],n_control])
        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)

class experience_buffer():
    def __init__(self, buffer_size = 50000):

        self.buffer = []
        self.buffer_size = buffer_size
    
    def add(self,experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
        self.buffer.extend(experience)
            
    def sample(self,size):
        return np.reshape(np.array(rd.sample(self.buffer,size)),[size,5])

#def processState(states):
#   return np.reshape(states,[21168])

