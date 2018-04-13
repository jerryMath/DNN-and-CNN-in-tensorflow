#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 22:13:53 2018

@author: jiayicheng

Build the DNN and CNN structures based on the forwarp propagation.

1. The DNN has 3 hidden layers and uses RELU as the activation function.

DNN structure:
         INPUT--------->HIDDEN_LAYER1--------->HIDDEN_LAYER1--------->HIDDEN_LAYER1--------->OUTPUT
size:    1*784  784*500 1*500          500*300 1*300          300*100 1*100          100*10  1*10

2. The CNN has 2 convolutional layers combined with RELU and MAX POOLING and 2 fully connected (FC) layers.

CNN structure:
         INPUT-->CONV1-->MAX_POOLING1-->CONV2-->MAX_POOLING2-->FC_LAYER1-->FC_LAYER2-->OUTPUT
size:    28*28*1 5*5*32  2*2            5*5*64  2*2            1*3136      1*512       1*10
"""
import tensorflow as tf

## define the size of input and output
INPUT_SIZE = 784
OUTPUT_SIZE = 10

## define the size of each layer of DNN
HIDDEN_LAYER1_SIZE = 512
HIDDEN_LAYER2_SIZE = 256
HIDDEN_LAYER3_SIZE = 128

## define the size of each layer of CNN
IMAGE_SIZE = 28   # the size of MNIST image
IMAGE_DEPTH = 1   # black and white
CONV1_SIZE = 5    # conv1 5*5*32
CONV1_DEPTH = 32  # out depth
CONV2_SIZE = 5    # conv2 5*5*64
CONV2_DEPTH = 64  # out depth 
FC1_SIZE = 512    # fc1



## DNN
def buildDNN(inputs):
    
    with tf.variable_scope('hidden_layer1'):
        weights = tf.get_variable('weights',[INPUT_SIZE, HIDDEN_LAYER1_SIZE],
                            initializer = tf.truncated_normal_initializer(stddev = 0.1))
        print(weights)
        biases = tf.get_variable('biases', [HIDDEN_LAYER1_SIZE],
                                 initializer = tf.constant_initializer(0.1))
        layer1 = tf.nn.relu(tf.matmul(inputs, weights) + biases)
        
    with tf.variable_scope('hidden_layer2'):
        weights = tf.get_variable('weights',[HIDDEN_LAYER1_SIZE, HIDDEN_LAYER2_SIZE],
                            initializer = tf.truncated_normal_initializer(stddev = 0.1))
        print(weights)
        biases = tf.get_variable('biases', [HIDDEN_LAYER2_SIZE],
                                 initializer = tf.constant_initializer(0.1))
        layer2 = tf.nn.relu(tf.matmul(layer1, weights) + biases)
        
    with tf.variable_scope('hidden_layer3'):
        weights = tf.get_variable('weights',[HIDDEN_LAYER2_SIZE, HIDDEN_LAYER3_SIZE],
                            initializer = tf.truncated_normal_initializer(stddev = 0.1))
        print(weights)
        biases = tf.get_variable('biases', [HIDDEN_LAYER3_SIZE],
                                 initializer = tf.constant_initializer(0.1))
        layer3 = tf.nn.relu(tf.matmul(layer2, weights) + biases)
    
    return layer3

## CNN
def conv(inputs, weights):
    return tf.nn.conv2d(inputs, weights, strides=[1, 1, 1, 1], padding='SAME')

def max_pooling(inputs):
    return tf.nn.max_pool(inputs, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def buildCNN(inputs, keepProb):
    with tf.variable_scope('conv_layer1'):
        weights = tf.get_variable('weights', [CONV1_SIZE, CONV1_SIZE, IMAGE_DEPTH, CONV1_DEPTH],
                                  initializer = tf.truncated_normal_initializer(stddev = 0.1))
        biases = tf.get_variable('biases', [CONV1_DEPTH], initializer = tf.constant_initializer(0.1))
        conv1 = tf.nn.relu(conv(inputs, weights) + biases)
        
    with tf.variable_scope('maxpooling_layer2'):
        maxPool1 = max_pooling(conv1)
    
    with tf.variable_scope('conv_layer3'):
        weights = tf.get_variable('weights', [CONV2_SIZE, CONV2_SIZE, CONV1_DEPTH, CONV2_DEPTH],
                                  initializer = tf.truncated_normal_initializer(stddev = 0.1))
        biases = tf.get_variable('biases', [CONV2_DEPTH], initializer = tf.constant_initializer(0.1))
        conv2 = tf.nn.relu(conv(maxPool1, weights) + biases)
        
    with tf.variable_scope('maxpooling_layer4'):
        maxPool2 = max_pooling(conv2)
        maxPool2Shape = maxPool2.get_shape().as_list() 
        numElements = maxPool2Shape[1] * maxPool2Shape[2] * maxPool2Shape[3]
        maxPool2Vectorize = tf.reshape(maxPool2, [-1, numElements])
    
    with tf.variable_scope('fc_layer5'):
        weights = tf.get_variable('weights',[numElements, FC1_SIZE],
                            initializer = tf.truncated_normal_initializer(stddev = 0.1))
        biases = tf.get_variable('biases', [FC1_SIZE],
                                 initializer = tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(maxPool2Vectorize, weights) + biases)
        fc1Drop = tf.nn.dropout(fc1, keepProb)
    
    with tf.variable_scope('fc_layer6'):
        weights = tf.get_variable('weights',[FC1_SIZE, OUTPUT_SIZE],
                            initializer = tf.truncated_normal_initializer(stddev = 0.1))
        biases = tf.get_variable('biases', [OUTPUT_SIZE],
                                 initializer = tf.constant_initializer(0.1))
        fc2 = tf.matmul(fc1Drop, weights) + biases
    
    return fc2
        
numParameterDNN = (784*512+512) + (512*256+256) + (256*128+128) + (128*10+10)         
numParameterCNN = (5*5*1*32+32) + (5*5*32*64+64) + (7*7*64*512+512) + (512*10+10)
        
        
        
        
        
        
        
        
        