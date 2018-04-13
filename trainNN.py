#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 23:45:58 2018

@author: jiayicheng

After building the DNN and CNN, we model the cost function by cross entropy 
and use Adam optimizer to minimize it. We record the validation and test
accuracy of DNN and CNN. Finally, we plot them.

"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import buildNN
import showResults
import time
## define the traing parameters
BATCH_SIZE = 100
LEARNING_RATE = 0.001
TRAINING_STEPS = 1000
INTERVAL_SHOW = 50

## record the validation and test accuracy of DNN and CNN
validationAccDNN = []
testAccDNN = []
validationAccCNN = []
testAccCNN = []
tf.reset_default_graph()

## define the training function
def trainNN(mnist, model):
    t = time.time()
    if not model:
        return None
    # define the common place holder of DNN and CNN
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, buildNN.INPUT_SIZE], name = 'xInput')
        yTrue = tf.placeholder(tf.float32, [None, buildNN.OUTPUT_SIZE], name = 'yInput')
    # DNN
    if model == 'DNN':
        # generate the predicted labels from DNN
        yPrediction = buildNN.buildDNN(x)
        # build the lost functions and I use crossentropy
        with tf.name_scope('lossDNN'):
            crossEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=yPrediction, labels=tf.argmax(yTrue, 1))
            loss = tf.reduce_mean(crossEntropy)
        # choose the Adam optimizer and the learning rate is chosen by trial and error
        with tf.name_scope('trainDNN'):
            train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
        # define the evaluation standard
        correctPrediction = tf.equal(tf.argmax(yPrediction,1), tf.argmax(yTrue,1))
        accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))
        # START training ... ...
        with tf.Session() as sess1:
            # use Tensorboard to visualize
            writer = tf.summary.FileWriter('graphDNN',sess1.graph)
            # initialize 
            sess1.run(tf.global_variables_initializer())
            # feed the validation and test data
            validationFeed = {x: mnist.validation.images, yTrue: mnist.validation.labels}
            testFeed = {x: mnist.test.images, yTrue: mnist.test.labels}
            tStartDNN = time.time() - t
            for i in range(TRAINING_STEPS+1):
                # feed the training data
                xTrain, yTrain = mnist.train.next_batch(BATCH_SIZE)
                # record the accuracies every INTERVAL_SHOW iterations
                if i % INTERVAL_SHOW == 0:
                    validationAcc = sess1.run(accuracy, feed_dict = validationFeed)
                    validationAccDNN.append(validationAcc)
                    testAcc = sess1.run(accuracy, feed_dict = testFeed)
                    testAccDNN.append(testAcc)
                    print("DNN: %d rounds, validationAcc = %g, testAcc=%g" % (i, validationAcc, testAcc))
                # keep training
                sess1.run(train, feed_dict = {x: xTrain, yTrue: yTrain})
            tEndDNN = time.time() - t
            timeCostDNN = tEndDNN - tStartDNN
            print(timeCostDNN)
    ## CNN  
    if model == 'CNN':
        keepProb = tf.placeholder(tf.float32)
        # the inputs of CNN should be not be vectors
        xReshaped = tf.reshape(x, [-1, buildNN.IMAGE_SIZE, buildNN.IMAGE_SIZE, 1])
        # generate the predicted labels from CNN
        yPrediction = buildNN.buildCNN(xReshaped, keepProb)
        # build the lost functions and I use crossentropy
        with tf.name_scope('lossCNN'):
            crossEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=yPrediction, labels=tf.argmax(yTrue, 1))
            loss = tf.reduce_mean(crossEntropy)
        # choose the Adam optimizer and the learning rate is chosen by trial and error
        with tf.name_scope('trainCNN'):
            train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
        # define the evaluation standard
        correctPrediction = tf.equal(tf.argmax(yPrediction,1), tf.argmax(yTrue,1))
        accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))
        # START training ... ...
        with tf.Session() as sess2:
            # use Tensorboard to visualize
            writer = tf.summary.FileWriter('graphCNN',sess2.graph)
            # initialize 
            sess2.run(tf.global_variables_initializer())
            # feed the validation and test data
            validationFeed = {x: mnist.validation.images, yTrue: mnist.validation.labels, keepProb: 1}
            testFeed = {x: mnist.test.images, yTrue: mnist.test.labels, keepProb: 1}
            tStartCNN = time.time() - t
            for i in range(TRAINING_STEPS+1):
                # feed the training data
                xTrain, yTrain = mnist.train.next_batch(BATCH_SIZE)
                # record the accuracies every INTERVAL_SHOW iterations
                if i % INTERVAL_SHOW == 0:
                    validationAcc = sess2.run(accuracy, feed_dict = validationFeed)
                    validationAccCNN.append(validationAcc)
                    testAcc = sess2.run(accuracy, feed_dict = testFeed)
                    testAccCNN.append(testAcc)
                    print("CNN: %d rounds, validationAcc=%g, testAcc=%g" % (i, validationAcc, testAcc))
                # keep training
                sess2.run(train, feed_dict = {x: xTrain, yTrue: yTrain, keepProb: 0.5})
            tEndCNN = time.time() - t
            timeCostCNN = tEndCNN - tStartCNN
            print(timeCostCNN)
    writer.close()
def main(argv = None):
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    trainNN(mnist,'DNN')
    # trainNN(mnist,'CNN')
    showResults.show(validationAccDNN, testAccDNN, validationAccCNN, testAccCNN, INTERVAL_SHOW, TRAINING_STEPS) 
    
if __name__ == '__main__':
    tf.app.run()
