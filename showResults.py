#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 10:56:37 2018

@author: jiayicheng

Plot the validation and test accuracy of DNN and CNN

"""
import matplotlib.pyplot as plt

def show(vDNN, tDNN, vCNN, tCNN, INTERVAL_SHOW, TRAINING_STEPS):
    
    xAxis = [i*INTERVAL_SHOW for i in range(TRAINING_STEPS // INTERVAL_SHOW + 1)]
    
    plt.close()
    plt.figure()
    p1 = plt.subplot(211)
    p2 = plt.subplot(212)
    ## plot the whole graph
    # plot the validation and test accuracy of DNN
    if vDNN and tDNN:
        p1.plot(xAxis,vDNN,"r-",label="DNN: validation acc")
        p1.plot(xAxis,tDNN,"r-.",label="DNN: test acc", linestyle = '--')
    if vCNN and tCNN:
        p1.plot(xAxis,vCNN,"g-",label="CNN: validation acc")
        p1.plot(xAxis,tCNN,"g-.",label="CNN: test acc", linestyle = '--')
    p1.set_ylabel("accuracy")
    p1.set_title("Validation and test results")
    p1.grid(True)
    p1.legend()
    p1.ax = plt.gca()
    p1.ax.spines['right'].set_color('none')
    p1.ax.spines['top'].set_color('none')
    
    ## plot the zoomin version
    # plot the validation and test accuracy of DNN
    if vDNN and tDNN:
        p2.plot(xAxis,vDNN,"r-",label="DNN: validation acc")
        p2.plot(xAxis,tDNN,"r-.",label="DNN: test acc", linestyle = '--')
    # plot the validation and test accuracy of CNN
    if vCNN and tCNN:
        p2.plot(xAxis,vCNN,"g-",label="CNN: validation acc")
        p2.plot(xAxis,tCNN,"g-.",label="CNN: test acc", linestyle = '--')
    p2.set_xlim((TRAINING_STEPS//2, TRAINING_STEPS)) 
    p2.set_ylim((0.85, 1.0))
    p2.set_xlabel("# of training steps")
    p2.set_ylabel("accuracy")
    p2.grid(True)
    p2.legend()
    p2.ax = plt.gca()
    p2.ax.spines['right'].set_color('none')
    p2.ax.spines['top'].set_color('none')
    plt.show()