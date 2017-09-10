#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 12:09:31 2017

@author: sarahlecam
"""

from matplotlib import pyplot as plt
import csv
import numpy as np
import scipy as sp

input_file = "train.csv"

def displayEachDigit():
	labels = list(range(10))
	i = 1
	fig = plt.figure()
	with open(input_file, 'r') as file:
		next(file, None)
		reader = csv.reader(file)
		for row in reader:
	    	#convert label to integer
			label = np.array(row[0], dtype='uint8')
			if label in labels:
				# convert to integer and reshape pixel dimensions
				pixels = np.array(row[1:], dtype='uint8').reshape(28,28)
				ax = fig.add_subplot(4,3,label+1)
				plt.subplots_adjust(wspace=.2, hspace=.8)
				i = i+1
				plt.imshow(pixels, cmap=plt.get_cmap('gray'))
				labels.remove(label)
	plt.savefig('mnist_digits_plot.png')
			
		
def displayDigitCounts():
    labels = []
    with open(input_file, 'r') as file:
        next(file, None)
        reader = csv.reader(file)
        for row in reader:
            label = float(row[0])
            labels.append(label)
	#plt.bar(list(labelCounts.keys()), labelCounts.values(), 1, color='g')
	#weights = np.ones_like(labels)/float(len(labels))
	#plt.hist(labels, weights=weights, normed=1)
    plt.hist(labels, normed=1)
    plt.title('Histogram of digit counts')
    plt.savefig('digitCounts_historgram.png')
    


def binComparisonHist () :
    
    file = open(input_file, 'r')
    file.readline()
    data = np.loadtxt(file, delimiter = ",").astype(int)
    
    digits0 = data[data[:,0] == 0]
    digits1 = data[data[:,0] == 1]
             
    genuine1 = sp.spatial.distance.pdist(digits1, 'euclidean');
    genuine0 = sp.spatial.distance.pdist(digits0, 'euclidean');
    genuine = np.concatenate((genuine1, genuine0))
            
    imposter = sp.spatial.distance.cdist(digits0, digits1, 'euclidean').flatten()
    
    plt.hist(genuine, bins=100, alpha= .5)
    plt.hist(imposter, bins=100, alpha= .5)
    
    plt.title('Histogram of imposter and genuine distances')
    plt.savefig('imposter_genuine.png')
    
		
displayEachDigit()
plt.close()
displayDigitCounts()
plt.close()
binComparisonHist ()

