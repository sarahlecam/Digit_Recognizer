#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 12:09:31 2017

@author: sarahlecam
"""

from matplotlib import pyplot as plt
import csv
import numpy as np

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
	weights = np.ones_like(labels)/float(len(labels))
	plt.hist(labels, weights=weights, normed=1)
	plt.title('Histogram of digit counts')
	plt.savefig('digitCounts_historgram.png')
    
def L2Dist(r1, r2) :
    with open(input_file, 'r') as file:
        next(file, None)
        reader = csv.reader(file)
        reader = list(reader)
        d1 = np.array(reader[r1][1:], dtype='uint8')
        d2 = np.array(reader[r2][1:], dtype='uint8')
        dist = np.linalg.norm(d1-d2)
    return dist
		
displayEachDigit()
plt.close()
displayDigitCounts()

