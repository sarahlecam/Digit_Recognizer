#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 12:09:31 2017

@author: sarahlecam
@author: ym224
"""

from matplotlib import pyplot as plt
import csv
import numpy as np
import scipy as sp

input_file = "train.csv"

def loadData():
	file = open(input_file, 'r')
	file.readline()
	data = np.loadtxt(file, delimiter=",").astype(int)
	return data

def displayEachDigit(data):
	digitSample = []
	labels = list(range(10))
	fig = plt.figure()			
	rows = data.shape[0]
	for row in range(rows):
		#convert label to integer
		label = np.array(data[row][0], dtype='uint8')
		if label in labels:
			digitSample.append(row)
			# convert to integer and reshape pixel dimensions
			pixels = np.array(data[row][1:], dtype='uint8').reshape(28,28)
			ax = fig.add_subplot(4,3,label+1)
			plt.subplots_adjust(wspace=.2, hspace=.8)
			plt.imshow(pixels, cmap=plt.get_cmap('gray'))
			labels.remove(label)
	plt.savefig('mnist_digits_plot.png')
	plt.close()
	return digitSample			
		

def displayDigitCounts(data):
	labels = []
	file = open(input_file, 'r')
	file.readline()
	for row in range(data.shape[0]):
		label = float(data[row][0])
		labels.append(label)
	plt.hist(labels, normed=1)
	plt.title('Histogram of digit counts')
	plt.savefig('digitCounts_historgram.png')
	plt.close()


def binComparisonHist(data):
     
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
    

def findNearestNeighbor(data, digitSample):
	nearestNeighbors = {}
	for sample in digitSample:
		pixelsSample = np.array(data[sample][1:], dtype='uint8').reshape(28,28)
		dist = 1000000
		neighborRow = 0
		for row in range(data.shape[0]):
			if (row != sample):
				pixels = np.array(data[row][1:], dtype='uint8').reshape(28,28)
				newDist = np.linalg.norm(pixelsSample - pixels)
				if newDist < dist:
					dist = newDist
					neighborRow = row
		nearestNeighbors[sample] = neighborRow
	return nearestNeighbors
				
def plotSampleWithNearestNeighbor(nearestNeighbors):
	fig = plt.figure()
	for sampleIndex, neighborIndex in nearestNeighbors.items():
		sampleLabel = np.array(data[sampleIndex][0], dtype='uint8')
		samplePixels = np.array(data[sampleIndex][1:], dtype='uint8').reshape(28,28)
		neighborPixels = np.array(data[neighborIndex][1:], dtype='uint8').reshape(28,28)
		plt.subplots_adjust(wspace=.2, hspace=.8)
		ax = fig.add_subplot(10,2, sampleLabel*2+1)
		plt.imshow(samplePixels, cmap=plt.get_cmap('gray'))
		ax = fig.add_subplot(10,2, sampleLabel*2+2)
		plt.imshow(neighborPixels, cmap=plt.get_cmap('gray'))
	plt.savefig('nearestNeighbors_plot.png')
	plt.close()
				

data = loadData()
digitSample = displayEachDigit(data)
displayDigitCounts(data)
nearestNeighbors = findNearestNeighbor(data, digitSample)
plotSampleWithNearestNeighbor(nearestNeighbors)
binComparisonHist()

