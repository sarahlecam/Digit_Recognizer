#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 12:09:31 2017

@author: sarahlecam
@author: ym224
"""

from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix 
import numpy as np
import scipy as sp

train_data = "train.csv"
test_data = "test.csv"
output_file = "digitRecognizer.csv"

def loadData(input_file):
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
    plt.close()
    

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


def KNNClassifierNaive(trainData, classifier, k):
	# construct list of indices of training data sorted by distance to classifier
	dist = []
	for i in range(trainData.shape[0]):
		dist.append(np.linalg.norm(trainData[i][1:] - classifier))
	sortedDistIndices = np.argsort(dist)

	# classify sample based on majority vote of class labels on the nearest neighbour list 
	classCount = {}
	for i in range(k):
		neighborIndex = sortedDistIndices[i]
		label = trainData[neighborIndex][0]
		classCount[label] = classCount.get(label, 0) + 1
	majorityClass = sorted(classCount, key=classCount.get, reverse=True)[0]
	return majorityClass

def perform3FoldCrossValidation(data, k):
	actual = []
	predicted = []
	kf = KFold(n_splits=3, random_state=None, shuffle=False)
	for train_index, test_index in kf.split(data):
		trainData = np.array([data[i] for i in train_index])
		testData = np.array([data[i] for i in test_index])
		for row in range(testData.shape[0]):
			predicted.append(KNNClassifierNaive(trainData, testData[row][1:], 10))
			actual.append(testData[row][0])
	cm = confusion_matrix(actual, predicted)
	accuracy = np.array(cm).trace()/np.array(cm).sum()
	print (cm)
	print (accuracy)
	return cm

def plotConfusionMatrix(cm):
	cm = np.array(cm)
	plt.figure()
	plt.imshow(cm)
	plt.title('Confusion matrix')
	labels = list(range(10))
	tick_marks = np.arange(len(labels))
	plt.xticks(tick_marks, labels, rotation=45)
	plt.yticks(tick_marks, labels)
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center", color="black")
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.savefig('confusion_matrix_3Fold_CV.png')

def runKNNClassifierOnTestData(trainData, testData):
	file = open(output_file, 'w')
	# write header
	file.write('ImageId, Label')
	for row in range(testData.shape[0]):
		label = KNNClassifierNaive(trainData, testData[row][:], 10)
		file.write('\n')
		file.write(str(row+1) + ',' + str(label))
	file.close()

# load training data
data = loadData(train_data)

# load test data
test_data = loadData(test_data)

# display first occurance of each digit and return indices
digitSample = displayEachDigit(data)

# create histogram of digit counts
displayDigitCounts(data)

# find indices of nearest neighbor to list of digit samples
nearestNeighbors = findNearestNeighbor(data, digitSample)

# plot digit sample with its nearest neighbor
plotSampleWithNearestNeighbor(nearestNeighbors)

# create histogram of pairwise comparison
binComparisonHist(data)

# create confusion matrix for 3Fold cross validation on training data
confusion_matrix = perform3FoldCrossValidation(data, 10)

# plot confusion matrix
plotConfusionMatrix(confusion_matrix)

# run KNN classifier on test data using training data and write result to file
runKNNClassifierOnTestData(data, test_data)