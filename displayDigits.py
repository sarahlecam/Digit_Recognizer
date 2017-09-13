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
	fig = plt.figure(num=None, figsize=(8, 10), dpi=85, facecolor='w', edgecolor='k')	
	rows = data.shape[0]
	data = np.array(data, dtype='uint8')
	for row in range(rows):
		label = data[row, 0]
		if label in labels:
			digitSample.append(row)
			# Reshape pixel dimensions
			pixels = data[row, 1:].reshape(28,28)
			ax = fig.add_subplot(4,3,label+1)
			plt.subplots_adjust(hspace=.4)
			ticks = list(range(0,28,25))
			plt.xticks(ticks)
			plt.yticks(ticks)
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
	plt.xlabel('Digit')
	plt.ylabel('Prior Probability')
	plt.savefig('digitCounts_historgram.png')
	plt.close()


def binComparisonHist(data):
    
    # Creating 2 arrays for digits 1 and 0
    digits0 = data[data[:,0] == 0]
    digits1 = data[data[:,0] == 1]
       
    # Running pairwise distance on both arrays and concatenating results     
    genuine1 = sp.spatial.distance.pdist(digits1, 'euclidean');
    genuine0 = sp.spatial.distance.pdist(digits0, 'euclidean');
    genuine = np.concatenate((genuine1, genuine0))
    
    # Pairwise distance between 2 sets for imposters
    imposter = sp.spatial.distance.cdist(digits0, digits1, 'euclidean').flatten()
    
    # plotting both histograms
    plt.hist(genuine, bins=100, alpha= .5)
    plt.hist(imposter, bins=100, alpha= .5)
    
    plt.title('Histogram of imposter and genuine distances')
    plt.savefig('imposter_genuine.png')
    plt.close()
    
    return genuine, imposter;
    
    
def ROCcurve(data, genuine, imposter) :
    fpr = []
    tpr = []
    eer = 0
    for i in np.arange(int(min(genuine)), int(max(imposter)), 100):
        true = len(genuine[genuine <= i]) / len(genuine)
        false = len(imposter[imposter <= i]) / len(imposter)
        tpr.append(true)
        fpr.append(false)
        #if (round((1 - true),1) == round(false,1)) :
            #eer = false
        
    #print(eer)
    
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], color='red', lw=1, linestyle='--')
    plt.xlim([0.0, 1.1])
    plt.ylim([0.0, 1.1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.savefig('ROC_curve.png')
    plt.close()
    

def findNearestNeighbor(data, digitSample):
	data = np.array(data)
	sampleToNeighbor = {}
	dists = sp.spatial.distance.cdist(data[digitSample, :], data, 'euclidean')
	dists[dists == 0] = np.nan
	for i in range(len(digitSample)):
		neighborIndex = np.nanargmin(dists[i, :])
		sampleToNeighbor[digitSample[i]] = neighborIndex
	return sampleToNeighbor

				
def plotSampleWithNearestNeighbor(data, sampleToNeighbor):
	data = np.array(data)
	for sampleIndex, neighborIndex in sampleToNeighbor.items():
		fig = plt.figure()
		sampleLabel = data[sampleIndex, 0]
		samplePixels = data[sampleIndex, 1:].reshape(28,28)
		neighborPixels = data[neighborIndex, 1:].reshape(28,28)
		plt.subplots_adjust(wspace=.2, hspace=.4)
		ticks = list(range(0,28,25))
		ax = fig.add_subplot(1, 2, 1)
		plt.imshow(samplePixels, cmap=plt.get_cmap('gray'))
		plt.xticks(ticks)
		plt.yticks(ticks)
		ax = fig.add_subplot(1, 2, 2)
		plt.imshow(neighborPixels, cmap=plt.get_cmap('gray'))
		plt.xticks(ticks)
		plt.yticks(ticks)
		plt.savefig('nearestNeighbors_plot_' + str(sampleLabel) + '.png')
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
	cm = np.array(confusion_matrix(actual, predicted))
	accuracy = cm.trace()/cm.sum()
	return cm

def plotConfusionMatrix(cm):
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
	file.write('ImageId,Label')
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
sampleToNeighbor = findNearestNeighbor(data, digitSample)

# # plot digit sample with its nearest neighbor
plotSampleWithNearestNeighbor(data, sampleToNeighbor)

# create histogram of pairwise comparison
genuine, imposter = binComparisonHist(data)

# create roc curve
ROCcurve(data, genuine, imposter)

# create confusion matrix for 3Fold cross validation on training data
confusion_matrix = perform3FoldCrossValidation(data, 10)

# plot confusion matrix
plotConfusionMatrix(confusion_matrix)

# run KNN classifier on test data using training data and write result to file
runKNNClassifierOnTestData(data, test_data)