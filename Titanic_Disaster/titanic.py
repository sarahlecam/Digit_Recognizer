#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 01:00:10 2017

@author: sarahlecam
@author: ym224
"""
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd

def loadData():
    # using pandas to read train + test data
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    
    # getting survival column and chosen features
    survived = train["Survived"].values
    feats = ["Pclass", "Sex", "Age"]
    
    # mapping sex strings to ints for model
    train["Sex"] = train["Sex"].apply(lambda sex:1 if sex == "female" else 0)
    test["Sex"] = test["Sex"].apply(lambda sex:1 if sex == "female" else 0)
    
    # fixing missing values
    train["Age"] = train["Age"].fillna(train["Age"].mean())
    test["Age"] = test["Age"].fillna(test["Age"].mean())
    
    trainData = train[feats].values
    testData = test[feats].values
    ID = test["PassengerId"].values
    
    return trainData, survived, testData, ID

def trainModel(trainData, survived, testData, ID) :
    model = LogisticRegression()
    model.fit(trainData, survived)
    predictsurv = model.predict(testData)
    return predictsurv
    
def generateSubmission(predictsurv) :
    submit = pd.DataFrame(columns=["PassengerId","Survived"])
    submit["PassengerId"] = ID
    submit["Survived"] = predictsurv
    submit.to_csv("TitanicSurvival.csv", index = False)

trainData, survived, testData, ID = loadData()
predictsurv = trainModel(trainData, survived, testData, ID)
generateSubmission(predictsurv)

