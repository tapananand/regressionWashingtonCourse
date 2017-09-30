# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 21:29:57 2017

@author: taanand
"""

import pandas as pd
import numpy as np

dtype_dict = {
    'bathrooms':float, 
    'waterfront':int, 
    'sqft_above':int, 
    'sqft_living15':float, 
    'grade':int, 
    'yr_renovated':int, 
    'price':float, 
    'bedrooms':float, 
    'zipcode':str, 
    'long':float, 
    'sqft_lot15':float, 
    'sqft_living':float, 
    'floors':str, 
    'condition':int, 
    'lat':float, 
    'date':str, 
    'sqft_basement':int, 
    'yr_built':int, 
    'id':str, 
    'sqft_lot':int, 
    'view':int
}

housesTrain = pd.read_csv("./kc_house_train_data.csv", dtype=dtype_dict)
housesTest = pd.read_csv("./kc_house_test_data.csv", dtype=dtype_dict)

def simpleLinearRegression(X, y):
    XSquaredMean = np.mean(np.square(X))
    XyMean = np.mean(np.multiply(X, y))
    XMean = np.mean(X)
    yMean = np.mean(y)
    
    slope = (XyMean - (XMean * yMean)) / (XSquaredMean - (XMean * XMean))
    
    intercept = yMean - slope * XMean
    
    return (intercept, slope)

def getRegressionPredictions(XTest, intercept, slope):
    output = []
    for X in XTest:
        output.append(intercept + X * slope)
    return output

def getResidualSumOfSquares(XTest, yTest, intercept, slope):
    yPred = getRegressionPredictions(XTest, intercept, slope)
    rss = 0
    for i in range(0, len(yPred)):
        rss += (yTest[i] - yPred[i]) ** 2
    
    return rss

def inverseRegressionPredictions(yTest, intercept, slope):
    X = []
    for y in yTest:
        X.append((y - intercept) / slope)
        
    return X

(sqftIntercept, sqftSlope) = simpleLinearRegression(housesTrain["sqft_living"], 
    housesTrain["price"])

print(getRegressionPredictions([2650], sqftIntercept, sqftSlope))

print(getResidualSumOfSquares(housesTrain["sqft_living"], 
    housesTrain["price"], sqftIntercept, sqftSlope))

print (inverseRegressionPredictions([800000], sqftIntercept, sqftSlope))

(bedroomsIntercept, bedroomsSlope) = simpleLinearRegression(housesTrain["bedrooms"], 
    housesTrain["price"])
print(getResidualSumOfSquares(housesTest["sqft_living"], housesTest["price"], 
      sqftIntercept, sqftSlope))

print(getResidualSumOfSquares(housesTest["bedrooms"], housesTest["price"], 
      bedroomsIntercept, bedroomsSlope))

    