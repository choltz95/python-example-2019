#!/usr/bin/env python3
# coding: utf-8


import sys
import pickle
import sklearn
import numpy as np
import pandas as pd

def load_sepsis_model():
    with open('nbclf.pkl','rb') as f:
        clf = pickle.load(f)

    return clf

def get_sepsis_score(data_mat, clf):
    # convert d to dataframe from numpy
    varofint = ['HR','O2Sat','Temp','SBP','MAP','DBP']
    d = pd.DataFrame(data=data_mat[:,0:6], columns=varofint)
    interpD = d[varofint].transform(lambda x: x.interpolate(limit=25,limit_direction='both') )
    if data_mat.shape[0] > 3:
        rollsumD = interpD[varofint].rolling(3, min_periods=1).sum().reset_index()
        rollsumD = rollsumD.fillna(0)
        rollvarD = interpD[varofint].rolling(3, min_periods=1).var().reset_index()
        rollvarD = rollvarD.fillna(0)
        rollmaxD = interpD[varofint].rolling(3, min_periods=1).max().reset_index()
        rollmaxD = rollmaxD.fillna(0)
        rollminD = interpD[varofint].rolling(3, min_periods=1).min().reset_index()
        rollminD = rollminD.fillna(0)

        # evaluation by rolling method
        nameL = ['sumHR','sumO2','sumTemp','sumSP','sumMAP','sumDP', 'varHR','varO2','varTemp','varSP','varMAP','varDP','maxHR','maxO2','maxTemp','maxSP','maxMAP','maxDP',         'minHR','minO2','minTemp','minSP','minMAP','minDP']

        sepD = interpD
        rollsumD = sepD[varofint].rolling(3, min_periods=1).sum().reset_index()
        rollsumD = rollsumD.fillna(0)
        rollvarD = sepD[varofint].rolling(3, min_periods=1).var().reset_index()
        rollvarD = rollvarD.fillna(0)
        rollmaxD = sepD[varofint].rolling(3, min_periods=1).max().reset_index()
        rollmaxD = rollmaxD.fillna(0)
        rollminD = sepD[varofint].rolling(3, min_periods=1).min().reset_index()
        rollminD = rollminD.fillna(0)
    else:
        return 0.85, 0
    rowN = sepD.shape[0]
    y = np.zeros(rowN)
    proba = np.zeros(rowN)
    for i in range(rowN):
        featureD = pd.DataFrame(columns=['sumHR','sumO2','sumTemp','sumSP','sumMAP','sumDP', 'varHR','varO2','varTemp','varSP','varMAP','varDP',                            'maxHR','maxO2','maxTemp','maxSP','maxMAP','maxDP',                            'minHR','minO2','minTemp','minSP','minMAP','minDP'])
        if (i+12) < rowN and y[i-1]!=1 and i>0 :
            rollsum = rollsumD[i:i+11]
            rollvar = rollvarD[i:i+11]
            rollmax = rollmaxD[i:i+11]
            rollmin = rollminD[i:i+11]
            valL = list(rollsum[varofint].max()) + list(rollvar[varofint].max()) + list(rollmax[varofint].max()) + list(rollmin[varofint].min())
            new_dict = dict(zip(nameL, valL))
            featureD = featureD.append(new_dict, ignore_index=True)
            prob = clf.predict_proba(featureD)[0][0]
            y[i] = 0 if prob > 0.5 else 1
            proba[i] = prob
        elif (i==0):
            proba[i]=1
            y[i]=0
        elif (i!=0) and (i<12):
            valL = list(rollsumD[varofint].max()) + list(rollvarD[varofint].max()) + list(rollmaxD[varofint].max()) + list(rollminD[varofint].min())
            new_dict = dict(zip(nameL, valL))
            featureD = featureD.append(new_dict, ignore_index=True)
            prob = clf.predict_proba(featureD)[0][0]
            y[i] = 0 if prob > 0.5 else 1
            proba[i] = prob
        else:
            y[i] = y[i-1]
            proba[i] = proba[i-1]
    return proba[0], y[0]
