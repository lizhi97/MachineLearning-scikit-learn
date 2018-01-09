#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 12:52:12 2018

@author: Joe
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
# use the same dataset

#tr_data = read_dataset('tr_server_data.csv')
tr_data = pd.read_csv('tr_server_data.csv') 
 
clf = svm.OneClassSVM(nu=0.05, kernel="rbf", gamma=0.1)
clf.fit(tr_data)
pred = clf.predict(tr_data)

# inliers are labeled 1, outliers are labeled -1

tr_data_m = tr_data.values
normal = tr_data_m[pred == 1]
abnormal = tr_data_m[pred == -1]
#print(abnormal)
plt.figure()
plt.plot(normal[:,0],normal[:,1],'bx')
plt.plot(abnormal[:,0],abnormal[:,1],'ro')
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')
plt.show()
