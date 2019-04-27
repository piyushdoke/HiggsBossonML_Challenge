from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import pandas as pd
from numpy import genfromtxt
import numpy as np
import csv
import math
from sklearn.metrics import precision_score, recall_score

def process_data(t_data):
	labels = t_data[1:,-1]
	labels = (labels == 's').astype(np.int32)
	weights = t_data[1:,31].astype(np.float32)
	data = t_data[1:,1:31]
	data = data.astype(np.float32)
	return data,labels,weights

def normalize_data(data):
    scaler = StandardScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    return data

def replace_missing_values(data):
    a, b = np.where(data == -999)
    new_r, new_c = np.where(data != -999)
    mean_list = np.mean(data[new_r][new_c], axis = 0)
    for i in range(len(a)):
        data[a[i]][data[a[i]] == -999] = mean_list[b[i]]
    x, y = np.where(data == -999)
    print(x,y)
    return data
    
def run_pca(data):
    pca = PCA(n_components=25)
    pca.fit(data)
    data = pca.transform(data)
    return data

def calc_ams(w,y,p):
    w = np.asarray(w)
    y = np.asarray(y)
    p = np.asarray(p)
    y_signal = w * (y == 1)
    y_background = w * (y == 0)
    s = np.sum(y_signal * (p == 1))
    b = np.sum(y_background * (p == 1))
    b_r=10.0
    a = np.sqrt( 2 * ((s + b + b_r) * math.log ( 1.0 + (s / (b + b_r) ) ) - s ))
    return a

def precision(labels, pred):
    precision = precision_score(labels,pred)
    return precision

def recall(labels, pred):
    recall = recall_score(labels,pred)
    return recall