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
import models
import helper

def process_data_jet_num(train_data,train_labels,num,weights):
    if num == 0:
        data = train_data[train_data[:,22] == 0]
        data = np.delete(data,[4,5,6,12,22,23,24,25,26,27,28],1)
        labels = train_labels[np.where(train_data[:,22] == 0)]
        weights = weights[np.where(train_data[:,22] == 0)]
    elif num == 1:
        data = train_data[train_data[:,22] == 1]
        data = np.delete(data,[4,5,6,12,26,27,28,22],1)
        labels = train_labels[np.where(train_data[:,22] == 1)]
        weights = weights[np.where(train_data[:,22] == 1)]
    else:
        data_2 = train_data[train_data[:,22] == 2]
        data_2 = np.delete(data_2,[22],1)
        labels_2 = train_labels[np.where(train_data[:,22] == 2)]
        labels_2_weights = weights[np.where(train_data[:,22] == 2)]

        data_3 = train_data[train_data[:,22] == 3]
        data_3 = np.delete(data_3,[22],1)
        labels_3 = train_labels[np.where(train_data[:,22] == 3)]
        labels_3_weights = weights[np.where(train_data[:,22] == 3)]
        
        data = np.concatenate((data_2,data_3),axis=0)
        labels = np.concatenate((labels_2,labels_3),axis=0)
        weights = np.concatenate((labels_2_weights,labels_3_weights),axis=0)
    return data, labels, weights

def merge_jet_num(preds_0, preds_1, preds_2_3, test_0_labels, test_1_labels, test_2_3_labels, test_0_weights, test_1_weights, test_2_3_weights):
    preds = np.concatenate((preds_0,preds_1,preds_2_3))
    labels = np.concatenate((test_0_labels,test_1_labels,test_2_3_labels))
    weights = np.concatenate((test_0_weights,test_1_weights,test_2_3_weights))
    print(preds.shape)
    print(labels.shape)
    print(weights.shape)
    print("jet-num-merged: ")
    print("Accuracy: ", accuracy_score(labels,preds))
    print("AMS score: ", helper.calc_ams(weights.astype(np.float32),labels,preds))
    print("Precision: ", helper.precision(labels,preds))
    print("Recall: ", helper.recall(labels,preds))



if __name__ == "__main__":
	training_data = genfromtxt('training.csv',dtype = str, delimiter=',')
	testing_data = genfromtxt('testing.csv',dtype = str, delimiter=',')

	train_data, train_labels, train_weights = process_data(training_data)
	test_data, test_labels, test_weights = process_data(testing_data)

	data_jet_num_0, data_jet_num_0_labels, data_jet_num_0_weights = process_data_jet_num(train_data,train_labels,0,train_weights)
	data_jet_num_0 = helper.normalize_data(data_jet_num_0)

	data_jet_num_1, data_jet_num_1_labels, data_jet_num_1_weights = process_data_jet_num(train_data,train_labels,1,train_weights)
	data_jet_num_1 = helper.normalize_data(data_jet_num_1)

	data_jet_num_2_3, data_jet_num_2_3_labels, data_jet_num_2_3_weights = process_data_jet_num(train_data,train_labels,2,train_weights)
	data_jet_num_2_3 = helper.normalize_data(data_jet_num_2_3)

	data_jet_num_0_test, data_jet_num_0_test_labels, data_jet_num_0_test_weights = process_data_jet_num(test_data,test_labels,0,test_weights)
	data_jet_num_0_test = helper.normalize_data(data_jet_num_0_test)

	data_jet_num_1_test, data_jet_num_1_test_labels, data_jet_num_1_test_weights = process_data_jet_num(test_data,test_labels,1,test_weights)
	data_jet_num_1_test = helper.normalize_data(data_jet_num_1_test)

	data_jet_num_2_3_test, data_jet_num_2_3_test_labels, data_jet_num_2_3_test_weights = process_data_jet_num(test_data,test_labels,2,test_weights)
	data_jet_num_2_3_test = helper.normalize_data(data_jet_num_2_3_test)

	preds_0 = models.run_lr(data_jet_num_0,data_jet_num_0_labels,data_jet_num_0_test,data_jet_num_0_test_labels,data_jet_num_0_test_weights)
	preds_1 = models.run_lr(data_jet_num_1,data_jet_num_1_labels,data_jet_num_1_test,data_jet_num_1_test_labels,data_jet_num_1_test_weights)
	preds_2_3 = models.run_lr(data_jet_num_2_3,data_jet_num_2_3_labels,data_jet_num_2_3_test,data_jet_num_2_3_test_labels,data_jet_num_2_3_test_weights)
	preds_lr = merge_jet_num(preds_0,preds_1,preds_2_3,data_jet_num_0_test_labels,data_jet_num_1_test_labels,data_jet_num_2_3_test_labels,data_jet_num_0_test_weights,data_jet_num_1_test_weights,data_jet_num_2_3_test_weights)

	preds_0 = models.run_gnb(data_jet_num_0,data_jet_num_0_labels,data_jet_num_0_test,data_jet_num_0_test_labels,data_jet_num_0_test_weights)
	preds_1 = models.run_gnb(data_jet_num_1,data_jet_num_1_labels,data_jet_num_1_test,data_jet_num_1_test_labels,data_jet_num_1_test_weights)
	preds_2_3 = models.run_gnb(data_jet_num_2_3,data_jet_num_2_3_labels,data_jet_num_2_3_test,data_jet_num_2_3_test_labels,data_jet_num_2_3_test_weights)
	preds_gnb = merge_jet_num(preds_0,preds_1,preds_2_3,data_jet_num_0_test_labels,data_jet_num_1_test_labels,data_jet_num_2_3_test_labels,data_jet_num_0_test_weights,data_jet_num_1_test_weights,data_jet_num_2_3_test_weights)

	preds_0 = models.run_gradient_boosting(data_jet_num_0,data_jet_num_0_labels,data_jet_num_0_test,data_jet_num_0_test_labels,data_jet_num_0_test_weights)
	preds_1 = models.run_gradient_boosting(data_jet_num_1,data_jet_num_1_labels,data_jet_num_1_test,data_jet_num_1_test_labels,data_jet_num_1_test_weights)
	preds_2_3 = models.run_gradient_boosting(data_jet_num_2_3,data_jet_num_2_3_labels,data_jet_num_2_3_test,data_jet_num_2_3_test_labels,data_jet_num_2_3_test_weights)
	preds_gb = merge_jet_num(preds_0,preds_1,preds_2_3,data_jet_num_0_test_labels,data_jet_num_1_test_labels,data_jet_num_2_3_test_labels,data_jet_num_0_test_weights,data_jet_num_1_test_weights,data_jet_num_2_3_test_weights)

	preds_0 = models.run_decision_tree(data_jet_num_0,data_jet_num_0_labels,data_jet_num_0_test,data_jet_num_0_test_labels,data_jet_num_0_test_weights)
	preds_1 = models.run_decision_tree(data_jet_num_1,data_jet_num_1_labels,data_jet_num_1_test,data_jet_num_1_test_labels,data_jet_num_1_test_weights)
	preds_2_3 = models.run_decision_tree(data_jet_num_2_3,data_jet_num_2_3_labels,data_jet_num_2_3_test,data_jet_num_2_3_test_labels,data_jet_num_2_3_test_weights)
	preds_dt = merge_jet_num(preds_0,preds_1,preds_2_3,data_jet_num_0_test_labels,data_jet_num_1_test_labels,data_jet_num_2_3_test_labels,data_jet_num_0_test_weights,data_jet_num_1_test_weights,data_jet_num_2_3_test_weights)

	preds_0 = models.run_xgboost(data_jet_num_0,data_jet_num_0_labels,data_jet_num_0_test,data_jet_num_0_test_labels,data_jet_num_0_test_weights)
	preds_1 = models.run_xgboost(data_jet_num_1,data_jet_num_1_labels,data_jet_num_1_test,data_jet_num_1_test_labels,data_jet_num_1_test_weights)
	preds_2_3 = models.run_xgboost(data_jet_num_2_3,data_jet_num_2_3_labels,data_jet_num_2_3_test,data_jet_num_2_3_test_labels,data_jet_num_2_3_test_weights)
	preds_xgb = merge_jet_num(preds_0,preds_1,preds_2_3,data_jet_num_0_test_labels,data_jet_num_1_test_labels,data_jet_num_2_3_test_labels,data_jet_num_0_test_weights,data_jet_num_1_test_weights,data_jet_num_2_3_test_weights)

	preds_0 = models.run_random_forest(data_jet_num_0,data_jet_num_0_labels,data_jet_num_0_test,data_jet_num_0_test_labels,data_jet_num_0_test_weights)
	preds_1 = models.run_random_forest(data_jet_num_1,data_jet_num_1_labels,data_jet_num_1_test,data_jet_num_1_test_labels,data_jet_num_1_test_weights)
	preds_2_3 = models.run_random_forest(data_jet_num_2_3,data_jet_num_2_3_labels,data_jet_num_2_3_test,data_jet_num_2_3_test_labels,data_jet_num_2_3_test_weights)
	preds_rf = merge_jet_num(preds_0,preds_1,preds_2_3,data_jet_num_0_test_labels,data_jet_num_1_test_labels,data_jet_num_2_3_test_labels,data_jet_num_0_test_weights,data_jet_num_1_test_weights,data_jet_num_2_3_test_weights)