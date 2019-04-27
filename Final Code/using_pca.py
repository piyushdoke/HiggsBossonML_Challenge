from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from collections import Counter
import pandas as pd
from numpy import genfromtxt
import numpy as np
import csv
import math
from sklearn.metrics import precision_score, recall_score
import models
import helper

if __name__ == "__main__":

	training_data = genfromtxt('training.csv',dtype = str, delimiter=',')
	testing_data = genfromtxt('testing.csv',dtype = str, delimiter=',')

	train_data, train_labels, train_weights = helper.process_data(training_data)
	test_data, test_labels, test_weights = helper.process_data(testing_data)

	train_data = helper.replace_missing_values(train_data)
	test_data = helper.replace_missing_values(test_data)

	train_data = helper.normalize_data(train_data)
	test_data = helper.normalize_data(test_data)

	train_data = helper.run_pca(train_data)
	test_data = helper.run_pca(test_data)

	models.run_lr(train_data,train_labels,test_data,test_labels,test_weights)
	models.run_gnb(train_data,train_labels,test_data,test_labels,test_weights)
	models.run_xgboost(train_data,train_labels,test_data,test_labels,test_weights)
	models.run_gradient_boosting(train_data,train_labels,test_data,test_labels,test_weights)
	models.run_decision_tree(train_data,train_labels,test_data,test_labels,test_weights)
	models.run_random_forest(train_data,train_labels,test_data,test_labels,test_weights)

