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
import helper

def run_lr(train_data,train_labels,test_data,test_labels,weights):
    print("Running Logistic Regression...")
    clf = LogisticRegression(random_state=0, solver='lbfgs')
    clf = clf.fit(train_data, train_labels)
    preds = clf.predict_proba(test_data)[:,1]
    preds[preds<0.6] = 0
    preds[preds>=0.6] = 1
    print("Logistic Regression:")
    print("Accuracy: ", clf.score(test_data,test_labels))
    print("AMS score: ", helper.calc_ams(weights.astype(np.float32),test_labels,preds))
    print("Precision: ", helper.precision(test_labels,preds))
    print("Recall: ", helper.recall(test_labels,preds))
    return preds

def run_gnb(train_data,train_labels,test_data,test_labels,weights):
    print("Running Gaussian Naive Bayes...")
    clf = GaussianNB()
    clf = clf.fit(train_data, train_labels)
    preds = clf.predict_proba(test_data)[:,1]
    preds[preds<0.6] = 0
    preds[preds>=0.6] = 1
    print("Gaussian Naive Bayes:")
    print("Accuracy: ", clf.score(test_data,test_labels))
    print("AMS score: ", helper.calc_ams(weights.astype(np.float32),test_labels,preds))
    print("Precision: ", helper.precision(test_labels,preds))
    print("Recall: ", helper.recall(test_labels,preds))
    return preds

def run_gradient_boosting(train_data,train_labels,test_data,test_labels,weights):
    print("Running Gradient Boosting...")
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=6, random_state=0)
    clf = clf.fit(train_data, train_labels)
    preds = clf.predict(test_data)
    preds[preds<0.6] = 0
    preds[preds>=0.6] = 1
    print("Gradient Boosting Classifier:")
    print("Accuracy: ", clf.score(test_data,test_labels))
    print("AMS score: ", helper.calc_ams(weights.astype(np.float32),test_labels,preds))
    print("Precision: ", helper.precision(test_labels,preds))
    print("Recall: ", helper.recall(test_labels,preds))
    return preds

def run_decision_tree(train_data,train_labels,test_data,test_labels,weights):
    print("Running Decision Tree...")
    clf = DecisionTreeClassifier(random_state=0)
    clf = clf.fit(train_data, train_labels)
    preds = clf.predict(test_data)
    preds[preds<0.6] = 0
    preds[preds>=0.6] = 1
    print("Decision Tree Classifier:")
    print("Accuracy: ", clf.score(test_data,test_labels))
    print("AMS score: ", helper.calc_ams(weights.astype(np.float32),test_labels,preds))
    print("Precision: ", helper.precision(test_labels,preds))
    print("Recall: ", helper.recall(test_labels,preds))
    return preds

def run_random_forest(train_data,train_labels,test_data,test_labels,weights):
    print("Running Random Forest...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    clf = clf.fit(train_data, train_labels)
    preds = clf.predict(test_data)
    preds[preds<0.6] = 0
    preds[preds>=0.6] = 1
    print("Random Forest Classifier:")
    print("Accuracy: ", clf.score(test_data,test_labels))
    print("AMS score: ", helper.calc_ams(weights.astype(np.float32),test_labels,preds))
    print("Precision: ", helper.precision(test_labels,preds))
    print("Recall: ", helper.recall(test_labels,preds))
    return preds

def run_xgboost(train_data,train_labels,test_data,test_labels,weights):
    print("Running XGBoost...")
    clf = XGBClassifier()
    clf = clf.fit(train_data, train_labels)
    preds = clf.predict(test_data)
    preds[preds<0.6] = 0
    preds[preds>=0.6] = 1
    print("XG Boost Classifier:")
    print("Accuracy: ", clf.score(test_data,test_labels))
    print("AMS score: ", calc_ams(weights.astype(np.float32),test_labels,preds))
    print("Precision: ", precision(test_labels,preds))
    print("Recall: ", recall(test_labels,preds))
    return preds