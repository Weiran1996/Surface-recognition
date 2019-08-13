import numpy as np 
import pandas as pd 
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import time
import pickle

def RandomForest(features, labels, trn_ratio, save_addr):
	train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=1-trn_ratio, random_state=0)
	clf = RandomForestClassifier(n_estimators=4500, min_samples_leaf=2, max_features='sqrt', oob_score = True, n_jobs=-1, random_state=0)
	clf.fit(train_x, train_y)
	trn_y_pred = clf.predict(train_x)
	tst_y_pred = clf.predict(test_x)
	acc_trn = accuracy_score(train_y, trn_y_pred)
	acc_tst = accuracy_score(test_y, tst_y_pred)

	pickle.dump(clf, open(save_addr, 'wb'))

	return acc_trn, acc_tst, clf.oob_score_

def SVM(features, labels, trn_ratio, save_addr):
	train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=1-trn_ratio, random_state=0)
	clf2 = SVC(C=10**6, gamma='scale').fit(train_x, train_y)
	trn_y_pred = clf2.predict(train_x)
	tst_y_pred = clf2.predict(test_x)
	acc_trn2 = accuracy_score(train_y, trn_y_pred)
	acc_tst2 = accuracy_score(test_y, tst_y_pred)

	pickle.dump(clf2, open(save_addr, 'wb'))

	return acc_trn2, acc_tst2