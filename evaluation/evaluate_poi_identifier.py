#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 


from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
print(clf.score(features_test, labels_test))

prediction = clf.predict(features_test)
print(sum(prediction))
print(len(features_test))
print((len(features_test)-sum(prediction))/(1.*len(features_test)))
print(sum(prediction*labels_test))

from sklearn.metrics import precision_score, recall_score

print("precision_score", precision_score(labels_test, prediction))
print("recall_score", recall_score(labels_test, prediction))

predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
import numpy as np
true_positiv = sum(np.multiply(true_labels, predictions))
print("true_positiv", true_positiv)
true_negativ = sum([1 for x, y in zip(true_labels, predictions) if x == 0 and y == 0])
print("true_negativ", true_negativ)
false_positiv = sum([1 for x, y in zip(true_labels, predictions) if x == 0 and y == 1])
print("false_positiv", false_positiv)
false_negativ = sum([1 for x, y in zip(true_labels, predictions) if x == 1 and y == 0])
print("false_negativ", false_negativ)
precision = (1. * true_positiv) / (true_positiv + false_positiv)
print("precision", precision)
recall = (1. * true_positiv) / (true_positiv + false_negativ)
print("recall", recall)