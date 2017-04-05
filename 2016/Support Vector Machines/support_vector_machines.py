import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import Exercise4_modules






#print(Exercise4_modules.compute_feature_matrix())

feature_matrix = Exercise4_modules.compute_feature_matrix()
features_positive_matrix = feature_matrix[:30,:]
features_negative_matrix = feature_matrix[30:,:]
yLabel = Exercise4_modules.vectorYLabel()
#print(yLabel)

training_set = feature_matrix[:59,:]
test_set = feature_matrix[feature_matrix.shape[0]-1,:]
print(test_set)


#m,n = yLabel.shape()
#print("M: ",m," N: ",n)


clf = svm.SVC(kernel="sigmoid", C=1,gamma=0.07)
clf.fit(feature_matrix,yLabel)
print(clf.predict([ 110,   60,    3,    6,    3]))
