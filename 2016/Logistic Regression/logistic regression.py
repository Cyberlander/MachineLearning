import csv
import numpy as np
import matplotlib.pyplot as plt
import math

data = []

with open("data.txt","r") as csvfile:
    reader = csv.reader(csvfile,delimiter= " ")
    for row in reader:
        data = data + row


def gradientDescent(x,y,theta,alpha):

    xTrans = x.transpose()

    for i in range(0,90000):
        hypothesis = np.dot(x,theta)
        hypothesis = sigmoid(hypothesis)
        loss = hypothesis - y
        gradient = np.dot(xTrans,loss)
        theta = theta - alpha * gradient

    return theta


def sigmoid(x):
    den = 1.0 + math.e ** (-1.0 * x)
    d = 1.0 / den
    return d


def plotPoints(matrix,theta):
    points = matrix[:,:2]
    positive_labeled_points = points[:50,:]
    Xpositive_labeled_points = positive_labeled_points[:,0]
    Ypositive_labeled_points = positive_labeled_points[:,1]

    negative_labeled_points = points[50:,:]
    Xnegative_labeled_points = negative_labeled_points[:,0]
    Ynegative_labeled_points = negative_labeled_points[:,1]
    plt.plot(Xnegative_labeled_points,Ynegative_labeled_points,"b^",Xpositive_labeled_points,Ypositive_labeled_points,"ro")
    xfunction = np.arange(-3,3,0.000001)
    #plt.plot(x,y,"ro",xfunction,theta[0]+ theta[1]*xfunction+ theta[2]*xfunction**2)
    plt.plot(xfunction,(theta[0]+theta[1]*xfunction)*((-1)/theta[2]))
    plt.show()



numpyArray1D = np.array(data, dtype=np.float64)

matrix = np.reshape(numpyArray1D, (-1,3));
xvalues = matrix[:,:2]
yvalues = matrix[:,2]


xmatrix = np.ones(shape=(100,1))
xvalues = np.concatenate((xmatrix,xvalues),axis=1)
yvalues = matrix[:,2]
theta = np.random.rand(3)
"""
hypothesis = np.dot(xvalues,theta)
hypothesis = sigmoid(hypothesis)
print(hypothesis)
"""
theta = gradientDescent(xvalues,yvalues,theta,0.005)

print(theta)
plotPoints(matrix,theta)
