import math
import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
    den = 1.0 + math.e ** (-1.0 * x)
    d = 1.0 / den
    return d

def gradientDescent(x,y,theta,alpha):

    xTrans = x.transpose()

    for i in range(0,90000):
        hypothesis = np.dot(x,theta)
        hypothesis = sigmoid(hypothesis)
        loss = hypothesis - y
        gradient = np.dot(xTrans,loss)
        theta = theta - alpha * gradient

    return theta

def plotPoints(dataMatrix):
    positive_points = dataMatrix[:40,:-1]
    negative_points = dataMatrix[40:,:-1]
    Xpositive_labeled_points = positive_points[:,0]
    Ypositive_labeled_points = positive_points[:,1]
    Xnegative_labeled_points = negative_points[:,0]
    Ynegative_labeled_points = negative_points[:,1]
    plt.plot(Xnegative_labeled_points,Ynegative_labeled_points,"b^",Xpositive_labeled_points,Ypositive_labeled_points,"ro")
    plt.axis([-15,15,-15,15]);
    plt.show();

def plotPointsWithLines(dataMatrix,theta):
    positive_points = dataMatrix[:40,:-1]
    negative_points = dataMatrix[40:,:-1]
    Xpositive_labeled_points = positive_points[:,0]
    Ypositive_labeled_points = positive_points[:,1]
    Xnegative_labeled_points = negative_points[:,0]
    Ynegative_labeled_points = negative_points[:,1]
    plt.plot(Xnegative_labeled_points,Ynegative_labeled_points,"b^",Xpositive_labeled_points,Ypositive_labeled_points,"ro")
    plt.axis([-15,15,-15,15]);
    xfunction = np.arange(-15,15,0.000001)
    print("!!!!!!!!!!!!!!",theta)
    plt.plot(xfunction,(theta[0]+theta[1]*xfunction)*((-1)/theta[2]))
    plt.show();

def plotPointsWithLines2(dataMatrix,theta,theta2):
    positive_points = dataMatrix[:40,:-1]
    negative_points = dataMatrix[40:,:-1]
    Xpositive_labeled_points = positive_points[:,0]
    Ypositive_labeled_points = positive_points[:,1]
    Xnegative_labeled_points = negative_points[:,0]
    Ynegative_labeled_points = negative_points[:,1]
    plt.plot(Xnegative_labeled_points,Ynegative_labeled_points,"b^",Xpositive_labeled_points,Ypositive_labeled_points,"ro")
    plt.axis([-15,15,-15,15]);
    xfunction = np.arange(-15,15,0.000001)
    plt.plot(xfunction,(theta[0]+theta[1]*xfunction)*((-1)/theta[2]))
    plt.plot(xfunction,(theta2[0]+theta2[1]*xfunction)*((-1)/theta2[2]))
    plt.show();
