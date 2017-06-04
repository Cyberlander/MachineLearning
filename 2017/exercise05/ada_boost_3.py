import csv
import numpy as np
import matplotlib.pyplot as plt
import math

class DataPoint:
    """ """
    def __init__( self, xvalue, yvalue, cl, weight ):
        self.x = xvalue
        self.y = yvalue
        self.c = cl
        self.w = weight

    def changeWeight( weight_new ):
        self.w = weight_new



def readAndPrepareData():
    data = []
    with open( "dataCircle.txt", "r" ) as csvfile:
         csvreader = csv.reader( csvfile, delimiter=" " )
         for row in csvreader:
             data = data + row
    arraysize = len( data )
    newData = []
    iterations = 0
    for i in data:
        iterations = iterations + 1
        if i !='':
            newData.append(i)
    # transform data to matrix
    newData = np.array( newData, dtype=np.float64 )
    newData = np.reshape( newData, (-1,3))
    return newData

def plot_data( data, classifier_vertical, classifier_horizontal ):
    class1 = data[:40,:]
    class0 = data[40:,:]
    plt.axis([-15,15,-15,15])
    plt.plot(class1[:,0], class1[:,1], "ro")
    plt.plot(class0[:,0], class0[:,1], "bo")
    for i in classifier_vertical:
        plt.plot((i,i), (-15,15),"k-" )
    for j in classifier_horizontal:
        plt.plot((-15,15), (j,j), "k-" )
    plt.show()

def findMaximaAndMinima( single_column_array ):
    print( "Maximum: " , np.amax(single_column_array) )
    print( "Minimum: ",  np.amin(single_column_array) )

def initialiseWeightVector( data ):
    vector = np.array( [1/len( data )] * len( data ) )
    vector = np.reshape( vector, (-1,1) )
    return vector





def findWeakClassifier( data, found_classifier_vertical, found_classifier_horizontal ):

    min_error_vertical = 1
    min_error_horizontal = 1
    min_error = 1


    min_error_classifier_vertical = 0
    min_error_classifier_horizontal = 0
    min_error_classifier = 0


    for i in range( -8, 10 ):
        if i in found_classifier_vertical:
            continue
        classified_data_vertical = classify_points_vertical( data, i )
        error_tmp = computeError( data, classified_data_vertical )
        print("Error-tmp vertical: ", error_tmp)
        if error_tmp < min_error_vertical:
            min_error_vertical = error_tmp
            min_error_classifier_vertical = i

    for j in range( -8, 10 ):
        if j in found_classifier_horizontal:
            continue
        classified_data_horizontal = classify_points_horizontal( data, j )
        error_tmp = computeError( data, classified_data_horizontal )
        print("Error-tmp horizontal: ", error_tmp)

        if error_tmp < min_error_horizontal:
            min_error_horizontal = error_tmp
            min_error_classifier_horizontal = j

#    if min_error_vertical <= min_error_horizontal:


    print("Min-Error-Vertical: ", min_error_vertical )
    print("Min-Error-Horizontal: ", min_error_horizontal )
    if min_error_vertical <= min_error_horizontal:
        min_error = min_error_vertical
        min_error_classifier = min_error_classifier_vertical
        alpha = computeAlpha(min_error)
        found_classifier_vertical.append( min_error_classifier )
        data = computingNewWeights( data, classify_points_vertical( data, min_error_classifier), alpha)
    else:
        min_error = min_error_horizontal
        min_error_classifier = min_error_classifier_horizontal
        alpha = computeAlpha(min_error)
        found_classifier_horizontal.append( min_error_classifier )
        data = computingNewWeights( data, classify_points_horizontal( data, min_error_classifier), alpha)

    print("Min-Error: ", min_error)
    print("Min-Error-Classifier: ", min_error_classifier)
    print( "Alpha: ", alpha )



    return min_error_classifier, alpha, data, found_classifier_vertical,found_classifier_horizontal



def computingNewWeights(data, classified_data, alpha):
    weight_updates = []
    for i in range(len(data)):
        if data[i][2] != classified_data[i][2]:
            data[i][3] = data[i][3] * math.exp( alpha )
        else:
            data[i][3] = data[i][3] * math.exp( (-1) * alpha )
    zt = np.sum( data[:,3])
    for i in range(len(data)):
        data[i][3]= data[i][3]/zt

    return data





def classify_points_vertical( points, classifier):
    classified_points = np.copy( points )
    for point in classified_points:
        if point[0] <= classifier:
            point[2] = 1
        elif point[0] > classifier:
            point[2] = 0
    return classified_points

def classify_points_horizontal( points, classifier ):
    classified_points = np.copy( points )
    for point in classified_points:
        if point[1] <= classifier:
            point[2] = 1
        elif point[1] > classifier:
            point[2] = 0
    return classified_points













def computeError( data, classified_data ):
    error = 0
    for i in range( len(data) ):
        if data[i][2] != classified_data[i][2]:
            error += data[i][3]
    return error

def computeAlpha( error ):
    part1 = (1 - error)/error
    alpha = (1/2)*math.log( part1 )
    return alpha




# spalte X Max: 9,99208 Min: -9.t79845
# spalte Y Max: 9.97164 Min: -9.97164
data = readAndPrepareData()
weightVector = initialiseWeightVector(data)

data_with_weight = np.concatenate( (data, weightVector),axis=1 )
found_classifier_vertical = []
found_classifier_horizontal = []

count_classifier = 5
classifier=[None]*count_classifier
alpha = [None] * count_classifier
data

for i in range( count_classifier ):
    classifier[i], alpha[i], data, found_classifier_vertical, found_classifier_horizontal =findWeakClassifier( data_with_weight, found_classifier_vertical, found_classifier_horizontal )

"""
classifier1, alpha1, data1, found_classifier_vertical, found_classifier_horizontal = findWeakClassifier( data_with_weight, found_classifier_vertical, found_classifier_horizontal )
classifier2, alpha2, data2, found_classifier_vertical, found_classifier_horizontal = findWeakClassifier( data1, found_classifier_vertical, found_classifier_horizontal )
classifier3, alpha3, data3, found_classifier_vertical, found_classifier_horizontal = findWeakClassifier( data2, found_classifier_vertical, found_classifier_horizontal )
classifier4, alpha4, data4, found_classifier_vertical, found_classifier_horizontal = findWeakClassifier( data3, found_classifier_vertical, found_classifier_horizontal )
classifier5, alpha5, data5, found_classifier_vertical, found_classifier_horizontal = findWeakClassifier( data4, found_classifier_vertical, found_classifier_horizontal )
"""

plot_data(data,found_classifier_vertical, found_classifier_horizontal )

print( "Classifier vertical: ", found_classifier_vertical)
print( "Classifier horizontal: ", found_classifier_horizontal)
for j in range( count_classifier ):
    print( "Classifier: ", classifier[j], " Alpha: ", alpha[j])

data_points = []
