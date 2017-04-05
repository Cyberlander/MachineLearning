import csv
import numpy as np
import matplotlib.pyplot as plt
import Exercise5_modules

data = []
weak_learner = [None] * 10

with open("dataCircle.txt","r") as csvfile:
    csvreader = csv.reader(csvfile, delimiter=" ")
    for row in csvreader:
        data = data + row




def dataCleaning(data):
    arraysize = len(data)
    newData = []
    iterations = 0
    for i in data:
        iterations = iterations + 1
        if i!='':
            newData.append(i)
    return newData


"""
wandelt die Daten, die als eindimensionaler Array kommen,
in eine Matrix um
"""
def transformDataToMatrix(data):
    data = np.array(data,dtype=np.float64)
    data = np.reshape(data,(-1,3))
    return data


def computeAlpha(errorRate):
    ln = np.log((1-errorRate)/errorRate)
    return (1.0/2)*ln


def inititalize_distribution(data_matrix):
    #Tupel Dimensionen der data_matrix
    data_matrix_shape = data_matrix.shape
    #Nummer der Punkte ist erster Wert des Dimensions-Tupels
    number_of_points = data_matrix_shape[0]
    distribution_vector = [None] * number_of_points
    for i in range(0,len(distribution_vector)):
        distribution_vector[i] = 1/number_of_points
    return distribution_vector


def prepare_logistic_regression_input(data_matrix):
    matrix_shape = data_matrix.shape
    first_column = np.reshape(data_matrix[:,0],(-1,1))
    second_column = np.reshape(data_matrix[:,1],(-1,1))
    xmatrix = np.ones(shape=(matrix_shape[0],1))

    xmatrix = np.concatenate((xmatrix,first_column),axis=1)
    xmatrix = np.concatenate((xmatrix,second_column),axis=1)

    yvalue = data_matrix[:,2]

    return xmatrix, yvalue

def ylabel_to_ylabel_AdaBoost(ylabel):
    y_label_AdaBoost = []
    for i in range(0,len(ylabel)):
        if ylabel[i]==1:
            y_label_AdaBoost.append(1)
        elif ylabel[i]==0:
            y_label_AdaBoost.append(-1)
    return y_label_AdaBoost


"""Umwandeln der y-Nullen in Minus-Einsen"""
def changeGivenData(data):
    new_y_ones = np.ones(data.shape[0]/2)
    new_y_minus_ones = np.empty(data.shape[0]/2)
    new_y_minus_ones.fill(-1)
    new_y = np.append(new_y_ones, new_y_minus_ones)
    new_y = np.reshape(new_y,(-1,1))
    data_without_y = data[:,:-1]
    new_matrix = np.concatenate((data_without_y,new_y),axis=1)
    return new_matrix


"""z = theta0 + theta1*x1 + theta2*x2"""
def evaluateClassifier(data_matrix,theta):
    xvalues_shape = data_matrix.shape
    current_z = 0
    evaluation_vector = []
    evaluation_vector_sigmoid = []
    for i in range(0,xvalues_shape[0]):
        #compute z
        current_z = theta[0] + theta[1]*data_matrix[i,0] + theta[2]*data_matrix[i,1]
        evaluation_vector.append(current_z)

    for i in range(0,len(evaluation_vector)):
        sigmoid_value = Exercise5_modules.sigmoid(evaluation_vector[i])
        if (sigmoid_value>0.5):
            evaluation_vector_sigmoid.append(1)
        elif (sigmoid_value<0.5):
            evaluation_vector_sigmoid.append(-1)
    return evaluation_vector_sigmoid



def computeError1AndNewPoints(datamatrix,eval_vector):
    eval_vector = np.reshape(eval_vector, (-1,1))
    xmatrix = np.concatenate((datamatrix,eval_vector),axis=1)
    #print(xmatrix)
    distribution_value = 1/datamatrix.shape[0]
    matrix_for_second_classifier_x1_values = []
    matrix_for_second_classifier_x2_values = []
    matrix_for_second_classifier_y_values = []
    error = 0

    #Sammeln der Koordinaten für Punkte, die falsch klassifiziert wurden
    for i in range(0, datamatrix.shape[0]):
        if (xmatrix[i,2]==1 and xmatrix[i,3]!=1):
                matrix_for_second_classifier_x1_values.append(xmatrix[i,0])
                matrix_for_second_classifier_x2_values.append(xmatrix[i,1])
                matrix_for_second_classifier_y_values.append(xmatrix[i,2])
                error = error + distribution_value
        elif (xmatrix[i,2]==0 and xmatrix[i,3]!=-1):
            matrix_for_second_classifier_x1_values.append(xmatrix[i,0])
            matrix_for_second_classifier_x2_values.append(xmatrix[i,1])
            matrix_for_second_classifier_y_values.append(xmatrix[i,2])
            error = error + distribution_value

    matrix_for_second_classifier_x1_values = np.array(matrix_for_second_classifier_x1_values)
    matrix_for_second_classifier_x1_values = np.reshape(matrix_for_second_classifier_x1_values,(-1,1))
    matrix_for_second_classifier_x2_values = np.array(matrix_for_second_classifier_x2_values)
    matrix_for_second_classifier_x2_values = np.reshape(matrix_for_second_classifier_x2_values,(-1,1))
    matrix_for_second_classifier_y_values = np.array(matrix_for_second_classifier_y_values)
    matrix_for_second_classifier_y_values = np.reshape(matrix_for_second_classifier_y_values,(-1,1))
    matrix_for_second_classifier = np.concatenate((matrix_for_second_classifier_x1_values,matrix_for_second_classifier_x2_values),axis=1)
    matrix_for_second_classifier = np.concatenate((matrix_for_second_classifier,matrix_for_second_classifier_y_values),axis=1)
    return error,matrix_for_second_classifier






"""
1. Schritt  Initialisieren der Verteilung
2. Schritt  Generieren des ersten Weak-Learners
3. Update alpha
4. Update Verteilung
"""





data = dataCleaning(data)
data = transformDataToMatrix(data)

#Datenpunkte von 102 auf 80 reduzieren
data = data[:80,:]
#Nullen in Minus-Einsen


#m-elementiger Vektor mit den initialisierten Distributionen
distribution_vector = inititalize_distribution(data)

# wandelt die Datenmatrix in x und y-Werte um
# zu der x-Matrix wird ein Vektor mit Einsen hinzugefügt



"""first weak learner"""
xvalues,yvalues = prepare_logistic_regression_input(data)
# Inititalisieren vom zufälligem Theta
theta = np.random.rand(3)
theta = Exercise5_modules.gradientDescent(xvalues,yvalues,theta,0.005)
print("Theta 1:",theta)
#Exercise5_modules.plotPointsWithLines(data,theta)

evaluation_vector = evaluateClassifier(data,theta)

error, matrix2 = computeError1AndNewPoints(data,evaluation_vector)
print("Error 1: " + str(error))
print("Alpha 1: " + str(computeAlpha(error)))


#compute Error and new matrix


#print(data)
#print(len(dataCleaning(data)))

"""second weak learner"""
xvalues2,yvalues2 = prepare_logistic_regression_input(matrix2)
theta2 = np.random.rand(3)

theta2 = Exercise5_modules.gradientDescent(xvalues2,yvalues2,theta2,0.005)
print("Theta 2:",theta2)
#print("Matrix 2", matrix2.shape)
#print(matrix2)
evaluation_vector2 = evaluateClassifier(matrix2,theta2)

Exercise5_modules.plotPointsWithLines2(data,theta,theta2)

error2, matrix3 = computeError1AndNewPoints(matrix2,evaluation_vector2)
#print("Error 2: " + str(error2))
#print("Alpha 2: " + str(computeAlpha(error2)))
