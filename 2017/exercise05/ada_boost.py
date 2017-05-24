import csv
import numpy as np
import matplotlib.pyplot as plt


data = []

with open( "dataCircle.txt","r" ) as csvfile:
    csvreader = csv.reader( csvfile, delimiter=" " )
    for row in csvreader:
        data = data + row




def data_cleaning( data ):
    arraysize = len(data)
    newData = []
    iterations = 0
    for i in data:
        iterations = iterations + 1
        if i!='':
            newData.append(i)
    return newData

def transform_data_to_matrix(data):
    data = np.array(data,dtype=np.float64)
    data = np.reshape(data,(-1,3))
    return data

def initialize_point_importance( points ):
    count_points = len( points)

def show_plot(class1, class2):
    plt.scatter(class1, class2, c=['red', 'blue'])
    plt.show()


data_circle = transform_data_to_matrix( data_cleaning( data ) )
points = data_circle[:,:-1]
show_plot(points[:,0], points[:,1])
print( initialize_point_importance(points ))
